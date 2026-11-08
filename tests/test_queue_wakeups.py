"""Queue wake-ups (Pyron fork): a queue's worker thread is woken by a NOTIFY from the
system database's queue triggers, or by this process freeing a slot, so its polling
interval is only a backstop. Every queue here polls far slower than any assertion waits,
so a workflow that starts in time was woken, not polled."""

import threading
import time
import uuid
from typing import Any, Generator

import psycopg
import pytest
import sqlalchemy as sa
from pytest_mock import MockerFixture

from dbos import (
    DBOS,
    DBOSClient,
    DBOSConfig,
    EnqueueOptions,
    SetEnqueueOptions,
    SetWorkflowID,
)
from dbos._error import DBOSInitializationError
from dbos._sys_db_postgres import QUEUE_WAKE_TRIGGERS
from dbos._utils import INTERNAL_QUEUE_NAME
from tests.conftest import retry_until_success

# Far longer than any wait below: a workflow that starts within WOKEN_WITHIN was woken.
SLOW_POLL_SEC = 60.0
WOKEN_WITHIN = 5.0


@pytest.fixture()
def slow_dbos(
    config: DBOSConfig, cleanup_test_databases: None, skip_with_sqlite: None
) -> Generator[DBOS, Any, None]:
    """DBOS with its own clocks slowed too (the supervisor and the internal queue), so
    nothing but a wake-up can start work within WOKEN_WITHIN."""
    DBOS.destroy(destroy_registry=True)
    config["queue_supervisor_interval_sec"] = SLOW_POLL_SEC
    config["internal_queue_polling_interval_sec"] = SLOW_POLL_SEC
    dbos = DBOS(config=config)
    DBOS.launch()
    yield dbos
    DBOS.destroy(destroy_registry=True)


def queue_thread_alive(queue_name: str) -> bool:
    return any(
        t.name == f"queue-worker-{queue_name}" and t.is_alive()
        for t in threading.enumerate()
    )


def wait_for_queue_thread(dbos: DBOS, queue_name: str) -> None:
    """Until the queue's worker thread runs and has made its first poll, so what follows
    is measured against a thread asleep on its wake-up."""

    def ready() -> None:
        assert queue_thread_alive(queue_name)
        assert not dbos._queue_wakeups.event(queue_name).is_set()

    retry_until_success(ready, interval=0.1, max_attempts=100)


def present_triggers(dbos: DBOS) -> set[str]:
    with dbos._sys_db.engine.begin() as c:
        rows = c.execute(
            sa.text(
                "SELECT tgname FROM pg_trigger WHERE tgrelid = to_regclass('dbos.workflow_status') AND tgname = ANY(:names)"
            ),
            {"names": list(QUEUE_WAKE_TRIGGERS)},
        ).fetchall()
    return {row[0] for row in rows}


def drop_triggers(dbos: DBOS) -> None:
    with dbos._sys_db.engine.begin() as c:
        for name in QUEUE_WAKE_TRIGGERS:
            c.execute(sa.text(f"DROP TRIGGER IF EXISTS {name} ON dbos.workflow_status"))


def test_triggers_installed_and_restored(dbos: DBOS, skip_with_sqlite: None) -> None:
    assert present_triggers(dbos) == set(QUEUE_WAKE_TRIGGERS)
    drop_triggers(dbos)
    assert present_triggers(dbos) == set()
    # Launch's migration step reinstalls them, whether or not a numbered migration runs.
    dbos._sys_db.run_migrations()
    assert present_triggers(dbos) == set(QUEUE_WAKE_TRIGGERS)
    # And leaves them alone when they are there.
    dbos._sys_db.run_migrations()
    assert present_triggers(dbos) == set(QUEUE_WAKE_TRIGGERS)


def test_an_older_marker_is_reinstalled(dbos: DBOS, skip_with_sqlite: None) -> None:
    """A changed function or trigger bumps the marker, and launch replaces what an older
    build installed, even though every trigger name is present."""
    with dbos._sys_db.engine.begin() as c:
        c.execute(sa.text("COMMENT ON FUNCTION dbos.queue_wake_function() IS 'older'"))
        assert not dbos._sys_db._queue_wake_triggers_installed(c)  # type: ignore[attr-defined]
    dbos._sys_db.run_migrations()
    with dbos._sys_db.engine.begin() as c:
        assert dbos._sys_db._queue_wake_triggers_installed(c)  # type: ignore[attr-defined]


def test_enqueue_wakes_queue(dbos: DBOS, skip_with_sqlite: None) -> None:
    started: list[float] = []

    @DBOS.workflow()
    def woken() -> str:
        started.append(time.monotonic())
        return "done"

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, polling_interval_sec=SLOW_POLL_SEC)
    wait_for_queue_thread(dbos, queue_name)

    enqueued_at = time.monotonic()
    handle = DBOS.enqueue_workflow(queue_name, woken)
    assert handle.get_result() == "done"
    assert started[0] - enqueued_at < WOKEN_WITHIN


def test_client_enqueue_wakes_queue(
    dbos: DBOS, client: DBOSClient, skip_with_sqlite: None
) -> None:
    """Another process's enqueue (the client's own engine) reaches the worker only
    through the trigger's NOTIFY."""
    started: list[float] = []

    @DBOS.workflow(name="wake_client_target")
    def woken(value: int) -> int:
        started.append(time.monotonic())
        return value * 2

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, polling_interval_sec=SLOW_POLL_SEC)
    wait_for_queue_thread(dbos, queue_name)

    options: EnqueueOptions = {
        "queue_name": queue_name,
        "workflow_name": "wake_client_target",
    }
    enqueued_at = time.monotonic()
    handle = client.enqueue(options, 21)
    assert handle.get_result() == 42
    assert started[0] - enqueued_at < WOKEN_WITHIN


def test_freed_local_slot_wakes_queue(dbos: DBOS, skip_with_sqlite: None) -> None:
    """A second workflow waiting behind this worker's one slot starts when the first
    ends, with the triggers gone: the release of the worker's own slot wakes the queue,
    since a dequeue reads the in-memory running count, which can lag the NOTIFY."""
    release_first = threading.Event()
    started: dict[str, float] = {}

    @DBOS.workflow()
    def one_at_a_time(name: str) -> str:
        started[name] = time.monotonic()
        if name == "first":
            release_first.wait(timeout=30)
        return name

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(
        queue_name, worker_concurrency=1, polling_interval_sec=SLOW_POLL_SEC
    )
    wait_for_queue_thread(dbos, queue_name)

    first = DBOS.enqueue_workflow(queue_name, one_at_a_time, "first")
    retry_until_success(lambda: started["first"], interval=0.05, max_attempts=100)
    second = DBOS.enqueue_workflow(queue_name, one_at_a_time, "second")
    time.sleep(0.5)
    assert "second" not in started
    drop_triggers(dbos)

    released_at = time.monotonic()
    release_first.set()
    assert first.get_result() == "first"
    assert second.get_result() == "second"
    assert started["second"] - released_at < WOKEN_WITHIN


def test_partition_slot_wakes_queue(dbos: DBOS, skip_with_sqlite: None) -> None:
    """One workflow per partition at a time (a conversation's turns): the next in the
    partition starts when the one before it ends."""
    release_first = threading.Event()
    started: dict[str, float] = {}

    @DBOS.workflow()
    def turn(name: str) -> str:
        started[name] = time.monotonic()
        if name == "first":
            release_first.wait(timeout=30)
        return name

    queue_name = f"wake_{uuid.uuid4().hex}"
    queue = DBOS.register_queue(
        queue_name, partition_concurrency=1, polling_interval_sec=SLOW_POLL_SEC
    )
    wait_for_queue_thread(dbos, queue_name)

    with SetEnqueueOptions(queue_partition_key="conversation-1"):
        first = queue.enqueue(turn, "first")
    retry_until_success(lambda: started["first"], interval=0.05, max_attempts=100)
    with SetEnqueueOptions(queue_partition_key="conversation-1"):
        second = queue.enqueue(turn, "second")
    time.sleep(0.5)
    assert "second" not in started

    released_at = time.monotonic()
    release_first.set()
    assert first.get_result() == "first"
    assert second.get_result() == "second"
    assert started["second"] - released_at < WOKEN_WITHIN


def _blocked_global_queue(
    dbos: DBOS,
) -> tuple[str, str, "Any", threading.Event, dict[str, float]]:
    """A queue with one slot across every worker, held by a blocked first workflow, and
    a second workflow waiting behind it."""
    release_first = threading.Event()
    started: dict[str, float] = {}

    @DBOS.workflow()
    def guarded(name: str) -> str:
        started[name] = time.monotonic()
        if name == "first":
            release_first.wait(timeout=30)
        return name

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, concurrency=1, polling_interval_sec=SLOW_POLL_SEC)
    wait_for_queue_thread(dbos, queue_name)

    first_id = str(uuid.uuid4())
    with SetWorkflowID(first_id):
        DBOS.enqueue_workflow(queue_name, guarded, "first")
    retry_until_success(lambda: started["first"], interval=0.05, max_attempts=100)
    second = DBOS.enqueue_workflow(queue_name, guarded, "second")
    time.sleep(0.5)
    assert "second" not in started
    return queue_name, first_id, second, release_first, started


def test_cancel_frees_global_slot(dbos: DBOS, skip_with_sqlite: None) -> None:
    """Cancelling clears queue_name, so the trigger names the queue from the old row: a
    global slot (counted in the database) frees at once, and the next workflow starts.
    """
    _, first_id, second, release_first, started = _blocked_global_queue(dbos)
    cancelled_at = time.monotonic()
    DBOS.cancel_workflow(first_id)
    try:
        assert second.get_result() == "second"
        assert started["second"] - cancelled_at < WOKEN_WITHIN
    finally:
        release_first.set()


def test_deleting_a_running_workflow_frees_global_slot(
    dbos: DBOS, skip_with_sqlite: None
) -> None:
    _, first_id, second, release_first, started = _blocked_global_queue(dbos)
    deleted_at = time.monotonic()
    DBOS.delete_workflow(first_id)
    try:
        assert second.get_result() == "second"
        assert started["second"] - deleted_at < WOKEN_WITHIN
    finally:
        release_first.set()


def test_resume_onto_the_internal_queue_wakes_it(slow_dbos: DBOS) -> None:
    """Resuming a waiting workflow moves it, still ENQUEUED, onto the internal queue: no
    status changes, so only the queue change can announce it."""
    _, first_id, second, release_first, started = _blocked_global_queue(slow_dbos)
    resumed_at = time.monotonic()
    DBOS.resume_workflow(second.workflow_id)
    try:
        assert second.get_result() == "second"
        assert started["second"] - resumed_at < WOKEN_WITHIN
    finally:
        release_first.set()


def test_a_queue_registered_after_launch_starts_at_once(slow_dbos: DBOS) -> None:
    """Registering wakes the supervisor, so the queue's thread does not wait for the
    supervisor's next check."""

    @DBOS.workflow()
    def late() -> str:
        return "late"

    queue_name = f"wake_{uuid.uuid4().hex}"
    registered_at = time.monotonic()
    DBOS.register_queue(queue_name, polling_interval_sec=SLOW_POLL_SEC)
    handle = DBOS.enqueue_workflow(queue_name, late)
    assert handle.get_result() == "late"
    assert time.monotonic() - registered_at < WOKEN_WITHIN


def test_a_failed_poll_is_retried_soon(
    dbos: DBOS, mocker: MockerFixture, skip_with_sqlite: None
) -> None:
    """A poll that fails spent its wake-up, so it is retried after a second, not at the
    backstop."""

    @DBOS.workflow()
    def retried() -> str:
        return "retried"

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, polling_interval_sec=SLOW_POLL_SEC)
    wait_for_queue_thread(dbos, queue_name)

    real = dbos._sys_db.start_queued_workflows
    failures: list[str] = []

    def fail_once(queue: Any, *args: Any, **kwargs: Any) -> list[str]:
        if queue.name == queue_name and not failures:
            failures.append(queue.name)
            raise RuntimeError("the system database went away")
        return real(queue, *args, **kwargs)

    mocker.patch.object(dbos._sys_db, "start_queued_workflows", side_effect=fail_once)
    enqueued_at = time.monotonic()
    handle = DBOS.enqueue_workflow(queue_name, retried)
    assert handle.get_result() == "retried"
    assert failures == [queue_name]
    assert time.monotonic() - enqueued_at < WOKEN_WITHIN


def test_a_superseded_generation_polls_shared_limits_on_its_clock(
    dbos: DBOS, mocker: MockerFixture, skip_with_sqlite: None
) -> None:
    """Once a newer generation serves the same queues, this one polls queues with
    limits shared across workers on a 1 s clock instead of on wake-ups (one NOTIFY would
    wake both into the same dequeue window), so its own work still starts promptly
    rather than at the backstop."""

    @DBOS.workflow()
    def old_generation_work() -> str:
        return "ran"

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(
        queue_name, partition_concurrency=1, polling_interval_sec=SLOW_POLL_SEC
    )
    wait_for_queue_thread(dbos, queue_name)
    dbos._superseded = True
    # Keep the supervisor from reading the real latest version back over the flag.
    mocker.patch.object(
        dbos._sys_db,
        "get_latest_application_version",
        return_value={"version_name": "a-newer-generation"},
    )
    # One more turn of the thread's loop, so its next wait is the clock's.
    dbos._queue_wakeups.wake(queue_name)
    wait_for_queue_thread(dbos, queue_name)
    queue = DBOS.retrieve_queue(queue_name)
    assert queue is not None
    enqueued_at = time.monotonic()
    with SetEnqueueOptions(queue_partition_key="p"):
        handle = queue.enqueue(old_generation_work)
    assert handle.get_result() == "ran"
    assert time.monotonic() - enqueued_at < WOKEN_WITHIN


def test_lost_notification_falls_back_to_poll(
    dbos: DBOS, skip_with_sqlite: None
) -> None:
    """With no trigger to send a NOTIFY, the polling interval still starts the work."""

    @DBOS.workflow()
    def polled() -> str:
        return "polled"

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, polling_interval_sec=1.0)
    wait_for_queue_thread(dbos, queue_name)
    drop_triggers(dbos)

    handle = DBOS.enqueue_workflow(queue_name, polled)
    assert handle.get_result() == "polled"


def test_listener_reconnect_listens_again(
    dbos: DBOS, mocker: MockerFixture, skip_with_sqlite: None
) -> None:
    """A NOTIFY sent while the listener's connection is down is lost, so reconnecting
    polls every queue once; and the new connection listens on the queues channel."""

    @DBOS.workflow()
    def after_reconnect() -> str:
        return "heard"

    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, polling_interval_sec=SLOW_POLL_SEC)
    wait_for_queue_thread(dbos, queue_name)

    def listener_pid() -> int:
        conn = dbos._sys_db.notification_conn
        assert conn is not None
        driver = conn.dbapi_connection
        assert isinstance(driver, psycopg.Connection)
        return driver.info.backend_pid

    pid = retry_until_success(listener_pid, interval=0.1, max_attempts=50)
    wake_all = mocker.spy(dbos._queue_wakeups, "wake_all")
    with dbos._sys_db.engine.begin() as c:
        c.execute(sa.text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
    retry_until_success(lambda: wake_all.assert_called(), interval=0.2, max_attempts=50)
    assert retry_until_success(listener_pid, interval=0.1, max_attempts=50) != pid
    wait_for_queue_thread(dbos, queue_name)

    enqueued_at = time.monotonic()
    handle = DBOS.enqueue_workflow(queue_name, after_reconnect)
    assert handle.get_result() == "heard"
    assert time.monotonic() - enqueued_at < WOKEN_WITHIN


def test_destroy_wakes_sleeping_queue_threads(
    dbos: DBOS, skip_with_sqlite: None
) -> None:
    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, polling_interval_sec=SLOW_POLL_SEC)
    wait_for_queue_thread(dbos, queue_name)

    started = time.monotonic()
    DBOS.destroy(destroy_registry=True)
    assert time.monotonic() - started < 5.0
    assert not queue_thread_alive(queue_name)


def test_interval_config(config: DBOSConfig, cleanup_test_databases: None) -> None:
    DBOS.destroy(destroy_registry=True)
    config["internal_queue_polling_interval_sec"] = 7.5
    config["queue_supervisor_interval_sec"] = 3.0
    dbos = DBOS(config=config)
    DBOS.launch()
    try:
        assert dbos._queue_supervisor_interval_sec == 3.0
        internal = dbos._registry.internal_queue_map[INTERNAL_QUEUE_NAME]
        assert internal._polling_interval_sec == 7.5
    finally:
        DBOS.destroy(destroy_registry=True)


@pytest.mark.parametrize(
    "key", ["internal_queue_polling_interval_sec", "queue_supervisor_interval_sec"]
)
@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), "15", True])
def test_interval_config_rejects_bad_values(
    config: DBOSConfig, key: str, value: Any
) -> None:
    DBOS.destroy(destroy_registry=True)
    bad: Any = dict(config)
    bad[key] = value
    with pytest.raises(DBOSInitializationError, match=key):
        DBOS(config=bad)
    DBOS.destroy(destroy_registry=True)
