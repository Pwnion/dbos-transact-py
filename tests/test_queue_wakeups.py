"""Queue wake-ups (Pyron fork): a queue's worker thread is woken by a NOTIFY from the
system database's queue triggers, or by this process freeing a slot, so its polling
interval is only a backstop. Every queue here polls far slower than any assertion waits,
so a workflow that starts in time was woken, not polled."""

import threading
import time
import uuid
from typing import Any

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
from dbos._utils import INTERNAL_QUEUE_NAME
from tests.conftest import retry_until_success

# Far longer than any wait below: a workflow that starts within WOKEN_WITHIN was woken.
SLOW_POLL_SEC = 60.0
WOKEN_WITHIN = 5.0
TRIGGERS = ("dbos_queue_wake_v1_insert", "dbos_queue_wake_v1_update")


def wait_for_queue_thread(dbos: DBOS, queue_name: str) -> None:
    """Until the queue's worker thread runs and has made its first poll, so what follows
    is measured against a thread asleep on its wake-up."""

    def ready() -> None:
        assert any(
            t.name == f"queue-worker-{queue_name}" and t.is_alive()
            for t in threading.enumerate()
        )
        assert not dbos._queue_wakeups.event(queue_name).is_set()

    retry_until_success(ready, interval=0.1, max_attempts=100)


def present_triggers(dbos: DBOS) -> set[str]:
    with dbos._sys_db.engine.begin() as c:
        rows = c.execute(
            sa.text(
                "SELECT tgname FROM pg_trigger WHERE tgrelid = to_regclass('dbos.workflow_status') AND tgname = ANY(:names)"
            ),
            {"names": list(TRIGGERS)},
        ).fetchall()
    return {row[0] for row in rows}


def drop_triggers(dbos: DBOS) -> None:
    with dbos._sys_db.engine.begin() as c:
        for name in TRIGGERS:
            c.execute(sa.text(f"DROP TRIGGER IF EXISTS {name} ON dbos.workflow_status"))


def test_triggers_installed_and_restored(dbos: DBOS, skip_with_sqlite: None) -> None:
    assert present_triggers(dbos) == set(TRIGGERS)
    drop_triggers(dbos)
    assert present_triggers(dbos) == set()
    # Launch's migration step reinstalls them, whether or not a numbered migration runs.
    dbos._sys_db.run_migrations()
    assert present_triggers(dbos) == set(TRIGGERS)
    # And leaves them alone when they are there.
    dbos._sys_db.run_migrations()
    assert present_triggers(dbos) == set(TRIGGERS)


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


def test_freed_slot_wakes_queue(dbos: DBOS, skip_with_sqlite: None) -> None:
    """A second workflow waiting behind a one-slot queue starts when the first ends.
    This worker's in-memory running count is what the dequeue reads, so the release of
    its own slot must wake the queue too (the NOTIFY can arrive before it)."""
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


def test_cancel_frees_global_slot(dbos: DBOS, skip_with_sqlite: None) -> None:
    """Cancelling clears queue_name, so the trigger reads the queue from the old row: a
    global slot (counted in the database) frees at once, and the next workflow starts.
    """
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

    cancelled_at = time.monotonic()
    DBOS.cancel_workflow(first_id)
    try:
        assert second.get_result() == "second"
        assert started["second"] - cancelled_at < WOKEN_WITHIN
    finally:
        release_first.set()


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


def test_listener_reconnect_wakes_every_queue(
    dbos: DBOS, mocker: MockerFixture, skip_with_sqlite: None
) -> None:
    """A NOTIFY sent while the listener's connection is down is lost, so reconnecting
    polls every queue once."""

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


def test_destroy_wakes_sleeping_queue_threads(
    dbos: DBOS, skip_with_sqlite: None
) -> None:
    queue_name = f"wake_{uuid.uuid4().hex}"
    DBOS.register_queue(queue_name, polling_interval_sec=SLOW_POLL_SEC)
    wait_for_queue_thread(dbos, queue_name)

    started = time.monotonic()
    DBOS.destroy(destroy_registry=True)
    assert time.monotonic() - started < 5.0
    assert not any(
        t.name == f"queue-worker-{queue_name}" and t.is_alive()
        for t in threading.enumerate()
    )


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
@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_interval_config_rejects_bad_values(
    config: DBOSConfig, key: str, value: float
) -> None:
    DBOS.destroy(destroy_registry=True)
    bad: Any = dict(config)
    bad[key] = value
    with pytest.raises(DBOSInitializationError, match=key):
        DBOS(config=bad)
    DBOS.destroy(destroy_registry=True)
