import hashlib
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, cast

import psycopg
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import DBAPIError

from dbos._migration import (
    ensure_dbos_schema,
    get_migration_versions,
    run_dbos_migrations,
    should_migrate,
)

from ._datasource_migration import DATASOURCE_MIGRATIONS_TABLE
from ._error import DBOSException
from ._logger import dbos_logger
from ._queue_wakeups import QUEUES_CHANNEL
from ._schemas.system_database import SystemSchema
from ._sys_db import (
    SystemDatabase,
    _dbos_notifications_channel,
    _dbos_streams_channel,
    _dbos_workflow_events_channel,
)
from ._utils import quote_identifier

# Serializes migrations, and the queue wake triggers' install, across processes.
MIGRATION_LOCK_ID = 1234567890

# The queue wake-ups' triggers on workflow_status (insert, update, delete) and the marker
# their function carries. Bump the marker with any change to the function or a trigger,
# and launch reinstalls them.
QUEUE_WAKE_TRIGGERS = (
    "dbos_queue_wake_insert",
    "dbos_queue_wake_update",
    "dbos_queue_wake_delete",
)
QUEUE_WAKE_MARKER = "dbos queue wake-ups v1"
QUEUE_WAKE_INSTALL_ATTEMPTS = 5


def _sets_idle_transaction_timeout(url: sa.URL, engine_kwargs: Dict[str, Any]) -> bool:
    """Whether the user already sets idle_in_transaction_session_timeout through libpq options."""
    connect_args = engine_kwargs.get("connect_args") or {}
    sources = [connect_args.get("options"), url.query.get("options")]
    return any(
        "idle_in_transaction_session_timeout" in str(source)
        for source in sources
        if source is not None
    )


class PostgresSystemDatabase(SystemDatabase):
    """PostgreSQL-specific implementation of SystemDatabase."""

    notification_conn: Optional[sa.PoolProxiedConnection] = None

    def _create_engine(
        self, system_database_url: str, engine_kwargs: Dict[str, Any]
    ) -> sa.Engine:
        url = sa.make_url(system_database_url).set(drivername="postgresql+psycopg")
        engine = sa.create_engine(url, **engine_kwargs)
        timeout_ms = self._idle_transaction_timeout_ms
        if timeout_ms is not None and not _sets_idle_transaction_timeout(
            url, engine_kwargs
        ):

            @event.listens_for(engine, "connect")
            def set_idle_transaction_timeout(
                dbapi_conn: Any, connection_record: Any
            ) -> None:
                with dbapi_conn.cursor() as cursor:
                    cursor.execute(
                        f"SET idle_in_transaction_session_timeout = {int(timeout_ms)}"
                    )
                # Committed: the pool rolls a connection back on return, which would undo a bare SET.
                dbapi_conn.commit()

            dbos_logger.debug(
                f"System database idle_in_transaction_session_timeout: {timeout_ms}ms"
            )
        return engine

    def run_migrations(self) -> None:
        """Run PostgreSQL-specific migrations."""
        system_db_url = self.engine.url
        sysdb_name = str(system_db_url.database)
        # Unless we were provided an engine, if the system database does not already exist, create it
        if self.created_engine:
            try:
                engine = sa.create_engine(
                    system_db_url.set(database="postgres"), **self._engine_kwargs
                )
                with engine.connect() as conn:
                    conn.execution_options(isolation_level="AUTOCOMMIT")
                    if not conn.execute(
                        sa.text("SELECT 1 FROM pg_database WHERE datname=:db_name"),
                        parameters={"db_name": sysdb_name},
                    ).scalar():
                        dbos_logger.info(f"Creating system database {sysdb_name}")
                        conn.execute(
                            sa.text(f"CREATE DATABASE {quote_identifier(sysdb_name)}")
                        )
            except Exception:
                dbos_logger.warning(
                    f"Could not connect to postgres database to verify existence of {sysdb_name}. Continuing..."
                )
            finally:
                engine.dispose()
        else:
            # If we were provided an engine, validate it can connect
            with self.engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))

        assert self.schema
        # Skip the advisory lock and migration work entirely if the schema
        # and dbos_migrations table already exist and are at the latest version.
        if not should_migrate(self.engine, self.schema, self.use_listen_notify):
            self._ensure_queue_wake_triggers()
            return

        # Use an advisory lock to serialize concurrent migrations.
        # Try to acquire the lock for up to 30 seconds. If we can't, log a
        # warning and proceed to run migrations without it, rather than
        # hanging startup indefinitely.
        MIGRATION_LOCK_TIMEOUT_SEC = 30
        locked = False
        # Use AUTOCOMMIT so the lock connection does not sit idle-in-transaction
        # while migrations run. Session-level advisory locks are independent of
        # transaction state, so AUTOCOMMIT is safe.
        with self.engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as conn:
            try:
                deadline = time.monotonic() + MIGRATION_LOCK_TIMEOUT_SEC
                while True:
                    got = conn.execute(
                        sa.text("SELECT pg_try_advisory_lock(:lock_id)"),
                        {"lock_id": MIGRATION_LOCK_ID},
                    ).scalar()
                    if got:
                        locked = True
                        break
                    if time.monotonic() >= deadline:
                        dbos_logger.warning(
                            f"Could not acquire migration advisory lock within {MIGRATION_LOCK_TIMEOUT_SEC}s. "
                            f"Attempting migrations without lock."
                        )
                        break
                    time.sleep(1)
                ensure_dbos_schema(self.engine, self.schema)
                run_dbos_migrations(self.engine, self.schema, self.use_listen_notify)
            finally:
                if locked:
                    conn.execute(
                        sa.text("SELECT pg_advisory_unlock(:lock_id)"),
                        {"lock_id": MIGRATION_LOCK_ID},
                    )
        self._ensure_queue_wake_triggers()

    def _queue_wake_sql(self) -> Tuple[str, str, Tuple[str, ...], str]:
        """The queue wake-ups' table, function, triggers and the install SQL."""
        assert self.schema
        schema = quote_identifier(self.schema)
        table = f"{schema}.workflow_status"
        function = f"{schema}.queue_wake_function"
        insert, update, delete = QUEUE_WAKE_TRIGGERS
        # Branch on TG_OP rather than combining tests: NEW is NULL in a DELETE trigger and
        # OLD in an INSERT one, and SQL does not promise to short-circuit.
        install = f"""
CREATE OR REPLACE FUNCTION {function}() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP <> 'DELETE' AND NEW.queue_name IS NOT NULL THEN
        PERFORM pg_notify('{QUEUES_CHANNEL}', NEW.queue_name);
    END IF;
    IF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('{QUEUES_CHANNEL}', OLD.queue_name);
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.queue_name IS NOT NULL AND OLD.queue_name IS DISTINCT FROM NEW.queue_name THEN
            PERFORM pg_notify('{QUEUES_CHANNEL}', OLD.queue_name);
        END IF;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

ALTER FUNCTION {function}() SET search_path = pg_catalog, pg_temp;

CREATE OR REPLACE TRIGGER {insert}
AFTER INSERT ON {table}
FOR EACH ROW WHEN (NEW.queue_name IS NOT NULL AND NEW.status = 'ENQUEUED')
EXECUTE FUNCTION {function}();

CREATE OR REPLACE TRIGGER {update}
AFTER UPDATE OF status, queue_name ON {table}
FOR EACH ROW WHEN (
    (NEW.queue_name IS NOT NULL OR OLD.queue_name IS NOT NULL)
    AND (NEW.status = 'ENQUEUED' OR OLD.status = 'PENDING')
    AND (
        OLD.status IS DISTINCT FROM NEW.status
        OR OLD.queue_name IS DISTINCT FROM NEW.queue_name
    )
)
EXECUTE FUNCTION {function}();

CREATE OR REPLACE TRIGGER {delete}
AFTER DELETE ON {table}
FOR EACH ROW WHEN (OLD.queue_name IS NOT NULL AND OLD.status = 'PENDING')
EXECUTE FUNCTION {function}();

COMMENT ON FUNCTION {function}() IS '{QUEUE_WAKE_MARKER}';
"""
        return table, function, QUEUE_WAKE_TRIGGERS, install

    def _queue_wake_triggers_installed(self, c: sa.Connection) -> bool:
        """Whether this version of the queue wake-ups is installed: every trigger, and
        the function's marker, which a change to any of them bumps."""
        table, function, triggers, _ = self._queue_wake_sql()
        row = c.execute(
            sa.text(
                "SELECT (SELECT count(*) FROM pg_trigger WHERE tgrelid = to_regclass(:table) "
                "AND tgname = ANY(:names)), "
                "obj_description(to_regprocedure(:function), 'pg_proc')"
            ),
            {"table": table, "names": list(triggers), "function": f"{function}()"},
        ).one()
        return bool(row[0] == len(triggers) and row[1] == QUEUE_WAKE_MARKER)

    def _ensure_queue_wake_triggers(self) -> None:
        """Install the triggers that wake queue threads (QueueWakeups) when a row on a queue
        becomes ENQUEUED (enqueued, recovered, resumed, moved between queues, a delay
        passed) or leaves PENDING (completed, failed, cancelled, timed out, deleted: a
        slot is free), naming both queues when a row moves between them.

        Not a numbered migration: this fork's numbers would be taken by upstream's next
        migrations, and an upgrade would skip them. So every launch that migrates checks
        for them and installs them only when missing or of an older marker. CREATE OR
        REPLACE TRIGGER (PostgreSQL 14+) takes SHARE ROW EXCLUSIVE, which waits behind any
        open write to workflow_status and holds back every write after it, so the install
        gives up on the lock after a few seconds and tries again, and gives up for good
        (failing launch) rather than stall the table.
        """
        if not self.use_listen_notify:
            return
        with self.engine.begin() as c:
            if self._queue_wake_triggers_installed(c):
                return
        _, _, _, install = self._queue_wake_sql()
        for attempt in range(QUEUE_WAKE_INSTALL_ATTEMPTS):
            try:
                with self.engine.begin() as c:
                    # Serialized with migrations and every other process's install.
                    locked = c.execute(
                        sa.text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
                        {"lock_id": MIGRATION_LOCK_ID},
                    ).scalar()
                    if locked:
                        if self._queue_wake_triggers_installed(c):
                            return
                        c.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
                        c.execute(sa.text(install))
                        return
            except DBAPIError as e:
                if not isinstance(e.orig, psycopg.errors.LockNotAvailable):
                    raise
            dbos_logger.warning(
                f"Queue wake-up triggers not installed yet (attempt {attempt + 1}): "
                "workflow_status or the migration lock is busy."
            )
            time.sleep(1 + attempt)
        raise DBOSException(
            "Could not install the queue wake-up triggers: workflow_status stayed locked. "
            "Without them every queue waits for its polling interval."
        )

    def verify_migrations(self) -> None:
        """Check the system database is migrated, creating and changing nothing."""
        assert self.schema
        current_version, latest_version = get_migration_versions(
            self.engine, self.schema, self.use_listen_notify
        )
        self._assert_migration_version(current_version, latest_version)
        if self.use_listen_notify:
            with self.engine.begin() as c:
                if not self._queue_wake_triggers_installed(c):
                    dbos_logger.warning(
                        "The queue wake-up triggers are missing or out of date, and this "
                        "process does not migrate: every queue waits for its polling "
                        "interval until a process that migrates launches."
                    )

    @contextmanager
    def retention_lock(self) -> Generator[bool, None, None]:
        """A session-level advisory lock, so only one retention round runs against
        this schema at a time. Session-scoped, so a crashed round releases it."""
        assert self.schema
        # Separate locks for separate schemas. Every SDK derives the key this way -- the
        # leading 8 bytes of SHA-256, big-endian signed -- so rounds in different languages
        # against one system database contend for the same lock.
        key = int.from_bytes(
            hashlib.sha256(f"dbos.retention.{self.schema}".encode()).digest()[:8],
            "big",
            signed=True,
        )
        # The lock is session-scoped, so the round holds this connection until it
        # ends; autocommit keeps the session clear of idle-in-transaction timeouts.
        with self.engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as conn:
            # CockroachDB stubs this out and always returns true, so it collects
            # unprotected rather than not at all.
            acquired = bool(
                conn.execute(
                    sa.text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
                ).scalar()
            )
            try:
                yield acquired
            finally:
                if acquired:
                    # Explicit, since closing only returns the session to the pool.
                    released = conn.execute(
                        sa.text("SELECT pg_advisory_unlock(:key)"), {"key": key}
                    ).scalar()
                    if not released:
                        # False means this session no longer holds it, which a
                        # transaction-pooling proxy causes by switching backends.
                        dbos_logger.warning(
                            "Could not release the retention lock: this session no "
                            "longer holds it. Retention will not proceed until "
                            "the lock is released, which happens when the holding "
                            "backend closes. A transaction-pooling proxy in front of "
                            "Postgres causes this; run DBOS through a session-pooled "
                            "or direct connection."
                        )

    def _vacuum_tables(self, tables: List[str]) -> None:
        assert self.schema
        if self._is_cockroach:
            return

        notices: List[str] = []

        def collect(diag: Any) -> None:
            notices.append(diag.message_primary or "")

        with self.engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as conn:
            # A refused or stalled VACUUM does not raise, it says so in a notice;
            # a successful one is silent, so anything here is worth surfacing.
            raw = conn.connection.dbapi_connection
            add_handler = getattr(raw, "add_notice_handler", None)
            remove_handler = getattr(raw, "remove_notice_handler", None)
            teardown: Optional[Callable[[Any], None]] = None
            if add_handler is not None and remove_handler is not None:
                add_handler(collect)
                teardown = remove_handler
            try:
                for table in tables:
                    # Per table, so one refusal does not skip the rest.
                    del notices[:]
                    try:
                        conn.execute(
                            sa.text(
                                "VACUUM (INDEX_CLEANUP ON, TRUNCATE OFF, ANALYZE) "
                                f"{quote_identifier(self.schema)}.{quote_identifier(table)}"
                            )
                        )
                    except Exception as e:
                        dbos_logger.warning(
                            f"Payload retention could not vacuum {table}: {e}"
                        )
                        continue
                    for notice in notices:
                        dbos_logger.warning(
                            f"Payload retention vacuuming {table}: {notice}"
                        )
            finally:
                # Release the notice handler before returning the connection to the pool
                if teardown is not None:
                    teardown(collect)

    def _cleanup_connections(self) -> None:
        """Clean up PostgreSQL-specific connections."""
        with self._listener_thread_lock:
            if self.notification_conn and self.notification_conn.dbapi_connection:
                self.notification_conn.dbapi_connection.close()
                self.notification_conn.invalidate()

    def _set_statement_timeout(self, conn: sa.Connection, timeout_ms: int) -> None:
        """SET LOCAL, so the cap covers exactly this transaction and is reverted on
        commit: a session-level SET would ride the pooled connection into unrelated
        queries, or leak through a transaction-pooling proxy."""
        # SET takes no bind parameters, so the value is interpolated; it is an int.
        conn.execute(sa.text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))

    def _is_statement_timeout(self, error: DBAPIError) -> bool:
        """57014 is query_canceled, which is how statement_timeout cancels a query.
        psycopg reports it as sqlstate, psycopg2 (which the CockroachDB dialect uses) as pgcode.
        """
        orig = error.orig
        return "57014" in (
            getattr(orig, "sqlstate", None),
            getattr(orig, "pgcode", None),
        )

    def _is_unique_constraint_violation(self, dbapi_error: DBAPIError) -> bool:
        """Check if the error is a unique constraint violation in PostgreSQL."""
        return dbapi_error.orig.sqlstate == "23505"  # type: ignore

    def _is_serialization_error(self, dbapi_error: DBAPIError) -> bool:
        """40001 is serialization_failure (an MVCC conflict), 40P01 deadlock_detected.
        psycopg reports them as sqlstate, psycopg2 (which the CockroachDB dialect
        uses) as pgcode."""
        orig = dbapi_error.orig
        return not {
            getattr(orig, "sqlstate", None),
            getattr(orig, "pgcode", None),
        }.isdisjoint(("40001", "40P01"))

    def _attributes_contains_clause(
        self, attributes: Dict[str, Any]
    ) -> sa.ColumnElement[bool]:
        # Coerce to JSONB so contains() renders the @> containment operator,
        # which is served by the GIN index on the attributes column. The
        # column's generic JSON type would render contains() as a string LIKE.
        return sa.type_coerce(
            SystemSchema.workflow_status.c.attributes, JSONB
        ).contains(attributes)

    def _is_foreign_key_violation(self, dbapi_error: DBAPIError) -> bool:
        """Check if the error is a foreign key violation in PostgreSQL."""
        return dbapi_error.orig.sqlstate == "23503"  # type: ignore

    @staticmethod
    def _truncate_system_database(database_url: str, schema: str) -> None:
        """Empty every DBOS table in the system database, leaving the schema intact.

        Version tables are spared: clearing them would re-run applied migrations."""
        # The plain PostgreSQL dialect cannot connect to CockroachDB at all: it
        # asserts on the server version string. Keep the caller's dialect there.
        url = sa.make_url(database_url)
        if not url.drivername.startswith("cockroachdb"):
            url = url.set(drivername="postgresql+psycopg")
        engine = sa.create_engine(url, connect_args={"connect_timeout": 10})
        try:
            try:
                # Evict other backends, as DROP DATABASE ... WITH (FORCE) does, or one
                # holding a lock blocks the deletes. Its own transaction, and best
                # effort: CockroachDB has no pg_terminate_backend, and a role may not signal.
                with engine.begin() as conn:
                    conn.execute(
                        sa.text(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
                        )
                    )
            except Exception as e:
                dbos_logger.warning(
                    f"Could not evict connections before truncating: {e}"
                )
            with engine.begin() as conn:
                tables = [
                    table
                    for table in conn.execute(
                        sa.text(
                            "SELECT tablename FROM pg_tables WHERE schemaname = :schema"
                        ),
                        {"schema": schema},
                    ).scalars()
                    # A datasource sharing the schema keeps its version too.
                    if table not in ("dbos_migrations", DATASOURCE_MIGRATIONS_TABLE)
                ]
                if not tables:
                    dbos_logger.warning(
                        f'Found no tables to empty in schema "{schema}" of system '
                        f"database {url.database}. If it uses a different schema, "
                        "pass that schema to reset_system_database."
                    )
                # For small test tables, DELETE is far faster than TRUNCATE
                for table in tables:
                    conn.execute(
                        sa.text(
                            f"DELETE FROM {quote_identifier(schema)}.{quote_identifier(table)}"
                        )
                    )
        except Exception as e:
            # Best effort: an absent or unreachable database must not fail the caller.
            dbos_logger.warning(f"Could not empty system database {url.database}: {e}")
        finally:
            engine.dispose()

    @staticmethod
    def _reset_system_database(
        database_url: str, *, truncate: bool = False, schema: Optional[str] = None
    ) -> None:
        """Reset the PostgreSQL system database by dropping it, or by emptying its tables."""
        system_db_url = sa.make_url(database_url)
        sysdb_name = system_db_url.database

        if sysdb_name is None:
            raise ValueError(f"System database name not found in URL {system_db_url}")

        if truncate:
            PostgresSystemDatabase._truncate_system_database(
                database_url, schema if schema else "dbos"
            )
            return

        try:
            # Connect to postgres default database
            engine = sa.create_engine(
                system_db_url.set(database="postgres", drivername="postgresql+psycopg"),
                connect_args={"connect_timeout": 10},
            )

            with engine.connect() as conn:
                # Set autocommit required for database dropping
                conn.execution_options(isolation_level="AUTOCOMMIT")

                # Drop the database
                conn.execute(
                    sa.text(
                        f"DROP DATABASE IF EXISTS {quote_identifier(sysdb_name)} WITH (FORCE)"
                    )
                )
            engine.dispose()
        except Exception as e:
            dbos_logger.error(f"Error resetting PostgreSQL system database: {str(e)}")
            raise e

    def _notification_listener(self) -> None:
        """Listen for PostgreSQL notifications using psycopg."""
        if not self.use_listen_notify:
            return self._notification_listener_polling()
        while self._run_background_processes:
            try:
                with self._listener_thread_lock:
                    self.notification_conn = self.engine.raw_connection()
                    self.notification_conn.detach()
                    psycopg_conn = cast(
                        psycopg.connection.Connection, self.notification_conn
                    )
                    psycopg_conn.set_autocommit(True)

                    psycopg_conn.execute(f"LISTEN {_dbos_notifications_channel}")
                    psycopg_conn.execute(f"LISTEN {_dbos_workflow_events_channel}")
                    psycopg_conn.execute(f"LISTEN {_dbos_streams_channel}")
                    if self.queue_wakeups is not None:
                        psycopg_conn.execute(f"LISTEN {QUEUES_CHANNEL}")
                if self.queue_wakeups is not None:
                    # A NOTIFY sent while this connection was down is lost: poll every queue once.
                    self.queue_wakeups.wake_all()
                while self._run_background_processes:
                    gen = psycopg_conn.notifies()
                    for notify in gen:
                        channel = notify.channel
                        dbos_logger.debug(
                            f"Received notification on channel: {channel}, payload: {notify.payload}"
                        )
                        if channel == _dbos_notifications_channel:
                            if notify.payload:
                                event = self.notifications_map.get(notify.payload)
                                if event is None:
                                    continue
                                event.set()
                                dbos_logger.debug(
                                    f"Signaled notifications event for {notify.payload}"
                                )
                        elif channel == _dbos_workflow_events_channel:
                            if notify.payload:
                                event = self.workflow_events_map.get(notify.payload)
                                if event is None:
                                    continue
                                event.set()
                                dbos_logger.debug(
                                    f"Signaled workflow_events event for {notify.payload}"
                                )
                        elif channel == _dbos_streams_channel:
                            if notify.payload:
                                event = self.streams_map.get(notify.payload)
                                if event is None:
                                    continue
                                event.set()
                                dbos_logger.debug(
                                    f"Signaled streams event for {notify.payload}"
                                )
                        elif channel == QUEUES_CHANNEL:
                            if notify.payload and self.queue_wakeups is not None:
                                self.queue_wakeups.wake(notify.payload)
                        else:
                            dbos_logger.error(f"Unknown channel: {channel}")
            except Exception as e:
                if self._run_background_processes:
                    dbos_logger.warning(f"Notification listener error: {e}")
                    time.sleep(1)
                    # Then the loop will try to reconnect and restart the listener
            finally:
                self._cleanup_connections()

    def _signal_notification(self, channel: str, payload: str) -> None:
        """Accumulate a coalesced wakeup on `channel` for `payload`; run_notifier pushes it off the write path (no-op without L/N)."""
        if not self.use_listen_notify:
            return
        with self._notifier_lock:
            self._pending_notifications.setdefault(channel, set()).add(payload)

    def run_notifier(self) -> None:
        """Periodically flush coalesced notifications across all channels, keeping the async-notify queue lock (which serializes notifying commits) off the write path."""
        if not self.use_listen_notify:
            return
        while self._run_background_processes:
            try:
                time.sleep(self._notification_coalesce_sec)
                self._flush_notifications()
            except Exception as e:
                # Last resort: a failing flush logs and drops its batch internally, so this catches only unexpected errors, which must not kill the sole push path.
                if self._run_background_processes:
                    dbos_logger.warning(f"Notifier error: {e}")
                    time.sleep(1)
        # Final flush so values written just before shutdown still wake readers promptly.
        try:
            self._flush_notifications()
        except Exception as e:
            dbos_logger.warning(f"Notifier final flush error: {e}")

    def _flush_notifications(self) -> None:
        """Emit one coalesced notifying transaction per channel for all pending payloads; drop a channel's batch if it fails."""
        with self._notifier_lock:
            if not any(self._pending_notifications.values()):
                return
            batches = self._pending_notifications
            self._pending_notifications = {}
        # One transaction per channel so an unsendable payload on one channel drops only its own batch.
        for channel, batch in batches.items():
            if not batch:
                continue
            try:
                # One statement, one transaction: one round trip and one acquisition of the async-notify queue lock; unnest emits one notification per payload.
                with self.engine.begin() as c:
                    c.execute(
                        sa.text(
                            "SELECT pg_notify(:channel, p) FROM unnest(CAST(:payloads AS text[])) AS p"
                        ),
                        {"channel": channel, "payloads": list(batch)},
                    )
            except Exception as e:
                # Drop the batch (do not requeue) on failure, e.g. a payload over pg_notify's 8000-byte limit; readers' polling fallback delivers these values, so one poison payload can't stall the notifier forever.
                dbos_logger.warning(f"Notifier flush error on {channel}: {e}")
