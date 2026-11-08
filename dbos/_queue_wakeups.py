import threading

# The LISTEN/NOTIFY channel a queue's wake-ups arrive on; the payload is the queue's name.
QUEUES_CHANNEL = "dbos_queues_channel"


class QueueWakeups:
    """Per-queue wake-ups, so a queue's worker thread sleeps until there may be work
    instead of polling on a short clock.

    Each queue's worker thread waits on its queue's event, bounded by the queue's polling
    interval, which is then only a backstop. The event is set by whatever can make a
    queued workflow runnable: a NOTIFY from the system database's queue triggers (a
    workflow enqueued, or a queued workflow leaving PENDING, which frees a slot), the
    notification listener reconnecting (a NOTIFY sent while it was down is lost), and
    this process releasing one of its own workflows' slots (the in-memory running count
    a dequeue reads can lag the NOTIFY of the same workflow's completion).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}

    def event(self, queue_name: str) -> threading.Event:
        """The queue's event, created set, so a newly started worker thread polls at once."""
        with self._lock:
            event = self._events.get(queue_name)
            if event is None:
                event = threading.Event()
                event.set()
                self._events[queue_name] = event
            return event

    def wake(self, queue_name: str) -> None:
        """Wake the queue's worker thread. A queue with no thread in this process is ignored:
        a thread started later polls at once."""
        with self._lock:
            event = self._events.get(queue_name)
        if event is not None:
            event.set()

    def wake_all(self) -> None:
        with self._lock:
            events = list(self._events.values())
        for event in events:
            event.set()
