import threading


class SessionLockManager:
    """
    Keeps a separate lock per session_id.
    Ensures ingestion happens sequentially per session.
    """

    def __init__(self):
        self.locks = {}
        self.global_lock = threading.Lock()

    def get_lock(self, session_id: str) -> threading.Lock:
        with self.global_lock:
            if session_id not in self.locks:
                self.locks[session_id] = threading.Lock()
            return self.locks[session_id]


lock_manager = SessionLockManager()