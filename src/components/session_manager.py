import uuid
from logger import logging

class SessionManager:
    def __init__(self):
        # { session_id: [ {"role": "user", "message": "..."}, {"role": "assistant", "message": "..."} ] }
        self.sessions = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        logging.info(f"New session created: {session_id}")
        return session_id

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def add_message(self, session_id: str, role: str, message: str):
        if session_id not in self.sessions:
            raise ValueError("Session not found.")

        self.sessions[session_id].append({
            "role": role,
            "message": message
        })
        logging.info(f"[Session {session_id[:8]}...] {role.upper()} message added.")

    def get_history(self, session_id: str, last_n: int = 6) -> list[dict]:
        """
        Return last N messages for context.
        """
        if session_id not in self.sessions:
            raise ValueError("Session not found.")

        return self.sessions[session_id][-last_n:]

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logging.info(f"Session deleted: {session_id}")