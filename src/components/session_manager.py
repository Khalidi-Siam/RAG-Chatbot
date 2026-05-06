import uuid
from logger import logging

class SessionManager:
    def __init__(self):
        # Only chat history (NOT session validation)
        self.sessions = {}

    def create_memory(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            logging.info(f"Chat memory created: {session_id}")

    def add_message(self, session_id: str, role: str, message: str):
        if session_id not in self.sessions:
            self.create_memory(session_id)

        self.sessions[session_id].append({
            "role": role,
            "message": message
        })

    def get_history(self, session_id: str, last_n: int = 6):
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id][-last_n:]

    def delete_session_memory(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logging.info(f"Memory cleared: {session_id}")


# ✅ Singleton instance (global shared memory)
session_manager = SessionManager()