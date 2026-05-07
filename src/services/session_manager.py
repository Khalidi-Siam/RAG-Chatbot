from upstash_redis import Redis
from config.settings import settings
import json
from logger import logging


class RedisSessionManager:
    def __init__(self):
        self.redis = Redis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token
        )

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    def create_memory(self, session_id: str, ttl_seconds: int = settings.ttl_seconds):
        key = self._key(session_id)

        if self.redis.exists(key) == 0:
            self.redis.set(key, json.dumps([]))
            self.redis.expire(key, ttl_seconds)
            logging.info(f"Redis memory created: {session_id} with TTL {ttl_seconds}s")

    def add_message(self, session_id: str, role: str, message: str, ttl_seconds: int = settings.ttl_seconds):
        key = self._key(session_id)

        raw = self.redis.get(key)
        messages = json.loads(raw) if raw else []

        messages.append({"role": role, "message": message})

        self.redis.set(key, json.dumps(messages))

        # ✅ IMPORTANT: Refresh TTL on every message (session stays alive if user is active)
        self.redis.expire(key, ttl_seconds)

    def get_history(self, session_id: str, last_n: int = 6, ttl_seconds: int = settings.ttl_seconds):
        key = self._key(session_id)

        raw = self.redis.get(key)
        if not raw:
            return []

        # ✅ OPTIONAL: Refresh TTL on read as well (keeps session alive while user is active)
        self.redis.expire(key, ttl_seconds)

        messages = json.loads(raw)
        return messages[-last_n:]

    def delete_session_memory(self, session_id: str):
        key = self._key(session_id)
        self.redis.delete(key)
        logging.info(f"Redis memory cleared: {session_id}")


# ✅ singleton instance
session_manager = RedisSessionManager()