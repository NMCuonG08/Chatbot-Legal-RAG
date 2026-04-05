import logging
import os
from urllib.parse import urlparse

import redis

from utils import generate_request_id



def _build_redis_client():
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    parsed_url = urlparse(redis_url)

    host = parsed_url.hostname or "127.0.0.1"
    port = parsed_url.port or 6379
    db = int((parsed_url.path or "/0").lstrip("/") or 0)

    return redis.StrictRedis(host=host, port=port, db=db)


redis_client = _build_redis_client()


def get_conversation_key(bot_id, user_id):
    return f"{bot_id}.{user_id}"


def get_conversation_id(bot_id, user_id, ttl_seconds=360):
    """
    Checks if the key exists in the Redis database.
    If it exists, return True; otherwise, set the value to 1 and return False.
    """
    key = get_conversation_key(bot_id, user_id)
    try:
        if redis_client.exists(key):
            redis_client.expire(key, ttl_seconds)
            return redis_client.get(key).decode("utf-8")
        else:
            conversation_id = generate_request_id()
            redis_client.set(key, conversation_id, ex=ttl_seconds)
            return conversation_id
    except Exception as e:
        # Handle any exceptions (e.g., connection errors)
        logging.exception(f"Get conversation error: {e}")
        # Fallback for local/dev so chat history still gets grouped and persisted.
        return key


def clear_conversation_id(bot_id, user_id):
    key = get_conversation_key(bot_id, user_id)
    try:
        redis_client.delete(key)
        return True
    except Exception as e:
        # Handle any exceptions (e.g., connection errors)
        logging.exception(f"Delte conversation error: {e}")
        return False