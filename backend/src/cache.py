import logging
import os
from urllib.parse import urlparse

import redis

from utils import generate_request_id



from database import settings

def _build_redis_client():
    redis_url = settings.redis_url
    parsed_url = urlparse(redis_url)

    host = parsed_url.hostname or "127.0.0.1"
    port = parsed_url.port or 6379
    db = int((parsed_url.path or "/0").lstrip("/") or 0)

    return redis.StrictRedis(host=host, port=port, db=db)


# Lazy module-level Redis client. Built on first use so importing this module
# does not require a live Redis (useful for tests/eval importers). Tests may
# inject a mock via ``set_redis_client``.
redis_client = None


def get_redis_client():
    """Return the module-level Redis client, building it lazily on first use."""
    global redis_client
    if redis_client is None:
        redis_client = _build_redis_client()
    return redis_client


def set_redis_client(client):
    """Test seam: inject a (mock) Redis client. Pass None to force rebuild."""
    global redis_client
    redis_client = client


def get_conversation_key(bot_id, user_id):
    return f"{bot_id}.{user_id}"


def get_conversation_id(bot_id, user_id, ttl_seconds=360):
    """
    Checks if the key exists in the Redis database.
    If it exists, return True; otherwise, set the value to 1 and return False.
    """
    key = get_conversation_key(bot_id, user_id)
    try:
        rc = get_redis_client()
        if rc.exists(key):
            rc.expire(key, ttl_seconds)
            return rc.get(key).decode("utf-8")
        else:
            conversation_id = generate_request_id()
            rc.set(key, conversation_id, ex=ttl_seconds)
            return conversation_id
    except Exception as e:
        # Handle any exceptions (e.g., connection errors)
        logging.exception(f"Get conversation error: {e}")
        # Fallback for local/dev so chat history still gets grouped and persisted.
        return key


def clear_conversation_id(bot_id, user_id):
    key = get_conversation_key(bot_id, user_id)
    try:
        get_redis_client().delete(key)
        return True
    except Exception as e:
        # Handle any exceptions (e.g., connection errors)
        logging.exception(f"Delte conversation error: {e}")
        return False