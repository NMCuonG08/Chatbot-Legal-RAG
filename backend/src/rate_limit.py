"""Redis fixed-window rate limiter for auth endpoints (audit 5.2).

/auth/login and /auth/register are public and previously unthrottled, so an
attacker could brute-force passwords or flood registrations without bound.
This module implements a small fixed-window limiter (max ``AUTH_LIMIT`` hits
per ``AUTH_WINDOW_SECONDS`` per IP per action) on top of redis.asyncio — no
new dependency. The caller passes the redis client (DI) so tests can inject a
fake.

The limiter is intentionally per-IP, not per-username: a per-username counter
would let an attacker lock out legitimate users (DoS-as-weapon), and does not
slow a distributed credential-stuffing sweep that rotates usernames. Per-IP
throttles the source of the brute force without an account-lockout footgun.
"""
import logging

logger = logging.getLogger(__name__)

AUTH_LIMIT = 5
AUTH_WINDOW_SECONDS = 60


def rate_limit_key(action: str, ip: str) -> str:
    """Namespaced Redis key for one (action, IP) counter."""
    return f"rl:auth:{action}:{ip}"


async def check_rate_limit(
    client,
    action: str,
    ip: str | None,
    limit: int = AUTH_LIMIT,
    window: int = AUTH_WINDOW_SECONDS,
) -> bool:
    """Increment the (action, IP) counter and return whether the request is
    allowed (True) or over the limit (False).

    On the first hit the counter is given a TTL of ``window`` seconds so the
    window self-resets without a background job. A missing/empty IP is allowed
    and does NOT touch Redis (we cannot bucket an unknown client, and silent
    bypass is safer than denying legitimate proxy-stripped requests).

    Redis errors fail OPEN (allowed): auth availability is more important than
    throttling precision — a degraded Redis must not lock out all logins.
    """
    if not ip:
        return True
    key = rate_limit_key(action, ip)
    try:
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, window)
        return count <= limit
    except Exception as exc:  # Redis down / network error -> fail open
        logger.warning("rate_limit check failed (allowing): %s", exc)
        return True