"""Audit (external 360°) bug 5.2 — Redis fixed-window rate limit on auth.

/auth/login and /auth/register had no rate limiting, so an attacker could
brute-force passwords or flood registrations unbounded. Fix: a Redis fixed-
window limiter — max 5 hits / 60s / IP / action — wired into both endpoints
(429 on the 6th). No new dep; reuses redis.asyncio.

These tests pin the limiter helper against a fake async Redis client.
"""
import pytest

pytestmark = pytest.mark.unit

from rate_limit import AUTH_LIMIT, AUTH_WINDOW_SECONDS, check_rate_limit, rate_limit_key


class _FakeRedis:
    def __init__(self):
        self.store = {}
        self.set_ttls = {}

    async def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    async def expire(self, key, ttl):
        self.set_ttls[key] = ttl
        return True


def test_key_namespaced_per_action_and_ip():
    assert rate_limit_key("login", "1.2.3.4") == "rl:auth:login:1.2.3.4"
    assert rate_limit_key("register", "1.2.3.4") != rate_limit_key("login", "1.2.3.4")


async def test_allows_up_to_limit():
    r = _FakeRedis()
    for _ in range(AUTH_LIMIT):
        assert await check_rate_limit(r, "login", "1.2.3.4") is True


async def test_blocks_over_limit():
    r = _FakeRedis()
    for _ in range(AUTH_LIMIT):
        await check_rate_limit(r, "login", "1.2.3.4")
    assert await check_rate_limit(r, "login", "1.2.3.4") is False


async def test_separate_ips_are_independent():
    r = _FakeRedis()
    for _ in range(AUTH_LIMIT):
        await check_rate_limit(r, "login", "1.1.1.1")
    # a different IP starts fresh
    assert await check_rate_limit(r, "login", "2.2.2.2") is True


async def test_separate_actions_are_independent():
    r = _FakeRedis()
    for _ in range(AUTH_LIMIT):
        await check_rate_limit(r, "login", "1.1.1.1")
    assert await check_rate_limit(r, "register", "1.1.1.1") is True


async def test_missing_ip_is_allowed_and_does_not_touch_redis():
    r = _FakeRedis()
    assert await check_rate_limit(r, "login", None) is True
    assert await check_rate_limit(r, "login", "") is True
    assert r.store == {}, "no IP -> must not write a counter"


async def test_first_hit_sets_window_ttl():
    r = _FakeRedis()
    await check_rate_limit(r, "login", "1.2.3.4")
    key = rate_limit_key("login", "1.2.3.4")
    assert r.set_ttls.get(key) == AUTH_WINDOW_SECONDS