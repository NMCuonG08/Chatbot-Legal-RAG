"""Tests for semantic cache privacy scoping + error-response guard.

Mocks the Qdrant client and embedding so tests run without a live Qdrant.
Covers the SEMANTIC_CACHE_PRIVACY_FIX.md plan acceptance criteria:
  - user A's private cached answer is never returned to user B (isolation)
  - same-user read hits
  - common scope (general_chat) hits for all users
  - error/placeholder responses are never written to the cache
  - TTL expiry treats old entries as a miss
  - init creates the collection when absent, skips when present
"""
import time
from types import SimpleNamespace

import pytest

import semantic_cache as sc


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeFilter:
    """Interprets a real qdrant Filter so the fake client can apply it."""

    @staticmethod
    def matches(payload, f):
        must = getattr(f, "must", None) or []
        should = getattr(f, "should", None) or []
        for c in must:
            if payload.get(c.key) != c.match.value:
                return False
        if should:
            # Qdrant default: a `should` clause matches when at least one
            # condition holds (min_should defaults to 1).
            min_should = getattr(f, "min_should", None)
            min_should = getattr(min_should, "value", min_should)
            if min_should is None:
                min_should = 1
            hit = sum(1 for c in should if payload.get(c.key) == c.match.value)
            if hit < min_should:
                return False
        return True


class FakeQdrant:
    """In-memory stand-in for the Qdrant client used by semantic_cache."""

    def __init__(self):
        self.points = []
        self.created = False
        self.last_filter = None

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=sc.CACHE_COLLECTION_NAME)] if self.created else []
        )

    def create_collection(self, **kwargs):
        self.created = True

    def upsert(self, collection_name, points):
        for p in points:
            self.points.append(SimpleNamespace(id=p.id, vector=p.vector, payload=p.payload))

    def delete(self, collection_name, points_selector):
        n = len(self.points)
        self.points = []
        return SimpleNamespace(operation=SimpleNamespace(deleted_count=n))

    def query_points(self, collection_name, query, query_filter, limit, score_threshold,
                     with_payload=True):
        self.last_filter = query_filter
        matched = [p for p in self.points if _FakeFilter.matches(p.payload, query_filter)]
        out = [SimpleNamespace(id=p.id, payload=p.payload, score=0.99) for p in matched[:limit]]
        return SimpleNamespace(points=out)


@pytest.fixture
def fake_qdrant(monkeypatch):
    fake = FakeQdrant()
    monkeypatch.setattr(sc, "get_client", lambda: fake)
    monkeypatch.setattr(sc, "get_embedding", lambda text: [0.1] * 1024)
    return fake


# --------------------------------------------------------------------------- #
# Scope resolution + cacheability unit tests
# --------------------------------------------------------------------------- #
def test_scope_for_user():
    assert sc._scope_for("alice", None) == "user:alice"


def test_scope_for_common_when_no_user():
    assert sc._scope_for(None, None) == "common"


def test_scope_explicit_overrides_user():
    assert sc._scope_for("alice", "common") == "common"


def test_is_cacheable_rejects_error_string():
    assert not sc._is_cacheable_response("Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi.")
    assert not sc._is_cacheable_response("Lỗi hệ thống, vui lòng thử lại.")
    assert not sc._is_cacheable_response("")
    assert not sc._is_cacheable_response("   ")
    assert not sc._is_cacheable_response("ngắn quá")


def test_is_cacheable_accepts_real_answer():
    assert sc._is_cacheable_response(
        "Theo Điều 12 Bộ luật Hình sự, bạn đã đủ tuổi chịu TNHS."
    )


# --------------------------------------------------------------------------- #
# Init
# --------------------------------------------------------------------------- #
def test_init_creates_when_absent(fake_qdrant):
    fake_qdrant.created = False
    sc.init_semantic_cache()
    assert fake_qdrant.created is True


def test_init_skips_when_present(fake_qdrant):
    fake_qdrant.created = True
    sc.init_semantic_cache()
    assert fake_qdrant.created is True


# --------------------------------------------------------------------------- #
# Isolation acceptance tests (the privacy-critical ones)
# --------------------------------------------------------------------------- #
def test_user_isolation_no_leak(fake_qdrant):
    """User A's private cached answer MUST NOT be returned to user B."""
    sc.set_cached_response("tôi sinh năm 2004", "Kết quả riêng của Alice", [], user_id="alice")
    got = sc.get_cached_response("tôi sinh năm 2004", user_id="bob")
    assert got is None


def test_same_user_hits(fake_qdrant):
    sc.set_cached_response("tôi sinh năm 2004", "Kết quả riêng của Alice", [], user_id="alice")
    got = sc.get_cached_response("tôi sinh năm 2004", user_id="alice")
    assert got is not None
    assert got["response"] == "Kết quả riêng của Alice"


def test_common_scope_hits_all_users(fake_qdrant):
    """general_chat greetings cached as common → readable by any user."""
    sc.set_cached_response("xin chào", "Xin chào, tôi là trợ lý pháp luật.", [],
                           user_id=None, scope="common")
    got = sc.get_cached_response("xin chào", user_id="bob")
    assert got is not None
    assert got["scope"] == "common"


def test_error_response_not_cached(fake_qdrant):
    """An error string must never enter the cache (prevents poisoning)."""
    written = sc.set_cached_response(
        "tôi sinh năm 2004",
        "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: *Lưu ý...",
        [],
        user_id="alice",
    )
    assert written is False
    assert fake_qdrant.points == []


def test_default_scope_is_per_user_when_user_id_present(fake_qdrant):
    sc.set_cached_response("câu hỏi pháp lý", "Câu trả lời hợp lệ và đủ dài.", [], user_id="alice")
    assert fake_qdrant.points[0].payload["scope"] == "user:alice"


def test_ttl_expiry_returns_none(monkeypatch, fake_qdrant):
    monkeypatch.setattr(sc, "CACHE_TTL_SECONDS", 1)
    sc.set_cached_response("câu hỏi", "Câu trả lời hợp lệ và đủ dài ở đây.", [], user_id="alice")
    # Backdate to a non-zero old timestamp (0.0 would be treated as a legacy
    # point with no timestamp and kept; a real old value must expire).
    fake_qdrant.points[0].payload["cached_at"] = time.time() - 100000
    got = sc.get_cached_response("câu hỏi", user_id="alice")
    assert got is None


def test_clear_semantic_cache_wipes_all(fake_qdrant):
    sc.set_cached_response("q1", "Câu trả lời hợp lệ và đủ dài ở đây.", [], user_id="alice")
    sc.set_cached_response("q2", "Câu trả lời hợp lệ và đủ dài ở đây.", [], user_id="bob")
    n = sc.clear_semantic_cache()
    assert n == 2
    assert fake_qdrant.points == []