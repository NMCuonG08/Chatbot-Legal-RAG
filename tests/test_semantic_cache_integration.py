"""Integration test for semantic_cache against a REAL Qdrant + REAL Cohere
embedding. Verifies the two things unit tests with a fake client cannot:

  1. ``query_points(score_threshold=...)`` actually FILTERS results by score
     (the old ``search`` API did; the new ``query_points`` API may only annotate
     — if it stops filtering, dissimilar questions would surface wrong cache
     hits). This is the CRITICAL regression guard.
  2. Per-user scope isolation holds through the real Qdrant Filter engine, not
     just our fake interpretation of it.

Skipped automatically when Qdrant is unreachable or COHERE_API_KEY is unset, so
CI without infra still passes. Run manually after Qdrant is up:
    SEMANTIC_CACHE_INTEGRATION=1 python -m pytest tests/test_semantic_cache_integration.py -s

Uses a throwaway temp collection (never touches the real ``semantic_cache``).
"""
import os
import uuid

import pytest

import semantic_cache as sc
from vectorize import get_client


def _qdrant_reachable() -> bool:
    try:
        client = get_client()
        client.get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable() or not os.environ.get("COHERE_API_KEY"),
    reason="Qdrant unreachable or COHERE_API_KEY unset — start Qdrant and set key to run.",
)


@pytest.fixture
def temp_cache_collection(monkeypatch):
    """Run against a throwaway collection so the real cache is never touched."""
    name = f"semantic_cache_itest_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(sc, "CACHE_COLLECTION_NAME", name)
    sc.init_semantic_cache()
    yield name
    try:
        get_client().delete_collection(collection_name=name)
    except Exception:
        pass


def test_score_threshold_filters_dissimilar(temp_cache_collection):
    """CRITICAL: query_points score_threshold must FILTER, not just annotate.

    Insert a cached answer for q1. A query for a totally UNRELATED question must
    return None — if query_points ignored score_threshold and returned the q1
    point with a low score, get_cached_response would wrongly treat it as a hit.
    """
    q1 = "Tôi sinh năm 2004, năm nay tôi bao nhiêu tuổi?"
    ans1 = "Năm nay 2026, bạn 22 tuổi."  # length > 15, not an error string
    sc.set_cached_response(q1, ans1, [], user_id="alice")
    hit = sc.get_cached_response(q1, user_id="alice")
    assert hit is not None
    assert hit["response"] == ans1
    miss = sc.get_cached_response(
        "Quy trình đăng ký thành lập doanh nghiệp TNHH hai thành viên trở lên?",
        user_id="alice",
    )
    assert miss is None, (
        "query_points score_threshold did NOT filter — a dissimilar question "
        "surfaced a cached answer it should not have. Check qdrant-client version."
    )


def test_real_qdrant_user_isolation(temp_cache_collection):
    """Per-user scope isolation through the REAL Qdrant Filter engine."""
    sc.set_cached_response("trợ cấp thôi việc 3 năm lương 15 triệu", "Kết quả riêng của Alice", [],
                           user_id="alice")
    got_alice = sc.get_cached_response("trợ cấp thôi việc 3 năm lương 15 triệu", user_id="alice")
    got_bob = sc.get_cached_response("trợ cấp thôi việc 3 năm lương 15 triệu", user_id="bob")
    assert got_alice is not None and got_alice["response"] == "Kết quả riêng của Alice"
    assert got_bob is None, "Cross-user leak through real Qdrant filter — privacy broken."


def test_real_error_response_not_cached(temp_cache_collection):
    written = sc.set_cached_response(
        "câu hỏi nào đó",
        "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi.",
        [],
        user_id="alice",
    )
    assert written is False
    got = sc.get_cached_response("câu hỏi nào đó", user_id="alice")
    assert got is None