"""Phase 4 unit tests — RLHF feedback store + few-shot injection + rerank boost.

Covers:
- ``save_feedback``: sentinel rejected, good -> MySQL + Qdrant, bad -> MySQL only,
  dedup skip when a similar good answer exists.
- ``find_similar_good``: sentinel -> None, scope-filtered (no cross-user leak),
  returns the top good answer.
- ``generate_node`` few-shot injection: a similar 👍 surfaces a system few-shot
  example in the LLM messages.
- ``_apply_rlhf_rerank_boost``: a chunk whose doc_id backed a 👍 answer is
  up-weighted and re-sorted to the top.
"""
import tasks
import rlhf_store


# ---- fake MySQL session ----

class _FakeDB:
    """Records added AgentFeedback rows; never commits to a real DB."""
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def close(self):
        pass


# ---- save_feedback ----

def _patch_store(monkeypatch, *, existing=None):
    monkeypatch.setattr(rlhf_store, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(rlhf_store, "get_embedding", lambda q: [0.1] * 1024)
    monkeypatch.setattr(rlhf_store, "search_vector",
                        lambda **kw: existing or [])
    added = {"vectors": []}
    monkeypatch.setattr(rlhf_store, "add_vector",
                        lambda collection_name=None, vectors=None, **kw: added["vectors"].extend(vectors.values()))
    return added


def test_save_feedback_rejects_sentinel(monkeypatch):
    added = _patch_store(monkeypatch)
    status = rlhf_store.save_feedback("anonymous", "c1", "m1", "q", "a", [], "good")
    assert status == "rejected_sentinel"
    assert added["vectors"] == []   # nothing written


def test_save_feedback_rejects_invalid_rating(monkeypatch):
    _patch_store(monkeypatch)
    status = rlhf_store.save_feedback("user-1", "c1", "m1", "q", "a", [], "meh")
    assert status == "rejected_sentinel"


def test_save_feedback_good_writes_mysql_and_qdrant(monkeypatch):
    added = _patch_store(monkeypatch)
    status = rlhf_store.save_feedback("user-1", "c1", "m1", "q?", "a!", [], "good")
    assert status == "saved"
    assert len(added["vectors"]) == 1
    payload = added["vectors"][0]["payload"]
    assert payload["rating"] == "good"
    assert payload["scope"] == "user:user-1"   # user-scoped, no leak
    assert payload["question"] == "q?"
    assert payload["response"] == "a!"


def test_save_feedback_bad_mysql_only_no_qdrant(monkeypatch):
    added = _patch_store(monkeypatch)
    status = rlhf_store.save_feedback("user-1", "c1", "m1", "q", "a", [], "bad")
    assert status == "saved"
    assert added["vectors"] == []   # bad -> no Qdrant good-answer pool write


def test_save_feedback_dedup_skips_qdrant(monkeypatch):
    existing = [{"payload": {"question": "q?", "response": "old"}, "score": 0.95}]
    added = _patch_store(monkeypatch, existing=existing)
    status = rlhf_store.save_feedback("user-1", "c1", "m1", "q?", "a!", [], "good")
    assert status == "skipped_duplicate"
    assert added["vectors"] == []   # dedup blocked the Qdrant write


# ---- find_similar_good ----

def test_find_similar_good_sentinel_returns_none(monkeypatch):
    monkeypatch.setattr(rlhf_store, "get_embedding", lambda q: [0.1] * 1024)
    monkeypatch.setattr(rlhf_store, "search_vector", lambda **kw: [{"payload": {}, "score": 1.0}])
    assert rlhf_store.find_similar_good("anonymous", "q") is None
    assert rlhf_store.find_similar_good("", "q") is None


def test_find_similar_good_returns_top(monkeypatch):
    monkeypatch.setattr(rlhf_store, "get_embedding", lambda q: [0.1] * 1024)
    canned = [{
        "payload": {"question": "q1", "response": "a1", "sources": [{"doc_id": 7}]},
        "score": 0.9,
    }]
    monkeypatch.setattr(rlhf_store, "search_vector", lambda **kw: canned)
    good = rlhf_store.find_similar_good("user-1", "similar q1")
    assert good is not None
    assert good["question"] == "q1"
    assert good["response"] == "a1"
    assert good["score"] == 0.9


def test_find_similar_good_filters_by_scope(monkeypatch):
    """User B's lookup must pass B's scope to search_vector (no cross-user leak)."""
    captured = {}
    monkeypatch.setattr(rlhf_store, "get_embedding", lambda q: [0.1] * 1024)
    def fake_search(**kw):
        captured["filters"] = kw.get("filters")
        return []
    monkeypatch.setattr(rlhf_store, "search_vector", fake_search)
    rlhf_store.find_similar_good("user-B", "q")
    assert captured["filters"] == {"scope": "user:user-B"}   # not user-A's scope


# ---- generate_node few-shot injection ----

def _patch_graph_heavy(monkeypatch):
    monkeypatch.setattr(tasks, "detect_route", lambda history, q: "legal_rag")
    monkeypatch.setattr(tasks, "follow_up_question", lambda history, q: q)
    monkeypatch.setattr(tasks, "rewrite_query_to_multi_queries", lambda q, num_queries=3: [q])
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": 0.9}])
    monkeypatch.setattr(tasks, "retrieve_with_multi_query_fallback", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": 0.9}])
    monkeypatch.setattr(tasks, "rerank_documents", lambda docs, q, top_n=5: docs)
    monkeypatch.setattr(tasks, "_llm_judge_relevance", lambda q, docs: {})
    monkeypatch.setattr(tasks, "_retrieve_episodic_context", lambda user_id, q: "")
    monkeypatch.setattr(tasks, "gen_doc_prompt", lambda docs: "CTX")
    monkeypatch.setattr(tasks, "rewrite_query_with_context", lambda q, history: q + "_rw")
    monkeypatch.setattr(tasks, "judge_answer", lambda q, a, s: {"score": 1.0, "rationale": "ok", "verdict": "supported"})
    tasks.guardrails_manager.initialized = False


def test_generate_node_injects_few_shot_from_good_answer(monkeypatch):
    _patch_graph_heavy(monkeypatch)
    captured = {}

    def fake_llm(msgs):
        captured["messages"] = msgs
        return "RAG_ANSWER"
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", fake_llm)
    monkeypatch.setattr(tasks, "find_similar_good",
                        lambda uid, q, **kw: {"question": "q_old", "response": "good_answer_text", "sources": [], "score": 0.9})

    result = tasks.run_chat_graph(history=[], question="q similar", user_id="user-1",
                                  conversation_id="test-fewshot")
    assert result["response"] == "RAG_ANSWER"
    system_content = captured["messages"][0]["content"]
    assert "Ví dụ trả lời tốt" in system_content
    assert "good_answer_text" in system_content


def test_generate_node_no_few_shot_when_no_good_answer(monkeypatch):
    _patch_graph_heavy(monkeypatch)
    captured = {}

    def fake_llm(msgs):
        captured["messages"] = msgs
        return "RAG_ANSWER"
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", fake_llm)
    monkeypatch.setattr(tasks, "find_similar_good", lambda uid, q, **kw: None)

    tasks.run_chat_graph(history=[], question="q", user_id="user-1",
                         conversation_id="test-nofewshot")
    system_content = captured["messages"][0]["content"]
    assert "Ví dụ trả lời tốt" not in system_content


# ---- _apply_rlhf_rerank_boost ----

def test_rerank_boost_upweights_marked_doc():
    ranked = [
        {"doc_id": 1, "content": "a", "relevance_score": 0.6},
        {"doc_id": 2, "content": "b", "relevance_score": 0.8},
        {"doc_id": 3, "content": "c", "relevance_score": 0.5},
    ]
    good_sources = [{"doc_id": 3}]   # doc 3 backed a 👍 answer
    # find_similar_good is called inside the helper (imported into tasks).
    orig = tasks.find_similar_good
    tasks.find_similar_good = lambda uid, q, **kw: {"question": "q", "response": "a", "sources": good_sources, "score": 0.9}
    try:
        out = tasks._apply_rlhf_rerank_boost(ranked, "user-1", "q")
    finally:
        tasks.find_similar_good = orig
    # doc 3 boosted from 0.5 -> 0.5+RLHF_RERANK_BOOST; should now sort near top.
    by_id = {d["doc_id"]: d["relevance_score"] for d in out}
    assert by_id[3] > 0.5
    # re-sorted descending
    scores = [d["relevance_score"] for d in out]
    assert scores == sorted(scores, reverse=True)
    # original list not mutated
    assert ranked[2]["relevance_score"] == 0.5


def test_rerank_boost_noop_without_user():
    ranked = [{"doc_id": 1, "content": "a", "relevance_score": 0.6}]
    out = tasks._apply_rlhf_rerank_boost(ranked, None, "q")
    assert out is ranked