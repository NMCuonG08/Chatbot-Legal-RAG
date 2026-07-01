"""Phase A unit tests for the self-corrective RAG (CRAG) layer.

Tests the pure grading/routing helpers plus the full graph happy-path and
reflection-loop-then-web-fallback path with heavy dependencies monkeypatched.
"""
import tasks
import pytest


# ---- pure helper tests ----

def test_tag_documents_all_above_threshold():
    docs = [{"content": "a", "relevance_score": 0.9}, {"content": "b", "relevance_score": 0.5}]
    graded = tasks._tag_documents_relevance(docs, {})
    assert [d["relevance"] for d in graded] == ["relevant", "relevant"]


def test_tag_documents_all_below_no_verdicts():
    docs = [{"content": "a", "relevance_score": 0.1}, {"content": "b", "relevance_score": 0.2}]
    graded = tasks._tag_documents_relevance(docs, {})
    assert [d["relevance"] for d in graded] == ["irrelevant", "irrelevant"]


def test_tag_documents_mixed_verdict_upgrades_borderline():
    docs = [{"content": "a", "relevance_score": 0.9}, {"content": "b", "relevance_score": 0.1}]
    verdicts = {hash("b"): "relevant"}
    graded = tasks._tag_documents_relevance(docs, verdicts)
    assert [d["relevance"] for d in graded] == ["relevant", "relevant"]


def test_tag_documents_missing_score_uses_verdict():
    docs = [{"content": "a"}]  # no relevance_score
    verdicts = {hash("a"): "irrelevant"}
    graded = tasks._tag_documents_relevance(docs, verdicts)
    assert graded[0]["relevance"] == "irrelevant"


def test_tag_documents_does_not_mutate_input():
    docs = [{"content": "a", "relevance_score": 0.9}]
    tasks._tag_documents_relevance(docs, {})
    assert "relevance" not in docs[0]


def test_decide_after_grade_generate_when_relevant():
    graded = [{"relevance": "relevant"}, {"relevance": "irrelevant"}]
    assert tasks._decide_after_grade(graded, reflection_count=1) == "generate"


def test_decide_after_grade_rewrite_when_all_irrelevant_below_cap():
    graded = [{"relevance": "irrelevant"}]
    assert tasks._decide_after_grade(graded, reflection_count=1) == "rewrite_query"


def test_decide_after_grade_web_search_when_cap_reached():
    graded = [{"relevance": "irrelevant"}]
    assert tasks._decide_after_grade(graded, reflection_count=tasks.REFLECTION_MAX) == "web_search"


# ---- LLM judge parsing ----

def test_llm_judge_relevance_parses_yes_no(monkeypatch):
    def fake_complete(messages):
        return "[0] yes\n[1] no\n"

    monkeypatch.setattr(tasks, "openai_chat_complete", fake_complete)
    docs = [{"content": "good"}, {"content": "bad"}]
    verdicts = tasks._llm_judge_relevance("question", docs)
    assert verdicts == {hash("good"): "relevant", hash("bad"): "irrelevant"}


def test_llm_judge_relevance_empty_docs_returns_empty():
    assert tasks._llm_judge_relevance("q", []) == {}


def test_llm_judge_relevance_llm_failure_returns_empty(monkeypatch):
    def boom(_):
        raise RuntimeError("llm down")
    monkeypatch.setattr(tasks, "openai_chat_complete", boom)
    assert tasks._llm_judge_relevance("q", [{"content": "x"}]) == {}


# ---- full graph paths ----

def _patch_heavy(monkeypatch, *, relevant):
    """Patch routing + retrieval + generation so the graph runs without infra."""
    monkeypatch.setattr(tasks, "detect_route", lambda history, q: "legal_rag")
    monkeypatch.setattr(tasks, "follow_up_question", lambda history, q: q)
    score = 0.9 if relevant else 0.1
    monkeypatch.setattr(tasks, "rewrite_query_to_multi_queries", lambda q, num_queries=3: [q])
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": score}])
    monkeypatch.setattr(tasks, "retrieve_with_multi_query_fallback", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": score}])
    monkeypatch.setattr(tasks, "rerank_documents", lambda docs, q, top_n=5: docs)
    monkeypatch.setattr(tasks, "_llm_judge_relevance", lambda q, docs: {})
    monkeypatch.setattr(tasks, "_retrieve_episodic_context", lambda user_id, q: "")
    monkeypatch.setattr(tasks, "gen_doc_prompt", lambda docs: "CTX")
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", lambda msgs: "RAG_ANSWER")
    monkeypatch.setattr(tasks, "rewrite_query_with_context", lambda q, history: q + "_rewritten")
    # Neutralize guardrails to avoid async/NeMo in unit tests.
    tasks.guardrails_manager.initialized = False


def test_graph_happy_path_relevant_docs_go_to_generate(monkeypatch):
    _patch_heavy(monkeypatch, relevant=True)
    # Unique conversation_id -> unique checkpoint thread_id, so MemorySaver state
    # from other tests does not resume here instead of re-running from START.
    result = tasks.run_chat_graph(history=[], question="q", user_id=None, conversation_id="test-happy")
    assert result["response"] == "RAG_ANSWER"
    assert result["route"] == "legal_rag"
    assert result["sources"] and result["sources"][0]["content"] == "doc1"


def test_graph_all_irrelevant_loops_then_web_search(monkeypatch):
    retrieve_calls = {"n": 0}

    def fake_retrieve(queries, top_k=4):
        retrieve_calls["n"] += 1
        return [{"content": "doc1", "relevance_score": 0.1}]

    _patch_heavy(monkeypatch, relevant=False)
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", fake_retrieve)
    monkeypatch.setattr(tasks, "tavily_search_legal", lambda q, max_results=5: "WEB_RESULTS")
    monkeypatch.setattr(tasks, "openai_chat_complete", lambda msgs: "WEB_ANSWER")

    result = tasks.run_chat_graph(history=[], question="q", user_id=None, conversation_id="test-web")

    assert result["response"] == "WEB_ANSWER"
    # retrieve ran twice: initial + one reflection-loop retry before hitting the cap.
    assert retrieve_calls["n"] == 2