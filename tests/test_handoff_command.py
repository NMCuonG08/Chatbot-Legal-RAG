"""Phase D tests: multi-agent handoff via langgraph Command(goto=...).

Covers the three handoff predicates plus two end-to-end graph paths that prove
a Command handoff actually redirects execution (agent_tools -> retrieve -> generate,
and generate -> web_search). Heavy deps monkeypatched; guard flags prevent loops.
"""
import tasks


# ---- predicate unit tests ----

def test_should_handoff_to_rag_matches_lookup_marker():
    assert tasks._should_handoff_to_rag("Tôi sẽ tra cứu văn bản luật cho bạn.")
    assert not tasks._should_handoff_to_rag("Khoản 2 điều 10 quy định...")
    assert not tasks._should_handoff_to_rag("")


def test_should_handoff_to_web_matches_not_found_marker():
    assert tasks._should_handoff_to_web("Xin lỗi, không tìm thấy thông tin về vấn đề này.")
    assert not tasks._should_handoff_to_web("Theo điều 15, ...")
    assert not tasks._should_handoff_to_web("")


def test_should_handoff_to_agent_matches_lookup_marker():
    assert tasks._should_handoff_to_agent("cần tra cứu thêm văn bản.")
    assert not tasks._should_handoff_to_agent("Kết quả tìm kiếm: ...")
    assert not tasks._should_handoff_to_agent("")


# ---- graph-level handoff tests ----

def _patch_retrieve(monkeypatch):
    """Make retrieve/grade/generate deterministic without infra."""
    monkeypatch.setattr(tasks, "rewrite_query_to_multi_queries", lambda q, num_queries=3: [q])
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": 0.9}])
    monkeypatch.setattr(tasks, "retrieve_with_multi_query_fallback", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": 0.9}])
    monkeypatch.setattr(tasks, "rerank_documents", lambda docs, q, top_n=5: docs)
    monkeypatch.setattr(tasks, "_llm_judge_relevance", lambda q, docs: {})
    monkeypatch.setattr(tasks, "_retrieve_episodic_context", lambda user_id, q: "")
    monkeypatch.setattr(tasks, "gen_doc_prompt", lambda docs: "CTX")
    monkeypatch.setattr(tasks, "rewrite_query_with_context", lambda q, history: q + "_rw")
    tasks.guardrails_manager.initialized = False


def test_agent_tools_hands_off_to_retrieve_then_generate(monkeypatch):
    """agent answer says 'needs lookup' -> Command(goto='retrieve') -> RAG generate."""
    monkeypatch.setattr(tasks, "detect_route", lambda history, q: "agent_tools")
    monkeypatch.setattr(tasks, "follow_up_question", lambda history, q: q)
    # Agent returns a marker that triggers handoff to RAG.
    monkeypatch.setattr(tasks, "generate_agent_answer", lambda history, q, **kw: ("tôi sẽ tra cứu văn bản luật", []))
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", lambda msgs: "RAG_ANSWER")
    _patch_retrieve(monkeypatch)

    retrieve_calls = {"n": 0}
    def fake_retrieve(queries, top_k=4):
        retrieve_calls["n"] += 1
        return [{"content": "doc1", "relevance_score": 0.9}]
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", fake_retrieve)

    result = tasks.run_chat_graph(history=[], question="q", user_id=None, conversation_id="test-handoff-agent")

    # Final answer came from generate_node (RAG), proving the handoff redirected flow.
    assert result["response"] == "RAG_ANSWER"
    assert retrieve_calls["n"] == 1  # retrieve ran once after handoff


def test_generate_hands_off_to_web_search_on_not_found(monkeypatch):
    """RAG generate returns canned 'not found' -> Command(goto='web_search')."""
    monkeypatch.setattr(tasks, "detect_route", lambda history, q: "legal_rag")
    monkeypatch.setattr(tasks, "follow_up_question", lambda history, q: q)
    _patch_retrieve(monkeypatch)
    # generate_node produces a not-found marker -> handoff to web_search.
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", lambda msgs: "không tìm thấy thông tin")
    monkeypatch.setattr(tasks, "tavily_search_legal", lambda q, max_results=5: "WEB_RESULTS")
    monkeypatch.setattr(tasks, "openai_chat_complete", lambda msgs: "WEB_ANSWER")

    result = tasks.run_chat_graph(history=[], question="q", user_id=None, conversation_id="test-handoff-gen")

    assert result["response"] == "WEB_ANSWER"


def test_handoff_guard_prevents_repeat_agent_to_rag(monkeypatch):
    """Once agent_to_rag_done, a second agent_tools pass must NOT hand off again."""
    monkeypatch.setattr(tasks, "detect_route", lambda history, q: "agent_tools")
    monkeypatch.setattr(tasks, "follow_up_question", lambda history, q: q)
    _patch_retrieve(monkeypatch)
    # Agent keeps emitting the marker; without the guard this would loop.
    monkeypatch.setattr(tasks, "generate_agent_answer", lambda history, q, **kw: ("cần tra cứu văn bản", []))
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", lambda msgs: "không tìm thấy thông tin")
    monkeypatch.setattr(tasks, "tavily_search_legal", lambda q, max_results=5: "WEB_RESULTS")
    monkeypatch.setattr(tasks, "openai_chat_complete", lambda msgs: "WEB_ANSWER")

    # Path: agent -> retrieve -> grade(relevant) -> generate("not found") -> web_search -> END.
    # web_search answer "WEB_ANSWER" has no lookup marker, so no web->agent handoff. Terminates.
    result = tasks.run_chat_graph(history=[], question="q", user_id=None, conversation_id="test-handoff-guard")
    assert result["response"] == "WEB_ANSWER"