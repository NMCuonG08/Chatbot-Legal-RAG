"""#1 — Semantic tool router tests (Flaw 2).

Logic tests use a fake keyword-basis embedding (deterministic, fast) to verify
cosine ranking + threshold + always-include + recall + fallback contract.
One real-embedding test proves the paraphrase case the keyword path misses:
'công ty cho tôi nghỉ' -> severance_pay (keyword 'cho nghỉ' is NOT a substring
of 'cho tôi nghỉ', so the keyword path drops it).
"""
import pytest

import tool_router
from tool_router import select_tools_semantic, _cosine


def _names(tools):
    return {getattr(getattr(t, "metadata", None), "name", "") for t in tools}


# ---- pure-logic ----

def _fake_embed_factory():
    """Bag-of-keywords embedding: dim = keyword index. Deterministic."""
    keywords = [
        "thôi việc", "sa thải", "trợ cấp", "nghỉ", "đuổi",          # severance
        "thuế", "tncn", "thu nhập",                                  # pit
        "trước bạ", "đất", "nhà đất", "lệ phí",                      # land
        "án phí", "tòa",                                            # court
        "tuổi", "sinh năm", "kết hôn",                               # legal_age
        "thừa kế", "di sản",                                         # inheritance
        "disclaimer", "current time", "tavily", "web", "recall", "memory",
    ]
    kw_index = {k: i for i, k in enumerate(keywords)}

    def embed(text):
        text = (text or "").lower()
        vec = [0.0] * len(keywords)
        for k, i in kw_index.items():
            if k in text:
                vec[i] = 1.0
        return vec
    return embed


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert abs(_cosine([1, 1], [1, 1]) - 1.0) < 1e-9


def test_severance_keyword_query_ranks_severance_top():
    """Fake-embed logic test: query sharing keywords with severance desc
    ranks severance into the picked set. (The paraphrase 'nghỉ'≈'thôi việc'
    case is proven by the real-embedding integration test below — a
    bag-of-keywords fake cannot model semantic synonymy by design.)"""
    tool_router.reset_index()
    embed = _fake_embed_factory()
    tools = select_tools_semantic(
        "tính trợ cấp thôi việc khi bị sa thải", get_embedding_fn=embed
    )
    assert tools is not None
    assert "severance_pay_tool" in _names(tools)


def test_pit_query_selects_pit_tool():
    tool_router.reset_index()
    embed = _fake_embed_factory()
    tools = select_tools_semantic("tính thuế thu nhập cá nhân", get_embedding_fn=embed)
    assert "pit_monthly_tool" in _names(tools)


def test_always_include_safety_tools():
    """Disclaimer + current_time + tavily always present even if query
    matches nothing semantically."""
    tool_router.reset_index()
    embed = _fake_embed_factory()
    tools = select_tools_semantic("zzz qqq xxx", get_embedding_fn=embed)
    assert tools is not None
    n = _names(tools)
    assert "legal_disclaimer_tool" in n
    assert "get_current_time" in n
    assert "tavily_search_tool" in n


def test_recall_added_when_history_present():
    tool_router.reset_index()
    embed = _fake_embed_factory()
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    tools = select_tools_semantic("hỏi", history=history, get_embedding_fn=embed)
    assert "recall_user_memory_tool" in _names(tools)


def test_recall_absent_without_history():
    tool_router.reset_index()
    embed = _fake_embed_factory()
    tools = select_tools_semantic("hỏi", get_embedding_fn=embed)
    assert "recall_user_memory_tool" not in _names(tools)


def test_returns_none_when_query_embedding_fails():
    """Embedding service down -> None so agent falls back to keyword path."""
    tool_router.reset_index()

    def boom(text):
        raise RuntimeError("embedding service unavailable")

    tools = select_tools_semantic("anything", get_embedding_fn=boom)
    assert tools is None


def test_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(tool_router, "SEMANTIC_TOOL_ROUTER_ENABLED", False)
    tool_router.reset_index()
    assert select_tools_semantic("x", get_embedding_fn=_fake_embed_factory()) is None


# ---- real embedding (local fallback, no external service) ----

@pytest.mark.integration
def test_real_paraphrase_matches_severance():
    """Real embedding proves the paraphrase the keyword path misses. Local
    sentence-transformer fallback runs offline; marked integration so the
    fast offline gate skips the model-load cost."""
    tool_router.reset_index()
    tools = select_tools_semantic("công ty cho tôi nghỉ thì có được tiền không")
    assert tools is not None
    assert "severance_pay_tool" in _names(tools)