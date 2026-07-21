"""Tool-selection regression tests.

Covers:
- #3 fallback tool list must be retrieval-first (most legal Q are retrieval,
  not calc). Old fallback returned calc tools (contract_penalty/legal_age/
  inheritance) which are useless for a generic unmatched query.
- keyword path regression: a real calc query still selects its calc tool.
"""
import agent
import agent_tool_wrappers as atw
import tool_router


def _names(tools):
    """Stable name set: prefer metadata.name, else the wrapper var via identity."""
    out = set()
    for t in tools:
        nm = getattr(getattr(t, "metadata", None), "name", None) or getattr(t, "name", "")
        out.add(nm)
    return out


def _keyword_only(monkeypatch):
    """Force the keyword path (disable semantic router) so these tests guard the
    keyword fallback in isolation. The semantic path is covered by
    test_tool_router.py."""
    monkeypatch.setattr(tool_router, "SEMANTIC_TOOL_ROUTER_ENABLED", False)


def test_fallback_is_retrieval_first_not_calc(monkeypatch):
    """Unmatched query ('xin chào') -> fallback must include retrieval tools,
    NOT the calc trio that the old default shipped."""
    _keyword_only(monkeypatch)
    tools = agent.filter_tools_for_query("xin chào")
    names = _names(tools)
    # retrieval tools present
    assert "article_lookup_tool" in names or "article_lookup_func_tool" in names, (
        f"fallback must include article_lookup, got {sorted(names)}"
    )
    assert "unified_legal_search_tool" in names or "unified_legal_search_func_tool" in names, (
        f"fallback must include unified_legal_search, got {sorted(names)}"
    )
    # calc trio absent from fallback
    assert "contract_penalty_calculator" not in names, (
        f"contract_penalty must not be default fallback, got {sorted(names)}"
    )
    assert "inheritance_calculator" not in names
    assert "legal_age_checker" not in names


def test_calc_query_still_matches_keyword_path(monkeypatch):
    """Regression: a real severance query still selects severance_pay_func_tool."""
    _keyword_only(monkeypatch)
    tools = agent.filter_tools_for_query("tính trợ cấp thôi việc lương 15 triệu làm 3 năm")
    names = _names(tools)
    assert "severance_pay_tool" in names or "severance_pay_func_tool" in names, (
        f"severance query must select severance tool, got {sorted(names)}"
    )


def test_fallback_includes_recall_when_history_present(monkeypatch):
    """Fallback with prior history must still attach recall_user_memory."""
    _keyword_only(monkeypatch)
    history = [
        {"role": "user", "content": "tôi tên Nam"},
        {"role": "assistant", "content": "chào Nam"},
        {"role": "user", "content": "xin chào"},
    ]
    tools = agent.filter_tools_for_query("xin chào", history=history)
    names = _names(tools)
    assert "recall_user_memory_tool" in names or "recall_user_memory_func_tool" in names