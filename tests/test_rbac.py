"""Phase 2 — RBAC tool-policy tests (no DB, pure set logic)."""
from types import SimpleNamespace

from rbac import (
    APPROVAL_EXEMPT_ROLES,
    LEGAL_TOOLS,
    Role,
    SENSITIVE_TOOLS,
    Principal,
    filter_tools_by_policy,
    is_tool_allowed,
    needs_approval,
)


def _tool(name: str):
    return SimpleNamespace(name=name)


def test_principal_admin_is_approval_exempt():
    p = Principal(user_id="u1", username="a", role=Role.ADMIN)
    assert p.is_admin is True
    assert p.is_approval_exempt is True


def test_principal_user_not_exempt():
    p = Principal(user_id="u2", username="b", role=Role.USER)
    assert p.is_admin is False
    assert p.is_approval_exempt is False


def test_filter_tools_admin_keeps_all():
    tools = [_tool(n) for n in ("web_search_tool", "get_current_time", "contract_penalty_calculator")]
    p = Principal(user_id="a", username="admin", role=Role.ADMIN)
    kept = filter_tools_by_policy(tools, p)
    assert {t.name for t in kept} == {t.name for t in tools}


def test_filter_tools_user_drops_web():
    tools = [_tool("web_search_tool"), _tool("contract_penalty_calculator"), _tool("recall_user_memory_tool")]
    p = Principal(user_id="u", username="user", role=Role.USER)
    kept = filter_tools_by_policy(tools, p)
    names = {t.name for t in kept}
    assert "web_search_tool" not in names
    assert "contract_penalty_calculator" in names
    assert "recall_user_memory_tool" in names


def test_filter_tools_guest_readonly_subset():
    tools = [_tool("article_lookup_tool"), _tool("web_search_tool"), _tool("contract_penalty_calculator")]
    p = Principal(user_id="g", username="guest", role=Role.GUEST)
    kept = filter_tools_by_policy(tools, p)
    names = {t.name for t in kept}
    assert names == {"article_lookup_tool"}
    assert "web_search_tool" not in names
    assert "contract_penalty_calculator" not in names


def test_filter_tools_none_principal_guest():
    tools = [_tool("article_lookup_tool"), _tool("web_search_tool")]
    kept = filter_tools_by_policy(tools, None)
    assert {t.name for t in kept} == {"article_lookup_tool"}


def test_is_tool_allowed_role_matrix():
    assert is_tool_allowed("web_search_tool", Role.ADMIN) is True
    assert is_tool_allowed("web_search_tool", Role.LAWYER) is True
    assert is_tool_allowed("web_search_tool", Role.USER) is False
    assert is_tool_allowed("contract_penalty_calculator", Role.USER) is True


def test_needs_approval_sensitive_for_non_exempt():
    assert needs_approval("web_search_tool", Principal("u", "u", Role.USER)) is True
    assert needs_approval("generate_document_template_tool", Principal("u", "u", Role.GUEST)) is True


def test_needs_approval_exempt_roles_skip():
    assert needs_approval("web_search_tool", Principal("a", "a", Role.ADMIN)) is False
    assert needs_approval("web_search_tool", Principal("l", "l", Role.LAWYER)) is False


def test_needs_approval_non_sensitive_never():
    assert needs_approval("get_current_time", Principal("u", "u", Role.USER)) is False
    assert needs_approval("contract_penalty_calculator", None) is False


def test_sensitive_tools_subset_of_legal_or_web():
    assert SENSITIVE_TOOLS.issubset(LEGAL_TOOLS | {"web_search_tool", "tavily_search_tool"})


def test_exempt_roles_are_admin_lawyer():
    assert APPROVAL_EXEMPT_ROLES == {Role.ADMIN, Role.LAWYER}