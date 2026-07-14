"""Phase 2 — tool-approval workflow tests (sqlite in-memory)."""
import uuid

import pytest

from approval import (
    APPROVAL_PREFIX,
    await_approval_response,
    decide_approval,
    evaluate_tool_gate,
    fetch_pending,
    get_approval,
    is_tool_allowed,
    mark_tool_allowed,
    request_approval,
)
from rbac import Principal, Role


def test_request_approval_creates_pending(sqlite_db):
    ap = request_approval("u1", "web_search_tool", {"q": "x"}, run_id="r1")
    assert ap.status == "pending"
    assert ap.tool_name == "web_search_tool"
    assert ap.run_id == "r1"
    assert ap.args_json == {"q": "x"}


def test_fetch_pending_lists_only_pending(sqlite_db):
    a1 = request_approval("u1", "web_search_tool", {}, run_id="r1")
    request_approval("u1", "tavily_search_tool", {}, run_id="r2")
    decide_approval(a1.id, "approved", decided_by="admin1")
    pending = fetch_pending(limit=10)
    assert len(pending) == 1
    assert pending[0]["tool_name"] == "tavily_search_tool"


def test_decide_approval_approve_marks_allowed(sqlite_db):
    ap = request_approval("u1", "web_search_tool", {}, run_id="run-x")
    res = decide_approval(ap.id, "approved", decided_by="admin1", note="ok")
    assert res.status == "approved"
    assert res.decided_by == "admin1"
    assert is_tool_allowed("run-x", "web_search_tool") is True


def test_decide_approval_reject_not_allowed(sqlite_db):
    ap = request_approval("u1", "web_search_tool", {}, run_id="run-y")
    res = decide_approval(ap.id, "rejected", decided_by="admin1")
    assert res.status == "rejected"
    assert is_tool_allowed("run-y", "web_search_tool") is False


def test_decide_approval_idempotent(sqlite_db):
    ap = request_approval("u1", "web_search_tool", {}, run_id="run-z")
    first = decide_approval(ap.id, "approved", decided_by="admin1")
    second = decide_approval(ap.id, "rejected", decided_by="admin2")
    assert second.status == "approved"
    assert second.decided_by == "admin1"


def test_decide_invalid_decision_raises(sqlite_db):
    ap = request_approval("u1", "web_search_tool", {}, run_id="r")
    with pytest.raises(ValueError):
        decide_approval(ap.id, "maybe", decided_by="admin1")


def test_decide_unknown_approval_returns_none(sqlite_db):
    assert decide_approval("no-such-id", "approved", decided_by="admin1") is None
    assert get_approval("no-such-id") is None


def test_mark_tool_allowed_then_check():
    mark_tool_allowed("run-a", "get_current_time")
    assert is_tool_allowed("run-a", "get_current_time") is True
    assert is_tool_allowed("run-a", "web_search_tool") is False


def test_approval_prefix_constant():
    assert APPROVAL_PREFIX.startswith("[CẦN PHÊ DUYỆT")


# ---- Phase 2b: pre-flight tool gate ----


def test_gate_exempt_role_proceeds(sqlite_db):
    d, a = evaluate_tool_gate(
        Principal("a", "a", Role.ADMIN), ["web_search_tool", "generate_document_template_tool"], uuid.uuid4().hex
    )
    assert d == "proceed" and a is None
    d, a = evaluate_tool_gate(
        Principal("l", "l", Role.LAWYER), ["web_search_tool"], uuid.uuid4().hex
    )
    assert d == "proceed" and a is None


def test_gate_none_principal_proceeds(sqlite_db):
    d, a = evaluate_tool_gate(None, ["web_search_tool"], uuid.uuid4().hex)
    assert d == "proceed" and a is None


def test_gate_user_sensitive_blocks(sqlite_db):
    run_id = uuid.uuid4().hex
    d, a = evaluate_tool_gate(
        Principal("u", "u", Role.USER),
        ["contract_penalty_calculator", "generate_document_template_tool"],
        run_id,
    )
    assert d == "await_approval"
    assert a is not None
    assert a.status == "pending"
    assert a.tool_name == "generate_document_template_tool"
    assert a.run_id == run_id


def test_gate_user_non_sensitive_proceeds(sqlite_db):
    d, a = evaluate_tool_gate(
        Principal("u", "u", Role.USER),
        ["contract_penalty_calculator", "get_current_time"],
        uuid.uuid4().hex,
    )
    assert d == "proceed" and a is None


def test_gate_skips_already_approved(sqlite_db):
    # First call blocks + creates a pending approval; admin approves it; the
    # re-run (client re-post) must proceed because the tool is now allowed.
    # Unique run_id avoids stale Redis allowed-set from prior sessions (TTL 1h).
    run_id = uuid.uuid4().hex
    d, a = evaluate_tool_gate(
        Principal("u", "u", Role.USER), ["generate_document_template_tool"], run_id
    )
    assert d == "await_approval"
    decide_approval(a.id, "approved", decided_by="admin1")
    d2, a2 = evaluate_tool_gate(
        Principal("u", "u", Role.USER), ["generate_document_template_tool"], run_id
    )
    assert d2 == "proceed" and a2 is None


def test_gate_surfaces_first_blocking_tool_only(sqlite_db):
    d, a = evaluate_tool_gate(
        Principal("u", "u", Role.USER),
        ["web_search_tool", "generate_document_template_tool"],
        uuid.uuid4().hex,
    )
    assert d == "await_approval"
    assert a.tool_name in ("web_search_tool", "generate_document_template_tool")


def test_await_approval_response_carries_id(sqlite_db):
    d, a = evaluate_tool_gate(
        Principal("u", "u", Role.USER), ["generate_document_template_tool"], uuid.uuid4().hex
    )
    msg = await_approval_response(a)
    assert a.id in msg
    assert "phê duyệt" in msg.lower()