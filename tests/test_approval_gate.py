"""Phase 0 — C1/C3 regression: the sensitive-tool approval gate must actually block.

``_maybe_block_on_approval`` (tasks.py) anticipates the agent's tool set via
``filter_tools_for_query`` and asks ``evaluate_tool_gate`` whether a non-exempt
role is about to call a SENSITIVE_TOOLS entry. Before this fix,
``filter_tools_for_query`` was never imported into ``tasks`` — the ``NameError``
was swallowed by the broad ``except Exception`` and the gate returned ``None``
(proceed) for EVERY non-exempt sensitive call. This file asserts the gate
returns a non-None "chờ phê duyệt" response when a sensitive tool is anticipated.
"""
from __future__ import annotations

import sys
import types

import pytest


class _FakeTool:
    """Minimal stand-in for a LlamaIndex FunctionTool (``_tool_name`` reads .name)."""
    def __init__(self, name: str):
        self.name = name


@pytest.fixture
def tasks_module(monkeypatch):
    """Import tasks with heavy deps stubbed + a controllable filter_tools_for_query."""
    for name in ("agent", "verify_answer", "metacognitive", "rlhf_store",
                 "summarizer", "guardrails_manager"):
        mod = types.ModuleType(name)
        if name == "agent":
            mod.ai_agent_handle = lambda *a, **k: None
            mod.clear_user_runtime_caches = lambda *a, **k: None
            # Default: anticipate nothing; per-test monkeypatch overrides this.
            mod.filter_tools_for_query = lambda query, history=None, role=None: []
        if name == "verify_answer":
            mod.judge_answer = lambda *a, **k: None
        if name == "metacognitive":
            mod.build_escalation = lambda *a, **k: None
            mod.ESCALATION_PREFIX = "[ESCALATION]"
        if name == "rlhf_store":
            mod.find_similar_good = lambda *a, **k: None
        if name == "summarizer":
            mod.summarize_text = lambda *a, **k: "summary"
        if name == "guardrails_manager":
            mod.LegalGuardrailsManager = type(
                "LegalGuardrailsManager", (), {"__init__": lambda self, *a, **k: None}
            )
        monkeypatch.setitem(sys.modules, name, mod)
    import tasks  # noqa: WPS433
    return tasks


def _patch_gate_to_await(monkeypatch, tasks_module):
    """Make evaluate_tool_gate return await_approval + a fake approval object,
    without touching the DB. ``_maybe_block_on_approval`` does a local
    ``from approval import evaluate_tool_gate`` at call time, so patching the
    approval module attribute is picked up."""
    import approval

    class _FakeApproval:
        id = "approval-123"
        tool_name = "web_search_tool"

    monkeypatch.setattr(
        approval, "evaluate_tool_gate",
        lambda principal, names, run_id=None: ("await_approval", _FakeApproval()),
    )
    monkeypatch.setattr(
        approval, "await_approval_response",
        lambda a: f"[CẦN PHÊ DUYỆT] sentinel-{a.tool_name}",
    )


def test_gate_blocks_non_exempt_sensitive_tool(tasks_module, monkeypatch):
    """Non-exempt role + anticipated sensitive tool -> gate returns non-None."""
    _patch_gate_to_await(monkeypatch, tasks_module)
    monkeypatch.setattr(
        tasks_module, "filter_tools_for_query",
        lambda query, history=None, role=None: [_FakeTool("web_search_tool")],
    )

    resp = tasks_module._maybe_block_on_approval(
        state=None, question="tìm kiếm trên web giúp tôi",
        history=[], role="user", user_id="u1", run_id="r1",
    )
    assert resp is not None
    assert "CẦN PHÊ DUYỆT" in resp


def test_gate_proceeds_when_no_sensitive_tool_anticipated(tasks_module, monkeypatch):
    """Non-exempt role but only non-sensitive tools anticipated -> proceed (None)."""
    import approval
    monkeypatch.setattr(
        approval, "evaluate_tool_gate",
        lambda principal, names, run_id=None: ("proceed", None),
    )
    monkeypatch.setattr(
        tasks_module, "filter_tools_for_query",
        lambda query, history=None, role=None: [_FakeTool("statute_lookup")],
    )

    resp = tasks_module._maybe_block_on_approval(
        state=None, question="tra cứu điều luật",
        history=[], role="user", user_id="u1", run_id="r1",
    )
    assert resp is None


def test_gate_proceeds_for_exempt_role(tasks_module, monkeypatch):
    """Exempt role (lawyer) -> always proceed even if sensitive tool anticipated."""
    monkeypatch.setattr(
        tasks_module, "filter_tools_for_query",
        lambda query, history=None, role=None: [_FakeTool("web_search_tool")],
    )

    resp = tasks_module._maybe_block_on_approval(
        state=None, question="tìm kiếm trên web",
        history=[], role="lawyer", user_id="u1", run_id="r1",
    )
    assert resp is None


def test_gate_proceeds_for_anonymous(tasks_module, monkeypatch):
    """Anonymous path (role None) -> proceed (legacy demo behavior)."""
    resp = tasks_module._maybe_block_on_approval(
        state=None, question="anything", history=[], role=None,
        user_id="anon", run_id="r1",
    )
    assert resp is None