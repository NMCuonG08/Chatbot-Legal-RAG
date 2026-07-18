"""@sandboxable opt-in isolation tests (criterion #6: guardrails in loop).

Marker: unit. Validates the ``sandboxable`` decorator on pure-compute tool
wrappers in ``agent_tool_wrappers``:

- ``SANDBOX_ENABLED`` false (default) -> pure passthrough to the in-process impl,
  zero subprocess overhead, identical output to the un-sandboxed path.
- ``SANDBOX_ENABLED`` true -> dispatches to ``sandbox.run_in_sandbox`` with the
  bound arguments (defaults applied), returns its JSON.
- dispatch failure -> graceful fallback to the in-process impl (never breaks
  the tool).

``run_in_sandbox`` is stubbed so no subprocess is spawned. The wrappers are
importable in tests (precedent: test_graph_memory imports agent_tool_wrappers).
"""
import json

import pytest

pytestmark = pytest.mark.unit


def _stub_sandbox(monkeypatch, sink):
    """Replace run_in_sandbox in agent_tool_wrappers with a recording stub."""
    import agent_tool_wrappers as w

    def _fake(tool_name, args, timeout_s=10.0):
        sink.append({"tool": tool_name, "args": args, "timeout_s": timeout_s})
        return {"sandboxed": True, "tool": tool_name, "args": args}

    monkeypatch.setattr(w, "run_in_sandbox", _fake)


def test_sandbox_disabled_passthrough(monkeypatch):
    import config
    import agent_tool_wrappers as w

    monkeypatch.setattr(config, "SANDBOX_ENABLED", False)
    sink = []
    _stub_sandbox(monkeypatch, sink)

    out = w.contract_penalty_calculator(1e9, 0.08, 30)
    data = json.loads(out)
    # in-process impl produced a real penalty figure, no sandbox dispatch
    assert sink == []
    assert "sandboxed" not in data
    assert "1,000,000,000 VNĐ" in data.get("contract_value", "")


def test_sandbox_enabled_dispatches_with_bound_args(monkeypatch):
    import config
    import agent_tool_wrappers as w

    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    sink = []
    _stub_sandbox(monkeypatch, sink)

    out = w.contract_penalty_calculator(1e9, 0.08, 30)
    data = json.loads(out)
    assert sink, "run_in_sandbox must be called when SANDBOX_ENABLED"
    assert sink[0]["tool"] == "contract_penalty_calculator"
    # bound arguments passed through verbatim
    assert sink[0]["args"] == {
        "contract_value": 1e9,
        "penalty_rate": 0.08,
        "days_late": 30,
    }
    assert data["sandboxed"] is True


def test_sandbox_applies_defaults(monkeypatch):
    import config
    import agent_tool_wrappers as w

    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    sink = []
    _stub_sandbox(monkeypatch, sink)

    # action_type omitted -> signature default "sign_contract" must be filled
    # before handing args to run_in_sandbox.
    w.legal_age_checker(2000)
    assert sink[0]["args"] == {
        "birth_year": 2000,
        "action_type": "sign_contract",
        "gender": "",
    }


def test_sandbox_dispatch_failure_falls_back(monkeypatch):
    import config
    import agent_tool_wrappers as w

    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)

    def _boom(tool_name, args, timeout_s=10.0):
        raise RuntimeError("subprocess broken")

    monkeypatch.setattr(w, "run_in_sandbox", _boom)

    out = w.contract_penalty_calculator(1e9, 0.08, 30)
    data = json.loads(out)
    # fell back to the in-process impl -> real result, no crash, no "sandboxed"
    assert "sandboxed" not in data
    assert "1,000,000,000 VNĐ" in data.get("contract_value", "")