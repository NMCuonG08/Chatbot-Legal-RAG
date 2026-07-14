"""Phase 2b — subprocess sandbox tests.

Exercises ``sandbox.run_in_sandbox`` against the real pure-compute tool impls
(no network / DB / Qdrant). The worker runs in a child process with a scrubbed
environment + UTF-8 stdio so Vietnamese output survives.
"""
import subprocess

from sandbox import SAFE_TO_SANDBOX, _scrub_env, is_sandboxable, run_in_sandbox


def test_safe_to_sandbox_excludes_network_and_sensitive():
    assert not is_sandboxable("web_search_tool")
    assert not is_sandboxable("tavily_search_tool")
    assert not is_sandboxable("generate_document_template_tool")
    assert not is_sandboxable("recall_legal_graph_tool")
    assert not is_sandboxable("recall_user_memory_tool")
    assert is_sandboxable("contract_penalty_calculator")
    assert is_sandboxable("pit_monthly_tool")


def test_run_in_sandbox_contract_penalty():
    r = run_in_sandbox(
        "contract_penalty_calculator",
        {"contract_value": 1_000_000_000, "penalty_rate": 0.08, "days_late": 30},
    )
    assert "error" not in r
    assert "24,000,000" in r["penalty_amount"]
    assert "418" in r["legal_basis"]


def test_run_in_sandbox_legal_age():
    r = run_in_sandbox("legal_age_checker", {"birth_year": 2000, "action_type": "sign_contract"})
    assert "error" not in r
    assert r["age"] >= 18


def test_run_in_sandbox_pit_monthly():
    r = run_in_sandbox("pit_monthly_tool", {"taxable_income": 50_000_000})
    assert "error" not in r


def test_run_in_sandbox_vietnamese_output_preserved():
    # Regression: scrubbed env must keep UTF-8 so Vietnamese chars don't hit cp1252.
    r = run_in_sandbox("contract_penalty_calculator", {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30})
    assert "VNĐ" in r["contract_value"] or "VNĐ" in r.get("penalty_amount", "")


def test_run_in_sandbox_rejects_unknown_tool():
    r = run_in_sandbox("does_not_exist", {})
    assert "not sandboxable" in r["error"]


def test_run_in_sandbox_rejects_sensitive_tool():
    r = run_in_sandbox("web_search_tool", {"query": "x"})
    assert "not sandboxable" in r["error"]


def test_run_in_sandbox_tool_error_captured():
    # Bad args (missing required) -> worker catches + returns tool_error, no raise.
    r = run_in_sandbox("contract_penalty_calculator", {"penalty_rate": 0.08})
    assert "error" in r
    assert "tool_error" in r["error"]


def test_run_in_sandbox_timeout_returns_error(monkeypatch):
    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="x", timeout=0.01)

    monkeypatch.setattr("sandbox.subprocess.run", _timeout)
    r = run_in_sandbox("contract_penalty_calculator", {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30}, timeout_s=0.01)
    assert r["error"] == "timeout"
    assert r["timeout_s"] == 0.01


def test_scrub_env_drops_secrets():
    import os
    os.environ["FAKE_SECRET_TOKEN"] = "leak-me"
    os.environ["GROQ_API_KEY"] = "should-not-leak"
    try:
        env = _scrub_env()
        assert "FAKE_SECRET_TOKEN" not in env
        assert "GROQ_API_KEY" not in env
        assert "PYTHONPATH" in env
        assert env["PYTHONIOENCODING"] == "utf-8"
    finally:
        os.environ.pop("FAKE_SECRET_TOKEN", None)
        os.environ.pop("GROQ_API_KEY", None)


def test_all_safe_to_sandbox_names_are_known_calc_tools():
    import importlib
    for tool_name, (module_name, fn_name) in SAFE_TO_SANDBOX.items():
        mod = importlib.import_module(module_name)
        assert hasattr(mod, fn_name), f"{tool_name} -> {module_name}.{fn_name} missing"