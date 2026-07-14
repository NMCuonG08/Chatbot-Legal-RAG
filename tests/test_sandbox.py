"""Phase 2b — subprocess sandbox tests.

Exercises ``sandbox.run_in_sandbox`` against the real pure-compute tool impls
(no network / DB / Qdrant). The worker runs in a child process with a scrubbed
environment + UTF-8 stdio so Vietnamese output survives.
"""
import io
import json
import subprocess
import sys

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


# ---- _worker_main (direct, in-process) ----
# The worker normally runs in a child process (not coverage-measured), so drive
# it directly here to cover its branches.


def test_worker_main_dispatches_calc(capsys):
    import io
    import json

    import sandbox

    payload = json.dumps({"tool": "contract_penalty_calculator", "args": {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30}})
    sandbox.sys.stdin = io.StringIO(payload)
    try:
        sandbox._worker_main()
    finally:
        sandbox.sys.stdin = sys.__stdin__
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "error" not in data


def test_worker_main_unknown_tool(capsys):
    import io
    import json

    import sandbox

    sandbox.sys.stdin = io.StringIO(json.dumps({"tool": "nope", "args": {}}))
    try:
        sandbox._worker_main()
    finally:
        sandbox.sys.stdin = sys.__stdin__
    out = capsys.readouterr().out
    assert "not sandboxable" in json.loads(out)["error"]


def test_worker_main_tool_error(capsys):
    import io
    import json

    import sandbox

    # Missing required args -> impl raises -> worker emits tool_error.
    sandbox.sys.stdin = io.StringIO(json.dumps({"tool": "contract_penalty_calculator", "args": {"penalty_rate": 0.08}}))
    try:
        sandbox._worker_main()
    finally:
        sandbox.sys.stdin = sys.__stdin__
    out = capsys.readouterr().out
    assert "tool_error" in json.loads(out)["error"]


def test_worker_main_bad_input(capsys):
    import io
    import json

    import sandbox

    sandbox.sys.stdin = io.StringIO("not-json{")
    try:
        sandbox._worker_main()
    finally:
        sandbox.sys.stdin = sys.__stdin__
    out = capsys.readouterr().out
    assert "bad_input" in json.loads(out)["error"]


def test_worker_main_empty_input(capsys):
    import io
    import json

    import sandbox

    sandbox.sys.stdin = io.StringIO("")
    try:
        sandbox._worker_main()
    finally:
        sandbox.sys.stdin = sys.__stdin__
    out = capsys.readouterr().out
    # Empty input -> req={} -> tool None -> not sandboxable.
    assert "not sandboxable" in json.loads(out)["error"]


# ---- run_in_sandbox error branches (returncode / empty / bad_json) ----


def test_run_in_sandbox_worker_nonzero_exit(monkeypatch):
    import sandbox

    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **kw: _Proc())
    r = sandbox.run_in_sandbox("contract_penalty_calculator", {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30})
    assert "worker_exit_2" in r["error"]


def test_run_in_sandbox_empty_output(monkeypatch):
    import sandbox

    class _Proc:
        returncode = 0
        stdout = "   "
        stderr = ""

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **kw: _Proc())
    r = sandbox.run_in_sandbox("contract_penalty_calculator", {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30})
    assert r["error"] == "empty_output"


def test_run_in_sandbox_bad_json(monkeypatch):
    import sandbox

    class _Proc:
        returncode = 0
        stdout = "not json at all"
        stderr = ""

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **kw: _Proc())
    r = sandbox.run_in_sandbox("contract_penalty_calculator", {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30})
    assert "bad_json" in r["error"]


def test_run_in_sandbox_spawn_failed(monkeypatch):
    import sandbox

    def _boom(*a, **kw):
        raise OSError("no such interpreter")

    monkeypatch.setattr(sandbox.subprocess, "run", _boom)
    r = sandbox.run_in_sandbox("contract_penalty_calculator", {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30})
    assert "spawn_failed" in r["error"]