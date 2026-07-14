"""Subprocess sandbox for pure-compute legal tools (defense-in-depth).

Deterministic calculation tools run in a throwaway child process with a
scrubbed environment + hard timeout, so a bug (infinite loop, runaway memory)
in a calc tool cannot wedge the agent/worker process. Network / retrieval /
graph / memory / sensitive tools are NOT sandboxed here — they either need
external resources (network, Qdrant, Neo4j) or are gated by the approval
workflow (``approval.evaluate_tool_gate``).

Public surface:
- ``SAFE_TO_SANDBOX`` — allowlist of FunctionTool names safe to isolate.
- ``run_in_sandbox(tool_name, args, timeout_s=10)`` — run one tool in a child
  process, return its Dict result, or an ``{"error": ...}`` Dict on timeout /
  unknown tool / subprocess failure.

Limitation: Windows has no seccomp; isolation is process-level + timeout +
env scrub only, not a syscall filter. Adequate for demo; not production
hardening.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)

# tool FunctionTool name -> (dotted module, function name). Only pure-compute
# tools (no network / file / DB / external API) belong here.
SAFE_TO_SANDBOX: Dict[str, tuple[str, str]] = {
    "contract_penalty_calculator": ("legal_tools", "calculate_contract_penalty"),
    "legal_age_checker": ("legal_tools", "check_legal_entity_age"),
    "inheritance_calculator": ("legal_tools", "calculate_inheritance_share"),
    "business_name_validator": ("legal_tools", "check_business_name_rules"),
    "statute_lookup": ("legal_tools", "get_statute_of_limitations"),
    "severance_pay_tool": ("legal_knowledge_tools", "calculate_severance_pay"),
    "overtime_pay_tool": ("legal_knowledge_tools", "calculate_overtime_pay"),
    "pit_monthly_tool": ("legal_knowledge_tools", "calculate_pit_monthly"),
    "land_registration_fee_tool": ("legal_knowledge_tools", "calculate_land_registration_fee"),
    "vehicle_registration_fee_tool": ("legal_knowledge_tools", "calculate_vehicle_registration_fee"),
    "court_fee_tool": ("legal_knowledge_tools", "calculate_court_fee"),
    "admin_fine_lookup_tool": ("legal_knowledge_tools", "lookup_administrative_fine"),
    "child_support_tool": ("legal_knowledge_tools", "calculate_child_support"),
    "law_version_tool": ("legal_knowledge_tools", "get_law_version"),
    "procedure_wizard_tool": ("legal_procedure_tools", "procedure_wizard"),
    "jurisdiction_resolver_tool": ("legal_procedure_tools", "jurisdiction_resolver"),
}

# Env vars the child genuinely needs: Python path resolution on Windows + the
# backend/src dir so `import legal_tools` works. Everything else is dropped.
_ENV_ALLOWLIST = {"SYSTEMROOT", "PATH", "PATHEXT", "PYTHONPATH", "PYTHONHOME", "TEMP", "TMP"}


def _scrub_env() -> Dict[str, str]:
    base = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    src_dir = os.path.dirname(os.path.abspath(__file__))
    existing = base.get("PYTHONPATH", "")
    base["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
    # Force UTF-8 stdio so Vietnamese output doesn't hit the cp1252 charmap on
    # Windows (the scrubbed env drops the locale vars that normally set this).
    base["PYTHONIOENCODING"] = "utf-8"
    base["PYTHONUTF8"] = "1"
    return base


def is_sandboxable(tool_name: str) -> bool:
    return tool_name in SAFE_TO_SANDBOX


def run_in_sandbox(tool_name: str, args: Dict[str, Any], timeout_s: float = 10.0) -> Dict[str, Any]:
    """Run a pure-compute tool in a child process. Returns the tool's Dict
    result, or an ``{"error": ...}`` Dict on failure. Never raises.
    """
    if tool_name not in SAFE_TO_SANDBOX:
        return {"error": f"tool {tool_name!r} not sandboxable"}
    payload = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False, default=str)
    cmd = [sys.executable, "-u", "-c", "from sandbox import _worker_main; _worker_main()"]
    try:
        proc = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_scrub_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "timeout_s": timeout_s, "tool": tool_name}
    except Exception as exc:
        logger.warning("sandbox spawn failed for %s: %s", tool_name, exc)
        return {"error": f"spawn_failed: {exc}", "tool": tool_name}

    if proc.returncode != 0:
        return {
            "error": f"worker_exit_{proc.returncode}",
            "tool": tool_name,
            "stderr": (proc.stderr or "")[-1000:],
        }
    out = (proc.stdout or "").strip()
    if not out:
        return {"error": "empty_output", "tool": tool_name, "stderr": (proc.stderr or "")[-1000:]}
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        return {"error": f"bad_json: {exc}", "tool": tool_name, "stdout": out[-1000:]}


def _worker_main() -> None:
    """Read {tool, args} JSON from stdin, dispatch to the impl fn, print JSON."""
    # Belt-and-suspenders UTF-8 stdout (env sets PYTHONIOENCODING too).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw else {}
    except Exception as exc:
        print(json.dumps({"error": f"bad_input: {exc}"}))
        return
    tool = req.get("tool")
    args = req.get("args", {}) or {}
    if tool not in SAFE_TO_SANDBOX:
        print(json.dumps({"error": f"tool {tool!r} not sandboxable"}))
        return
    module_name, fn_name = SAFE_TO_SANDBOX[tool]
    try:
        mod = __import__(module_name)
        fn = getattr(mod, fn_name)
        result = fn(**args)
        print(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as exc:
        print(json.dumps({"error": f"tool_error: {exc}", "tool": tool}))


__all__ = ["SAFE_TO_SANDBOX", "is_sandboxable", "run_in_sandbox"]


if __name__ == "__main__":  # pragma: no cover
    print(run_in_sandbox("contract_penalty_calculator", {"contract_value": 1e9, "penalty_rate": 0.08, "days_late": 30}))