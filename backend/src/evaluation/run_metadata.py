"""Run metadata: pin everything that affects eval reproducibility.

A bare eval score is meaningless without the context that produced it. This
module captures that context into a frozen ``RunMetadata`` record so two runs
can be compared apples-to-apples and a regression diff can blame a real change
vs. a model/prompt drift:

- git sha + dirty flag + branch (which code ran)
- agent + judge LLM provider/model (which model judged)
- judge prompt hash + system prompt hash (which prompts scored)
- python version + platform (which runtime)
- env snapshot (which knobs were set) — **secrets are never captured**, only
  presence booleans for ``*_API_KEY`` style vars.

Public surface:
- ``RunMetadata`` — frozen dataclass, the pinned context.
- ``compute_prompt_hash`` / ``compute_system_prompt_hash`` — sha256 of prompts.
- ``capture_env`` — scrubbed env snapshot from an allowlist.
- ``capture_git`` — git sha/dirty/branch via subprocess.
- ``build_run_metadata`` — assemble a ``RunMetadata`` for the current process.
- ``metadata_to_dict`` — JSON-serializable view.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# Bumped when the eval payload schema or metric definitions change in a way
# that breaks cross-run comparison. Regression diff refuses to compare runs
# with mismatched eval_version.
EVAL_VERSION = "1.0.0"

# Env var prefixes whose VALUES are safe to record (model names, URLs, knobs).
# Anything matching ``*_API_KEY`` / ``*_SECRET`` / ``*_TOKEN`` / ``*_PASSWORD``
# is recorded as a presence boolean only — the value is never persisted.
_ENV_VALUE_PREFIXES = (
    "LLM_", "JUDGE_", "OLLAMA_", "CUSTOM_EMBEDDING_", "RERANKER_",
    "USE_OLLAMA_AS_MAIN", "OTEL_BRIDGE_ENABLED", "OTEL_EXPORTER_OTLP_ENDPOINT",
    "GUARDRAILS_PII_OUTPUT_ENABLED", "VERIFY_ANSWER_THRESHOLD",
)
_SECRET_SUFFIXES = ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "PWD")


def _is_secret(name: str) -> bool:
    upper = name.upper()
    return any(upper.endswith(s) for s in _SECRET_SUFFIXES)


def compute_prompt_hash(prompts: Iterable[str]) -> str:
    """sha256 over the concatenated prompt source strings (stable ordering)."""
    h = hashlib.sha256()
    for p in prompts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")  # boundary so ["ab","c"] != ["a","bc"]
    return h.hexdigest()


def compute_system_prompt_hash() -> str:
    """sha256 of the canonical system prompt used by the agent generation path.

    Falls back to a constant when the prompt builder is unavailable (tests /
    partial imports), so the field is always populated.
    """
    try:
        from tasks import SYSTEM_PROMPT  # type: ignore
        return compute_prompt_hash([SYSTEM_PROMPT])
    except Exception:
        return compute_prompt_hash(["<system-prompt-unavailable>"])


def capture_env(allowlist_prefixes: Iterable[str] = _ENV_VALUE_PREFIXES) -> Dict[str, object]:
    """Snapshot env vars matching the prefixes; secrets -> presence bool only."""
    snapshot: Dict[str, object] = {}
    prefixes = tuple(allowlist_prefixes)
    for name, value in sorted(os.environ.items()):
        upper = name.upper()
        if _is_secret(name):
            snapshot[name] = bool(value)
            continue
        if any(upper.startswith(p) for p in prefixes):
            snapshot[name] = value
    return snapshot


def capture_git() -> Dict[str, Optional[str]]:
    """git sha / dirty / branch via subprocess. Never raises."""
    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode != 0:
                return None
            return out.stdout.strip() or None
        except Exception as exc:
            logger.debug("git %s failed: %s", " ".join(args), exc)
            return None

    sha = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "git_sha": sha,
        "git_dirty": bool(status),
        "git_branch": branch,
    }


@dataclass(frozen=True)
class RunMetadata:
    """Pinned context for one eval run. Immutable + hashable."""

    run_id: str
    created_at: str
    eval_version: str
    git_sha: Optional[str]
    git_dirty: bool
    git_branch: Optional[str]
    agent_llm_provider: str
    agent_llm_model: str
    judge_provider: str
    judge_model: str
    judge_temperature: float
    judge_prompt_hash: str
    system_prompt_hash: str
    python_version: str
    platform: str
    env_snapshot: Dict[str, object] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)
    judge_panel_hashes: Optional[str] = None  # P2: multi-judge panel hash


def _agent_identity() -> Dict[str, str]:
    """Read the agent provider/model without importing heavy deps."""
    try:
        from brain import get_main_provider
        prov = get_main_provider()
        return {"provider": prov.name, "model": prov.model}
    except Exception as exc:
        logger.debug("agent identity fallback: %s", exc)
        return {
            "provider": os.environ.get("LLM_PROVIDER", "groq"),
            "model": os.environ.get("LLM_MODEL", "unknown"),
        }


def build_run_metadata(
    *,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_prompt_hash: str,
    system_prompt_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> RunMetadata:
    """Assemble a ``RunMetadata`` for the current process + judge config."""
    ident = _agent_identity()
    git = capture_git()
    return RunMetadata(
        run_id=run_id or uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(),
        eval_version=EVAL_VERSION,
        git_sha=git["git_sha"],
        git_dirty=git["git_dirty"],
        git_branch=git["git_branch"],
        agent_llm_provider=ident["provider"],
        agent_llm_model=ident["model"],
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        judge_prompt_hash=judge_prompt_hash,
        system_prompt_hash=system_prompt_hash or compute_system_prompt_hash(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        env_snapshot=capture_env(),
        extra=dict(extra or {}),
    )


def metadata_to_dict(meta: RunMetadata) -> Dict[str, object]:
    """JSON-serializable dict view of a ``RunMetadata``."""
    return asdict(meta)


__all__ = [
    "EVAL_VERSION",
    "RunMetadata",
    "build_run_metadata",
    "capture_env",
    "capture_git",
    "compute_prompt_hash",
    "compute_system_prompt_hash",
    "metadata_to_dict",
]