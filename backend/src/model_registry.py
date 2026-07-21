"""Model & prompt registry loader (LLMOps Tier 1).

Loads backend/models.yaml once at process start and exposes pinned versions.
Wire into trace/run_metadata so every chat response carries the version set
that produced it — backbone of rollback + reproducibility.

Usage (backend/src/app.py or tasks.py at startup):
    from model_registry import get_versions, log_versions
    log_versions()              # prints + best-effort emit to trace
    v = get_versions()          # dict for metadata pinning

No heavy deps (stdlib + pyyaml) so importable anywhere without side effects.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as _e:  # pragma: no cover
    raise RuntimeError("pyyaml required: pip install pyyaml") from _e

_CACHE: dict[str, Any] | None = None


def _resolve_path() -> Path:
    """Find models.yaml: env MODELS_YAML > backend/models.yaml > repo root."""
    env = os.environ.get("MODELS_YAML")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).resolve().parent  # backend/src/
    candidates = [
        here.parent / "models.yaml",        # backend/models.yaml
        here.parent.parent / "models.yaml", # repo root
    ]
    for c in candidates:
        if c.exists():
            return c
    return here.parent / "models.yaml"


def load() -> dict[str, Any]:
    """Load + cache models.yaml. Empty dict if file missing (graceful)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = _resolve_path()
    if not path.exists():
        _CACHE = {}
        return _CACHE
    with open(path, "r", encoding="utf-8") as f:
        _CACHE = yaml.safe_load(f) or {}
    return _CACHE


def get_versions() -> dict[str, str]:
    """Flat {component: 'name@version'} map for trace/metadata."""
    r = load()
    out: dict[str, str] = {"registry_version": str(r.get("registry_version", ""))}
    for key in ("embedding", "reranker", "llm", "judge"):
        blk = r.get(key, {}) or {}
        # embedding/reranker use name+version; llm/judge use model+pinned_at.
        nm = blk.get("name") or blk.get("model", "?")
        ver = blk.get("version") or blk.get("pinned_at") or "?"
        out[key] = f"{nm}@{ver}"
    out["prompts"] = str(r.get("prompts", {}))
    out["corpus_sha"] = str((r.get("rag") or {}).get("corpus_sha", ""))
    return out


def log_versions() -> None:
    """Best-effort log active versions to stdout + trace if available."""
    v = get_versions()
    print(f"[model_registry] active versions: {v}", flush=True)
    try:
        from trace import emit_step  # type: ignore
        emit_step({"event": "model_registry", **v})
    except Exception:
        pass  # trace not available yet (import-time)