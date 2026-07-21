"""Prompt loader (LLMOps Tier 1.2) — tách prompt inline ra file + version.

Thay vì prompt hardcoded trong agent.py / tasks.py, load từ file:
    backend/prompts/<name>.<version>.txt

Version chọn theo:
    1. PROMPT_<NAME> env (override từng prompt, vd PROMPT_AGENT_SYSTEM=v2)
    2. models.yaml -> prompts.<name>  (registry trung tâm)
    3. default "v1"

Template variables (vd {current_date}, {current_year}, {conversation_str})
dùng str.format(**vars). KHÔNG dùng f-string trong file prompt.

Usage:
    from prompt_loader import load_prompt, prompt_version
    prompt = load_prompt("agent_system", current_date=..., current_year=...)
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE: dict[str, str] = {}


def _prompts_dir() -> Path:
    here = Path(__file__).resolve().parent  # backend/src/
    for c in (here.parent / "prompts", here.parent.parent / "prompts"):
        if c.exists():
            return c
    return here.parent / "prompts"


def _resolve_version(name: str) -> str:
    # 1. env override
    env_ver = os.environ.get(f"PROMPT_{name.upper()}")
    if env_ver:
        return env_ver
    # 2. models.yaml registry
    try:
        from model_registry import load  # type: ignore
        reg = load().get("prompts", {}) or {}
        if name in reg:
            return str(reg[name])
    except Exception:
        pass
    # 3. default
    return "v1"


def load_prompt(name: str, **template_vars: object) -> str:
    """Load + cache prompt text, apply str.format(**template_vars)."""
    ver = _resolve_version(name)
    key = f"{name}:{ver}"
    if key not in _CACHE:
        path = _prompts_dir() / f"{name}.{ver}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}. "
                f"Create it or set PROMPT_{name.upper()} to an existing version."
            )
        _CACHE[key] = path.read_text(encoding="utf-8")
    text = _CACHE[key]
    if template_vars:
        text = text.format(**template_vars)
    return text


def prompt_version(name: str) -> str:
    """Return active version tag (for trace/metadata)."""
    return _resolve_version(name)


def reload() -> None:
    """Clear cache — gọi sau khi đổi prompt file lúc dev/debug."""
    _CACHE.clear()