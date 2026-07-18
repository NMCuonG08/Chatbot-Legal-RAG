"""Robust JSON extraction from LLM output (handles code fences + prose).

LLMs asked for structured JSON often wrap it in ```json ... ``` fences or
surround it with prose. This module finds and parses the first JSON value
(object or array) without relying on the model being perfectly disciplined.

Import-light (stdlib only) so supervisor/planner/judge modules can use it
without pulling langchain/llama-index.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _try_loads(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _scan_first(text: str) -> Any:
    """Balanced-brace/bracket scan: find the first complete {...} or [...]."""
    start_idx = None
    open_ch = close_ch = ""
    for i, c in enumerate(text):
        if c in "{[":
            start_idx, open_ch = i, c
            close_ch = "}" if c == "{" else "]"
            break
    if start_idx is None:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start_idx, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return _try_loads(text[start_idx : i + 1])
    return None


def extract_json(text: str) -> Any:
    """Extract + ``json.loads`` the first JSON value (object or array) in text.

    Tries fenced blocks first, then the whole text, then a balanced-scan.
    Returns the parsed object/array, or ``None`` when nothing parses.
    Never raises.
    """
    if not text:
        return None
    for m in _FENCE_RE.finditer(text):
        obj = _try_loads(m.group(1))
        if obj is not None:
            return obj
    obj = _try_loads(text)
    if obj is not None:
        return obj
    return _scan_first(text)


__all__ = ["extract_json"]