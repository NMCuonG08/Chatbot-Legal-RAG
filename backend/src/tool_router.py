"""Semantic tool router — embed query vs tool descriptions, top-k cosine.

Fixes Flaw 2 (brittle keyword tool filter). A paraphrase like "công ty cho tôi
nghỉ" misses the keyword "cho nghỉ" (not a substring of "cho tôi nghỉ") but
semantically matches ``severance_pay_tool``. Embedding the query against each
tool's description and keeping the top-k above a similarity threshold is
robust to paraphrase and needs zero keyword maintenance.

Contract: returns ``None`` when semantic routing is disabled OR the embedding
service is unavailable, so ``agent.filter_tools_for_query`` can fall back to
the existing keyword path. Never raises.

Reuse: ``brain.get_embedding`` (same embedding used by episodic/RAG). The tool
index is built once (module-level cache) — tool descriptions are static, so
embeddings are computed a single time per process.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SEMANTIC_TOOL_ROUTER_ENABLED = os.getenv("SEMANTIC_TOOL_ROUTER_ENABLED", "1") == "1"
TOOL_TOP_K = int(os.getenv("TOOL_TOP_K", "8"))
TOOL_SIM_THRESHOLD = float(os.getenv("TOOL_SIM_THRESHOLD", "0.30"))

# Tool fns always included regardless of similarity (safety/utility contract
# preserved from the old keyword path).
_ALWAYS_NAMES = ("legal_disclaimer_tool", "get_current_time", "tavily_search_tool")
_RECALL_NAMES = ("recall_user_memory_tool",)

_index = None  # list[tuple[tool, name, desc, emb]]


def _cosine(a, b) -> float:
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def reset_index() -> None:
    """Drop the cached tool-embedding index (tests / hot-reload)."""
    global _index
    _index = None


def _build_index(embed_fn) -> list:
    global _index
    if _index is not None:
        return _index
    import agent_tool_wrappers as atw
    entries = []
    for t in atw.all_tools:
        meta = t.metadata
        name = meta.name or ""
        desc = (meta.description or "") + " " + name
        try:
            emb = embed_fn(desc)
        except Exception as exc:
            logger.warning("tool_router: embed failed for %s: %s", name, exc)
            emb = None
        entries.append((t, name, desc, emb))
    _index = entries
    return entries


def _find_by_name(name: str, entries: list):
    for (t, n, _d, _e) in entries:
        if n == name:
            return t
    return None


def select_tools_semantic(
    query: str,
    history: Optional[list] = None,
    get_embedding_fn=None,
) -> Optional[list]:
    """Return top-k tools by query↔description cosine, plus always-include.

    ``None`` => caller falls back to keyword path. Never raises.
    """
    if not SEMANTIC_TOOL_ROUTER_ENABLED:
        return None
    embed = get_embedding_fn
    if embed is None:
        try:
            from brain import get_embedding as embed  # noqa: WPS433
        except Exception as exc:
            logger.warning("tool_router: brain.get_embedding unavailable: %s", exc)
            return None
    entries = _build_index(embed)
    try:
        q_emb = embed(query)
    except Exception as exc:
        logger.warning("tool_router: query embed failed, falling back: %s", exc)
        return None
    if not q_emb:
        return None

    scored = []
    for (t, _name, _desc, emb) in entries:
        if emb is None:
            continue
        scored.append((_cosine(q_emb, emb), t))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [t for sim, t in scored[:TOOL_TOP_K] if sim >= TOOL_SIM_THRESHOLD]

    picked = _ensure_always(picked, entries, history)
    return picked


def _ensure_always(picked: list, entries: list, history: Optional[list]) -> list:
    """Append safety/utility + recall tools if not already present."""
    picked_ids = {id(t) for t in picked}
    for name in _ALWAYS_NAMES:
        t = _find_by_name(name, entries)
        if t is not None and id(t) not in picked_ids:
            picked.append(t)
            picked_ids.add(id(t))
    # Long-term memory recall when there is conversation context.
    has_context = bool(history) and len(history) > 1
    if has_context:
        t = _find_by_name(_RECALL_NAMES[0], entries)
        if t is not None and id(t) not in picked_ids:
            picked.append(t)
    return picked