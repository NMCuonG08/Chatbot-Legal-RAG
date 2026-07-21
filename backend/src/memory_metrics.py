"""P6 — Memory ops monitoring counters (Prometheus / LLMOps online signals).

Lightweight, non-invasive counters for the memory subsystem. Mounted on the
existing ``/metrics`` endpoint (``app.py`` Instrumentator exposes the default
registry; these counters register on the same default registry so they appear
automatically). Guarded so a missing ``prometheus_client`` never breaks the chat
path — counters degrade to no-op stubs.

Counters (P6c):
- ``memory_short_term_hits_total`` / ``memory_short_term_misses_total``
- ``episodic_extractions_total{result=success|skipped_none|skipped_duplicate|skipped_empty|error}``
- ``episodic_pollution_total``  (legal-citation bleed into extracted facts)
- ``react_memory_recall_total{hit=hit|miss}``
- ``profile_merge_total{field=...}``

Usage:
    from memory_metrics import inc_short_term_hit, inc_short_term_miss, ...
    inc_short_term_hit()
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter as _PromCounter
    _PROM = True
except Exception:  # pragma: no cover - prometheus_client optional in dev
    _PROM = False


class _NoopCounter:
    """Drop-in for prometheus_client.Counter when prometheus is unavailable."""

    def __init__(self, *a, **k):
        pass

    def inc(self, n: float = 1.0) -> None:
        pass

    def labels(self, **_kw):
        return self


def _make_counter(name: str, desc: str, labels: Optional[list] = None):
    if not _PROM:
        return _NoopCounter()
    try:
        return _PromCounter(name, desc, labelnames=labels or [])
    except Exception:
        # Already registered (duplicate import in long-lived process) -> no-op
        # to avoid a double-registration crash. The first registration wins.
        return _NoopCounter()


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

SHORT_TERM_HITS = _make_counter("memory_short_term_hits_total",
                                "Short-term rolling-summary cache/Redis hits.")
SHORT_TERM_MISSES = _make_counter("memory_short_term_misses_total",
                                  "Short-term rolling-summary misses (empty/new).")
EPISODIC_EXTRACTIONS = _make_counter("episodic_extractions_total",
                                     "Episodic extraction outcomes.",
                                     labels=["result"])
EPISODIC_POLLUTION = _make_counter("episodic_pollution_total",
                                   "Legal-citation text bled into extracted user facts (target 0).")
REACT_RECALL = _make_counter("react_memory_recall_total",
                             "recall_user_memory_tool recall outcomes.",
                             labels=["hit"])
PROFILE_MERGE = _make_counter("profile_merge_total",
                              "UserProfile field merges.",
                              labels=["field"])


# ---------------------------------------------------------------------------
# Convenience wrappers (callable without worrying about labels/prom presence)
# ---------------------------------------------------------------------------

def inc_short_term_hit() -> None:
    try:
        SHORT_TERM_HITS.inc()
    except Exception:
        pass


def inc_short_term_miss() -> None:
    try:
        SHORT_TERM_MISSES.inc()
    except Exception:
        pass


def inc_episodic_extraction(result: str) -> None:
    try:
        EPISODIC_EXTRACTIONS.labels(result=result).inc()
    except Exception:
        pass


def inc_episodic_pollution() -> None:
    try:
        EPISODIC_POLLUTION.inc()
    except Exception:
        pass


def inc_react_recall(hit: bool) -> None:
    try:
        REACT_RECALL.labels(hit="hit" if hit else "miss").inc()
    except Exception:
        pass


def inc_profile_merge(field: str) -> None:
    try:
        PROFILE_MERGE.labels(field=field).inc()
    except Exception:
        pass