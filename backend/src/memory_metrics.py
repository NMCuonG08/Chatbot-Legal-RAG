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
    from prometheus_client import Histogram as _PromHistogram
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


class _NoopHistogram:
    """Drop-in for prometheus_client.Histogram when prometheus is unavailable."""

    def __init__(self, *a, **k):
        pass

    def observe(self, _v: float) -> None:
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


def _make_histogram(name: str, desc: str, buckets: Optional[list] = None,
                    labels: Optional[list] = None):
    if not _PROM:
        return _NoopHistogram()
    try:
        return _PromHistogram(name, desc, buckets=buckets, labelnames=labels or [])
    except Exception:
        return _NoopHistogram()


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

# Audit 4.2 — verify_answer (PEV) online LLMOps signals. Rejection rate =
# verify_rejection_total / verify_judged_total; alert when > 15% over 15m.
VERIFY_JUDGED = _make_counter("verify_judged_total",
                              "verify_answer judgments (denominator for rejection rate).")
VERIFY_REJECTIONS = _make_counter("verify_rejection_total",
                                  "verify_answer rejected verdicts (partial/unsupported).",
                                  labels=["verdict"])
VERIFY_SCORE = _make_histogram(
    "verify_faithfulness_score",
    "verify_answer faithfulness score distribution (0..1).",
    buckets=[0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


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


def inc_verify_judged() -> None:
    try:
        VERIFY_JUDGED.inc()
    except Exception:
        pass


def inc_verify_rejection(verdict: str) -> None:
    try:
        VERIFY_REJECTIONS.labels(verdict=verdict).inc()
    except Exception:
        pass


def observe_verify_score(score: float) -> None:
    try:
        VERIFY_SCORE.observe(float(score))
    except Exception:
        pass