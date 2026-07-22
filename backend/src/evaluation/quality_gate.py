"""Audit 4.1 — golden-set absolute quality gate (blocking PR floor).

The existing ``regression`` gate only diffs a candidate run against a baseline;
it has no absolute floor, so a PR can ship a model whose average
faithfulness/relevance is mediocre (e.g. 0.5) as long as it did not regress
vs. an equally-bad baseline. This module adds the missing *absolute* gate:

- ``apply_quality_gate(summary)`` reads a single run's ``generation_summary``
  and returns PASS only when BOTH ``faithfulness_mean`` and
  ``answer_relevance_mean`` are >= the floor (default 0.80), with enough
  samples (>= ``min_n``) for the mean to be a signal rather than noise.
- ``gate_from_report(path)`` loads a persisted run JSON and applies the gate.

Used by the CI quality-gate workflow (``eval-gate.yml``) which runs the golden
eval set and exits 1 when the gate is FAIL — blocking the PR.

Public surface: ``QualityGatePolicy``, ``QualityGateResult``,
``apply_quality_gate``, ``gate_from_report``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default floor mirrors the audit spec: a golden-set PR gate that fails the
# build when average faithfulness OR answer_relevance < 0.80. Kept in sync with
# evaluation.regression.GatePolicy.min_faithfulness (also raised 0.7 -> 0.80).
DEFAULT_FLOOR = 0.80
# Minimum samples for a non-INCONCLUSIVE verdict (mean of 1 sample is noise).
DEFAULT_MIN_N = 10


@dataclass(frozen=True)
class QualityGatePolicy:
    """Absolute-floor thresholds for a single golden-set run."""
    floor: float = DEFAULT_FLOOR
    min_n: int = DEFAULT_MIN_N


@dataclass(frozen=True)
class QualityGateResult:
    gate: str = "INCONCLUSIVE"  # PASS | FAIL | INCONCLUSIVE
    floor: float = DEFAULT_FLOOR
    faithfulness_mean: Optional[float] = None
    answer_relevance_mean: Optional[float] = None
    n_queries: int = 0
    reasons: List[str] = field(default_factory=list)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def apply_quality_gate(
    summary: Optional[Dict],
    policy: Optional[QualityGatePolicy] = None,
) -> QualityGateResult:
    """Apply the absolute floor to a single run's generation summary.

    Args:
        summary: ``generation_summary`` dict from ``run_eval`` /
            ``summarize_generation_results`` (fields ``faithfulness_mean``,
            ``answer_relevance_mean``, ``n_queries``). ``None`` -> INCONCLUSIVE.
        policy: optional override; defaults to floor=0.80, min_n=10.

    Returns:
        ``QualityGateResult`` with gate PASS (both means >= floor and enough
        samples), FAIL (a mean below floor), or INCONCLUSIVE (missing summary
        or too few samples — cannot decide, never blocks the build).
    """
    policy = policy or QualityGatePolicy()
    reasons: List[str] = []

    if not summary:
        reasons.append("missing generation_summary — no scores to gate on")
        return QualityGateResult(gate="INCONCLUSIVE", floor=policy.floor,
                                 reasons=reasons)

    n = int(summary.get("n_queries", 0) or 0)
    faith = _safe_float(summary.get("faithfulness_mean"))
    rel = _safe_float(summary.get("answer_relevance_mean"))

    if n < policy.min_n:
        reasons.append(
            f"too few samples (n_queries={n} < min {policy.min_n}) — mean is noise"
        )
        return QualityGateResult(gate="INCONCLUSIVE", floor=policy.floor,
                                 faithfulness_mean=faith,
                                 answer_relevance_mean=rel, n_queries=n,
                                 reasons=reasons)

    fail = False
    if faith is None:
        reasons.append("faithfulness_mean missing — cannot gate")
        fail = True
    elif faith < policy.floor:
        reasons.append(
            f"faithfulness_mean {faith:.3f} < floor {policy.floor}"
        )
        fail = True

    if rel is None:
        reasons.append("answer_relevance_mean missing — cannot gate")
        fail = True
    elif rel < policy.floor:
        reasons.append(
            f"answer_relevance_mean {rel:.3f} < floor {policy.floor}"
        )
        fail = True

    verdict = "FAIL" if fail else "PASS"
    return QualityGateResult(
        gate=verdict, floor=policy.floor,
        faithfulness_mean=faith, answer_relevance_mean=rel,
        n_queries=n, reasons=reasons,
    )


def gate_from_report(
    report_path: str | Path,
    policy: Optional[QualityGatePolicy] = None,
) -> QualityGateResult:
    """Load a persisted run JSON and apply the absolute quality gate."""
    p = Path(report_path)
    with p.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return apply_quality_gate(payload.get("generation_summary"), policy)


__all__ = [
    "QualityGatePolicy",
    "QualityGateResult",
    "apply_quality_gate",
    "gate_from_report",
    "DEFAULT_FLOOR",
]