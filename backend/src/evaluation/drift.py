"""Distribution drift detection between two eval runs.

Compares a baseline run payload against a recent run payload across categorical
distributions (route, verify_verdict, tool_success) and latency histograms using
PSI (Population Stability Index) and symmetric KL divergence. Flags a drift
alert when either statistic exceeds its threshold.

Run payloads are the JSON written by ``run_eval`` (``eval_reports/runs/<id>.json``).
Only the metric fields listed below are consumed; missing fields are skipped.

Public surface:
- ``DriftReport`` frozen.
- ``distribution_psi``, ``distribution_kl``, ``detect_drift``.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

EPS = 1e-6
DEFAULT_PSI_THRESHOLD = 0.2   # >0.2 = significant drift (industry convention)
DEFAULT_KL_THRESHOLD = 0.5

# Categorical metrics in the run payload whose distribution we compare.
_CATEGORICAL_METRICS = ("route_distribution", "verify_verdict_distribution")
# Latency bucketed into a histogram for PSI/KL.
_LATENCY_METRIC = "latency_ms"


@dataclass(frozen=True)
class DriftReport:
    metric: str
    baseline_run_id: Optional[str]
    recent_run_id: Optional[str]
    psi: float
    kl: float
    threshold: float
    alert: bool
    detail: Dict[str, float] = field(default_factory=dict)


def _normalize(counts: Dict[str, float]) -> Dict[str, float]:
    """Normalize a count/freq dict to a probability distribution (sums to 1)."""
    total = sum(counts.values()) if counts else 0.0
    if total <= 0:
        return {}
    return {k: max(v / total, EPS) for k, v in counts.items()}


def _aligned_keys(a: Dict[str, float], b: Dict[str, float]) -> List[str]:
    return sorted(set(a.keys()) | set(b.keys()))


def distribution_psi(baseline: Dict[str, float],
                     recent: Dict[str, float]) -> float:
    """PSI for categorical distributions. 0 = identical. Larger = more drift."""
    p = _normalize(baseline)
    q = _normalize(recent)
    if not p or not q:
        return 0.0
    keys = _aligned_keys(p, q)
    psi = 0.0
    for k in keys:
        pk = max(p.get(k, EPS), EPS)
        qk = max(q.get(k, EPS), EPS)
        psi += (qk - pk) * math.log(qk / pk)
    return float(psi)


def distribution_kl(baseline: Dict[str, float],
                    recent: Dict[str, float]) -> float:
    """Symmetric KL divergence (average of both directions)."""
    p = _normalize(baseline)
    q = _normalize(recent)
    if not p or not q:
        return 0.0
    keys = _aligned_keys(p, q)

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            ak = max(a.get(k, EPS), EPS)
            bk = max(b.get(k, EPS), EPS)
            s += ak * math.log(ak / bk)
        return s

    return float((_kl(p, q) + _kl(q, p)) / 2.0)


def _latency_to_hist(values: Sequence[float], bins: int = 10) -> Dict[str, float]:
    """Bucket latency values into equal-width histogram bins keyed by edge label."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {"bin0": float(len(vals))}
    width = (hi - lo) / bins
    hist: Dict[str, float] = {f"bin{i}": 0.0 for i in range(bins)}
    for v in vals:
        idx = min(bins - 1, int((v - lo) / width))
        hist[f"bin{idx}"] += 1.0
    return hist


def _load_run_payload(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Run payload not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _extract_metric(payload: dict, metric: str) -> Dict[str, float]:
    """Pull a distribution metric out of the run payload as a {label: count} dict."""
    val = payload.get(metric)
    if val is None:
        return {}
    if isinstance(val, dict):
        return {str(k): float(v) for k, v in val.items() if v is not None}
    if isinstance(val, list):
        return _latency_to_hist(val)
    return {}


def detect_drift(
    baseline_run_path: Path | str,
    recent_run_path: Path | str,
    metrics: Optional[Sequence[str]] = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    kl_threshold: float = DEFAULT_KL_THRESHOLD,
) -> List[DriftReport]:
    """Compare two run payloads. Returns one DriftReport per comparable metric."""
    base = _load_run_payload(baseline_run_path)
    recent = _load_run_payload(recent_run_path)
    base_id = (base.get("run_metadata") or {}).get("run_id")
    recent_id = (recent.get("run_metadata") or {}).get("run_id")

    metric_names = list(metrics) if metrics else (
        list(_CATEGORICAL_METRICS) + [_LATENCY_METRIC]
    )
    reports: List[DriftReport] = []
    for m in metric_names:
        b_dist = _extract_metric(base, m)
        r_dist = _extract_metric(recent, m)
        if not b_dist and not r_dist:
            continue  # metric absent in both -> skip
        psi = distribution_psi(b_dist, r_dist)
        kl = distribution_kl(b_dist, r_dist)
        alert = psi > psi_threshold or kl > kl_threshold
        threshold = psi_threshold if m != _LATENCY_METRIC else kl_threshold
        reports.append(DriftReport(
            metric=m, baseline_run_id=base_id, recent_run_id=recent_id,
            psi=psi, kl=kl, threshold=threshold, alert=alert,
            detail={"baseline_n": sum(b_dist.values()),
                    "recent_n": sum(r_dist.values())},
        ))
    return reports


__all__ = [
    "DriftReport",
    "distribution_psi",
    "distribution_kl",
    "detect_drift",
]