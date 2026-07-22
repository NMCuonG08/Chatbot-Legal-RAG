"""Regression diff between two eval runs + a merge gate.

Loads two persisted run JSONs (baseline + candidate), aligns them by
``sample_id``, and applies significance tests to decide whether the candidate
regressed vs. baseline:

- success_rate (binary): ``paired_mcnemar``
- faithfulness / answer_relevance (continuous): ``wilcoxon`` + bootstrap CI

A ``GatePolicy`` then turns the deltas + p-values into a PASS / FAIL /
INCONCLUSIVE verdict. Refuses to compare runs with mismatched ``eval_version``
(schemas that aren't comparable).

Public surface:
- ``GatePolicy``, ``MetricDelta``, ``RegressionReport`` — frozen records.
- ``load_run`` — load + lightly validate a run JSON.
- ``diff_runs`` — compute per-metric deltas + significance.
- ``apply_gate`` — policy decision from a ``RegressionReport``.
- ``write_regression_report`` — markdown render.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.stats import (
    bootstrap_ci,
    holm_bonferroni,
    paired_mcnemar,
    wilcoxon,
)

logger = logging.getLogger(__name__)

# Metrics compared as paired continuous scores (Wilcoxon + bootstrap CI).
_SCORE_METRICS = ("faithfulness", "answer_relevance")
# Minimum paired samples for a non-INCONCLUSIVE verdict.
_MIN_N_FOR_GATE = 10


@dataclass(frozen=True)
class GatePolicy:
    """Thresholds that turn a diff into a merge decision."""
    min_success_rate: float = 0.8
    # Audit 4.1: raised 0.7 -> 0.80 to match the golden-set quality gate floor
    # (evaluation.quality_gate.DEFAULT_FLOOR). The regression gate already
    # blocks *relative* regressions; this raises the *absolute* floor so a
    # candidate cannot pass by merely failing to regress vs. a bad baseline.
    min_faithfulness: float = 0.80
    max_regression_rel: float = -0.05  # >5% relative drop = regression
    alpha: float = 0.05
    min_n: int = _MIN_N_FOR_GATE


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    candidate: float
    delta: float
    relative: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: Optional[float]
    significant: Optional[bool]  # None when no test applies


@dataclass(frozen=True)
class RegressionReport:
    baseline_run_id: str
    candidate_run_id: str
    eval_version_match: bool
    n_paired: int
    deltas: List[MetricDelta] = field(default_factory=list)
    gate: str = "INCONCLUSIVE"  # PASS | FAIL | INCONCLUSIVE
    gate_reasons: List[str] = field(default_factory=list)


def load_run(path: str | Path) -> dict:
    """Load a run JSON; returns the parsed dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _eval_version(run: dict) -> Optional[str]:
    meta = run.get("run_metadata") or {}
    return meta.get("eval_version")


def _run_id(run: dict) -> str:
    meta = run.get("run_metadata") or {}
    return meta.get("run_id") or "unknown"


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _per_sample_success(run: dict) -> Dict[str, bool]:
    """sample_id -> success bool from e2e_results (no error + has answer)."""
    out: Dict[str, bool] = {}
    for r in run.get("e2e_results", []) or []:
        sid = r.get("sample_id")
        if not sid:
            continue
        out[sid] = (not r.get("error")) and bool(r.get("answer"))
    return out


def _per_sample_score(run: dict, key: str) -> Dict[str, float]:
    """sample_id -> score from generation_results[].scores[key]."""
    out: Dict[str, float] = {}
    for r in run.get("generation_results", []) or []:
        sid = r.get("sample_id")
        if not sid:
            continue
        val = _safe_float((r.get("scores") or {}).get(key))
        if val is not None:
            out[sid] = val
    return out


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def diff_runs(
    baseline_path: str | Path,
    candidate_path: str | Path,
    policy: Optional[GatePolicy] = None,
) -> RegressionReport:
    """Compute per-metric deltas + significance between two runs."""
    policy = policy or GatePolicy()
    base = load_run(baseline_path)
    cand = load_run(candidate_path)

    base_id = _run_id(base)
    cand_id = _run_id(cand)
    version_match = _eval_version(base) == _eval_version(cand)

    deltas: List[MetricDelta] = []
    p_values: List[float] = []

    # --- success_rate (binary, McNemar) ---
    base_ok = _per_sample_success(base)
    cand_ok = _per_sample_success(cand)
    paired_ids = sorted(set(base_ok) & set(cand_ok))
    n = len(paired_ids)
    if n:
        b_succ = [base_ok[i] for i in paired_ids]
        c_succ = [cand_ok[i] for i in paired_ids]
        mc = paired_mcnemar(b_succ, c_succ)
        b_rate = _mean([float(x) for x in b_succ])
        c_rate = _mean([float(x) for x in c_succ])
        delta = c_rate - b_rate
        rel = (delta / b_rate) if b_rate else 0.0
        deltas.append(MetricDelta(
            metric="success_rate", baseline=b_rate, candidate=c_rate,
            delta=delta, relative=rel, ci_lower=None, ci_upper=None,
            p_value=mc.p_value, significant=mc.p_value < policy.alpha,
        ))
        p_values.append(mc.p_value)

    # --- continuous scores (Wilcoxon + bootstrap CI) ---
    for key in _SCORE_METRICS:
        b_scores = _per_sample_score(base, key)
        c_scores = _per_sample_score(cand, key)
        ids = sorted(set(b_scores) & set(c_scores))
        if not ids:
            continue
        b_vals = [b_scores[i] for i in ids]
        c_vals = [c_scores[i] for i in ids]
        b_mean = _mean(b_vals)
        c_mean = _mean(c_vals)
        delta = c_mean - b_mean
        rel = (delta / b_mean) if b_mean else 0.0
        ci = bootstrap_ci([c - b for c, b in zip(c_vals, b_vals)],
                          seed=42, n_boot=1000)
        try:
            w = wilcoxon(c_vals, b_vals)
            p = w.p_value
        except Exception:
            p = None
        deltas.append(MetricDelta(
            metric=key, baseline=b_mean, candidate=c_mean,
            delta=delta, relative=rel,
            ci_lower=ci.lower, ci_upper=ci.upper,
            p_value=p, significant=(p is not None and p < policy.alpha),
        ))
        if p is not None:
            p_values.append(p)

    # Holm-Bonferroni across the tests we ran.
    if p_values:
        holm = holm_bonferroni(p_values, alpha=policy.alpha)
        idx = 0
        for j, d in enumerate(deltas):
            if d.p_value is not None:
                d_sig = holm[idx].adjusted_p <= policy.alpha
                deltas[j] = MetricDelta(
                    metric=d.metric, baseline=d.baseline, candidate=d.candidate,
                    delta=d.delta, relative=d.relative,
                    ci_lower=d.ci_lower, ci_upper=d.ci_upper,
                    p_value=d.p_value, significant=d_sig,
                )
                idx += 1

    report = RegressionReport(
        baseline_run_id=base_id, candidate_run_id=cand_id,
        eval_version_match=version_match, n_paired=n,
        deltas=deltas, gate="INCONCLUSIVE", gate_reasons=[],
    )
    report = apply_gate(report, policy)
    return report


def apply_gate(report: RegressionReport, policy: Optional[GatePolicy] = None) -> RegressionReport:
    """Apply the policy to a report, returning a new report with a verdict."""
    policy = policy or GatePolicy()
    reasons: List[str] = []

    if not report.eval_version_match:
        reasons.append("eval_version mismatch — runs are not comparable")
    if report.n_paired < policy.min_n:
        reasons.append(f"too few paired samples ({report.n_paired} < {policy.min_n})")

    if reasons:
        return RegressionReport(
            baseline_run_id=report.baseline_run_id,
            candidate_run_id=report.candidate_run_id,
            eval_version_match=report.eval_version_match,
            n_paired=report.n_paired, deltas=report.deltas,
            gate="INCONCLUSIVE", gate_reasons=reasons,
        )

    fail = False
    for d in report.deltas:
        if d.metric == "success_rate":
            if d.candidate < policy.min_success_rate:
                reasons.append(
                    f"success_rate {d.candidate:.3f} < min {policy.min_success_rate}"
                )
                fail = True
            if d.significant and d.relative < policy.max_regression_rel:
                reasons.append(
                    f"significant success_rate regression rel={d.relative:.3f}"
                )
                fail = True
        elif d.metric == "faithfulness":
            if d.candidate < policy.min_faithfulness:
                reasons.append(
                    f"faithfulness {d.candidate:.3f} < min {policy.min_faithfulness}"
                )
                fail = True
            if d.significant and d.relative < policy.max_regression_rel:
                reasons.append(
                    f"significant faithfulness regression rel={d.relative:.3f}"
                )
                fail = True
        elif d.metric == "answer_relevance":
            if d.significant and d.relative < policy.max_regression_rel:
                reasons.append(
                    f"significant answer_relevance regression rel={d.relative:.3f}"
                )
                fail = True

    verdict = "FAIL" if fail else "PASS"
    return RegressionReport(
        baseline_run_id=report.baseline_run_id,
        candidate_run_id=report.candidate_run_id,
        eval_version_match=report.eval_version_match,
        n_paired=report.n_paired, deltas=report.deltas,
        gate=verdict, gate_reasons=reasons,
    )


def write_regression_report(report: RegressionReport, out_path: str | Path) -> Path:
    """Render a RegressionReport to markdown."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Regression Report",
        "",
        f"- baseline: `{report.baseline_run_id}`",
        f"- candidate: `{report.candidate_run_id}`",
        f"- eval_version_match: {report.eval_version_match}",
        f"- n_paired: {report.n_paired}",
        f"- gate: **{report.gate}**",
        "",
    ]
    if report.deltas:
        lines.append("| metric | baseline | candidate | delta | relative | p_value | significant |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for d in report.deltas:
            pv = "n/a" if d.p_value is None else f"{d.p_value:.4f}"
            sig = "n/a" if d.significant is None else str(d.significant)
            lines.append(
                f"| {d.metric} | {d.baseline:.4f} | {d.candidate:.4f} | "
                f"{d.delta:+.4f} | {d.relative:+.3f} | {pv} | {sig} |"
            )
        lines.append("")
    if report.gate_reasons:
        lines.append("## Reasons")
        for r in report.gate_reasons:
            lines.append(f"- {r}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


__all__ = [
    "GatePolicy",
    "MetricDelta",
    "RegressionReport",
    "load_run",
    "diff_runs",
    "apply_gate",
    "write_regression_report",
]