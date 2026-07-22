"""Audit 4.1 — golden-set absolute quality gate (blocking PR floor).

The regression gate only diffs a candidate vs. a *baseline* run; it has no
absolute floor, so a green PR can still ship a model whose average
faithfulness/relevance is, say, 0.5 — as long as it did not regress vs. an
equally-bad baseline. The golden-set quality gate closes that hole: it reads a
single run's ``generation_summary`` and FAILs when the mean faithfulness OR
answer_relevance drops below the floor (default 0.80). This is the signal the
CI quality-gate workflow exits 1 on.

Pure-logic, no live services. Reuses the run payload shape produced by
``run_eval`` / ``summarize_generation_results``.
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from evaluation.quality_gate import (
    QualityGatePolicy,
    apply_quality_gate,
    gate_from_report,
)


def _summary(faith, rel, n=20):
    return {
        "n_queries": n,
        "faithfulness_mean": faith,
        "answer_relevance_mean": rel,
    }


# ---------------------------------------------------------------------------
def test_both_above_floor_passes():
    res = apply_quality_gate(_summary(0.85, 0.82))
    assert res.gate == "PASS"
    assert res.reasons == []


def test_faithfulness_below_floor_fails():
    res = apply_quality_gate(_summary(0.65, 0.90))
    assert res.gate == "FAIL"
    assert any("faithfulness" in r for r in res.reasons)


def test_relevance_below_floor_fails():
    res = apply_quality_gate(_summary(0.90, 0.55))
    assert res.gate == "FAIL"
    assert any("answer_relevance" in r for r in res.reasons)


def test_floor_is_inclusive_at_boundary():
    # 0.80 exactly meets the floor (>=), so it must PASS.
    res = apply_quality_gate(_summary(0.80, 0.80))
    assert res.gate == "PASS"


def test_custom_floor_honored():
    res = apply_quality_gate(_summary(0.72, 0.95),
                             QualityGatePolicy(floor=0.70))
    assert res.gate == "PASS"


def test_missing_summary_is_inconclusive():
    res = apply_quality_gate(None)
    assert res.gate == "INCONCLUSIVE"
    assert res.reasons  # explains the missing summary


def test_too_few_samples_is_inconclusive():
    # A single-sample mean is noise, not a gate signal.
    res = apply_quality_gate(_summary(0.90, 0.90, n=1))
    assert res.gate == "INCONCLUSIVE"
    assert any("n_queries" in r or "samples" in r for r in res.reasons)


def test_gate_from_report_reads_summary(tmp_path: Path):
    payload = {"generation_summary": _summary(0.50, 0.60)}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    res = gate_from_report(p)
    assert res.gate == "FAIL"
    assert any("faithfulness" in r for r in res.reasons)