"""Unit tests for regression: diff_runs, apply_gate, write_regression_report."""
import json
from pathlib import Path

from evaluation.regression import (
    MetricDelta,
    RegressionReport,
    apply_gate,
    diff_runs,
    write_regression_report,
)
from evaluation.run_metadata import EVAL_VERSION


def _run_json(run_id, *, success_ids, fail_ids, scores=None):
    """Build a synthetic run payload."""
    e2e = []
    for sid in success_ids:
        e2e.append({"sample_id": sid, "answer": "ok", "error": None})
    for sid in fail_ids:
        e2e.append({"sample_id": sid, "answer": "", "error": "boom"})
    gen = []
    for sid, sc in (scores or {}).items():
        gen.append({"sample_id": sid, "scores": {
            "faithfulness": sc[0], "answer_relevance": sc[1]}})
    return {
        "run_metadata": {"run_id": run_id, "eval_version": EVAL_VERSION},
        "e2e_results": e2e,
        "e2e_summary": {"success_rate": len(success_ids) / max(1, len(e2e))},
        "generation_results": gen,
        "generation_summary": {"faithfulness_mean": 0.0},
    }


def _write_run(tmp_path: Path, run_id: str, payload: dict) -> Path:
    p = tmp_path / f"{run_id}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_diff_pass_when_identical(tmp_path):
    ids = [f"t{i}" for i in range(15)]
    base = _write_run(tmp_path, "base", _run_json("base", success_ids=ids, fail_ids=[]))
    cand = _write_run(tmp_path, "cand", _run_json("cand", success_ids=ids, fail_ids=[]))
    report = diff_runs(base, cand)
    assert report.eval_version_match is True
    assert report.n_paired == 15
    assert report.gate == "PASS"


def test_diff_fail_on_success_rate_drop(tmp_path):
    ids = [f"t{i}" for i in range(20)]
    base = _write_run(tmp_path, "base", _run_json("base", success_ids=ids, fail_ids=[]))
    cand = _write_run(tmp_path, "cand",
                      _run_json("cand", success_ids=ids[:10], fail_ids=ids[10:]))
    report = diff_runs(base, cand)
    assert report.gate == "FAIL"
    assert any("success_rate" in r for r in report.gate_reasons)


def test_diff_inconclusive_on_version_mismatch(tmp_path):
    ids = [f"t{i}" for i in range(15)]
    base_p = _run_json("base", success_ids=ids, fail_ids=[])
    base_p["run_metadata"]["eval_version"] = "0.0.0"
    base = _write_run(tmp_path, "base", base_p)
    cand = _write_run(tmp_path, "cand", _run_json("cand", success_ids=ids, fail_ids=[]))
    report = diff_runs(base, cand)
    assert report.gate == "INCONCLUSIVE"
    assert any("eval_version" in r for r in report.gate_reasons)


def test_diff_inconclusive_on_small_n(tmp_path):
    ids = [f"t{i}" for i in range(3)]  # < min_n (10)
    base = _write_run(tmp_path, "base", _run_json("base", success_ids=ids, fail_ids=[]))
    cand = _write_run(tmp_path, "cand", _run_json("cand", success_ids=ids, fail_ids=[]))
    report = diff_runs(base, cand)
    assert report.gate == "INCONCLUSIVE"


def test_write_regression_report_renders(tmp_path):
    ids = [f"t{i}" for i in range(15)]
    base = _write_run(tmp_path, "base", _run_json("base", success_ids=ids, fail_ids=[]))
    cand = _write_run(tmp_path, "cand", _run_json("cand", success_ids=ids, fail_ids=[]))
    report = diff_runs(base, cand)
    out = write_regression_report(report, tmp_path / "regression_report.md")
    text = out.read_text(encoding="utf-8")
    assert "# Regression Report" in text
    assert "PASS" in text


def test_apply_gate_respects_faithfulness_floor():
    deltas = [MetricDelta("faithfulness", 0.9, 0.5, -0.4, -0.44,
                          None, None, 0.01, True)]
    rep = RegressionReport("b", "c", True, 20, deltas=deltas)
    out = apply_gate(rep)
    assert out.gate == "FAIL"