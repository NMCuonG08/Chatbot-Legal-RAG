"""Unit tests for drift: PSI, KL, detect_drift on synthetic run payloads."""
import json

import pytest

from evaluation import drift


def _write_run(path, run_id, routes, verdicts, latencies):
    payload = {
        "run_metadata": {"run_id": run_id},
        "route_distribution": routes,
        "verify_verdict_distribution": verdicts,
        "latency_ms": latencies,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_psi_identical_is_zero():
    d = {"legal_rag": 50, "general_chat": 50}
    assert drift.distribution_psi(d, d) == 0.0


def test_psi_shifted_is_positive():
    base = {"legal_rag": 80, "general_chat": 20}
    recent = {"legal_rag": 20, "general_chat": 80}
    assert drift.distribution_psi(base, recent) > 0.0


def test_psi_empty_returns_zero():
    assert drift.distribution_psi({}, {}) == 0.0
    assert drift.distribution_psi({"a": 1}, {}) == 0.0


def test_kl_identical_is_zero():
    d = {"a": 50, "b": 50}
    assert drift.distribution_kl(d, d) == 0.0


def test_kl_shifted_finite_and_positive():
    base = {"a": 90, "b": 10}
    recent = {"a": 10, "b": 90}
    kl = drift.distribution_kl(base, recent)
    assert kl > 0.0
    assert kl == kl  # finite (not nan)


def test_detect_drift_no_alert_when_identical(tmp_path):
    b = _write_run(tmp_path / "base.json", "b1",
                   {"legal_rag": 50, "general_chat": 50},
                   {"supported": 40, "unsupported": 10},
                   [100, 110, 120, 130, 140])
    r = _write_run(tmp_path / "recent.json", "r1",
                   {"legal_rag": 50, "general_chat": 50},
                   {"supported": 40, "unsupported": 10},
                   [100, 110, 120, 130, 140])
    reports = drift.detect_drift(b, r)
    assert len(reports) == 3
    assert all(not rep.alert for rep in reports)
    assert all(rep.psi == 0.0 for rep in reports)


def test_detect_drift_alert_on_route_shift(tmp_path):
    b = _write_run(tmp_path / "base.json", "b1",
                   {"legal_rag": 80, "general_chat": 20},
                   {"supported": 50, "unsupported": 0},
                   [100, 100, 100])
    r = _write_run(tmp_path / "recent.json", "r1",
                   {"legal_rag": 10, "general_chat": 90},
                   {"supported": 50, "unsupported": 0},
                   [100, 100, 100])
    reports = drift.detect_drift(b, r)
    route_rep = next(rep for rep in reports if rep.metric == "route_distribution")
    assert route_rep.alert is True
    assert route_rep.psi > 0.2
    assert route_rep.baseline_run_id == "b1"
    assert route_rep.recent_run_id == "r1"


def test_detect_drift_skips_missing_metric(tmp_path):
    b = _write_run(tmp_path / "base.json", "b1",
                   {"legal_rag": 50, "general_chat": 50},
                   {"supported": 50, "unsupported": 0},
                   [100, 110, 120])
    r = tmp_path / "recent.json"
    r.write_text(json.dumps({
        "run_metadata": {"run_id": "r1"},
        "route_distribution": {"legal_rag": 50, "general_chat": 50},
        "verify_verdict_distribution": {"supported": 50, "unsupported": 0},
    }), encoding="utf-8")
    reports = drift.detect_drift(b, r)
    metrics = {rep.metric for rep in reports}
    assert "latency_ms" in metrics  # base has it -> compared (recent empty)


def test_drift_report_frozen(tmp_path):
    b = _write_run(tmp_path / "base.json", "b1",
                   {"legal_rag": 50}, {"supported": 50}, [100])
    r = _write_run(tmp_path / "recent.json", "r1",
                   {"legal_rag": 50}, {"supported": 50}, [100])
    rep = drift.detect_drift(b, r)[0]
    with pytest.raises(Exception):
        rep.alert = True