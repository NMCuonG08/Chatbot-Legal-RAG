"""Audit 4.2 — verify_answer Prometheus metrics + rejection alerting.

verify_answer judged every final answer but exported NO metrics, so the MLOps
team was blind to the hallucination/rejection rate. judge_answer now records:
- verify_judged_total         (denominator)
- verify_rejection_total{verdict}  (partial/unsupported = rejected)
- verify_faithfulness_score   (histogram of the faithfulness score)

The recording decorator is guarded: a metric failure never breaks the graph or
alters the returned verdict dict.
"""
import verify_answer
import memory_metrics


class _JR:
    def __init__(self, score, rationale="r"):
        self.score = score
        self.rationale = rationale


def _record_spy(monkeypatch):
    rec = {"judged": 0, "rejections": [], "scores": []}

    def _j():
        rec["judged"] += 1

    def _r(verdict):
        rec["rejections"].append(verdict)

    def _s(score):
        rec["scores"].append(score)

    monkeypatch.setattr(memory_metrics, "inc_verify_judged", _j)
    monkeypatch.setattr(memory_metrics, "inc_verify_rejection", _r)
    monkeypatch.setattr(memory_metrics, "observe_verify_score", _s)
    return rec


def _judge_with_score(monkeypatch, score):
    monkeypatch.setattr(verify_answer, "evaluate_faithfulness", lambda *a, **k: _JR(score))
    return verify_answer.judge_answer(
        "q", "a real answer here with enough length", [{"content": "ctx"}],
        judge_fn=lambda msgs: "x",
    )


# ---------------------------------------------------------------------------
def test_supported_verdict_records_judged_score_no_rejection(monkeypatch):
    rec = _record_spy(monkeypatch)
    out = _judge_with_score(monkeypatch, 0.9)
    assert out["verdict"] == "supported"
    assert rec["judged"] == 1
    assert rec["scores"] == [0.9]
    assert rec["rejections"] == []  # supported -> not a rejection


def test_partial_verdict_records_rejection(monkeypatch):
    rec = _record_spy(monkeypatch)
    out = _judge_with_score(monkeypatch, 0.5)  # 0.35 <= 0.5 < 0.7 -> partial
    assert out["verdict"] == "partial"
    assert rec["rejections"] == ["partial"]
    assert rec["judged"] == 1


def test_unsupported_verdict_records_rejection(monkeypatch):
    rec = _record_spy(monkeypatch)
    out = _judge_with_score(monkeypatch, 0.2)  # < 0.35 -> unsupported
    assert out["verdict"] == "unsupported"
    assert rec["rejections"] == ["unsupported"]
    assert rec["scores"] == [0.2]


def test_empty_answer_records_rejection(monkeypatch):
    rec = _record_spy(monkeypatch)
    out = verify_answer.judge_answer("q", "", [{"content": "ctx"}])
    assert out["verdict"] == "unsupported"
    assert rec["judged"] == 1
    assert rec["rejections"] == ["unsupported"]
    assert rec["scores"] == [0.0]


def test_metric_failure_does_not_break_judge(monkeypatch):
    def _boom():
        raise RuntimeError("prom down")
    monkeypatch.setattr(memory_metrics, "inc_verify_judged", _boom)
    monkeypatch.setattr(verify_answer, "evaluate_faithfulness", lambda *a, **k: _JR(0.9))
    out = verify_answer.judge_answer(
        "q", "a real answer here with enough length", [{"content": "ctx"}],
        judge_fn=lambda msgs: "x",
    )
    # verdict still returned despite metric error
    assert out["verdict"] == "supported"


# ---------------------------------------------------------------------------
# Smoke: wrappers exist + do not raise
# ---------------------------------------------------------------------------
def test_wrappers_are_callable_no_raise():
    memory_metrics.inc_verify_judged()
    memory_metrics.inc_verify_rejection("unsupported")
    memory_metrics.observe_verify_score(0.42)