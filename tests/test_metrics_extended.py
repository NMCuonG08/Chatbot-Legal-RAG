"""Unit tests for metrics_extended: tool-call accuracy, context utilization,
hallucination rate, latency p99, noise sensitivity (mocked judge)."""
from evaluation import metrics_extended as me


def test_tool_call_accuracy_right_tool_and_args():
    calls = [{"name": "search_law", "args": {"query": "điều 10"}},
             {"name": "calculator", "args": {"expr": "1+1"}}]
    score = me.tool_call_accuracy(
        calls, ["search_law"],
        expected_args={"search_law": {"query": "điều 10"}})
    assert score.right_tool == 1.0
    assert score.right_args == 1.0
    assert score.score == 1.0


def test_tool_call_accuracy_wrong_tool():
    calls = [{"name": "calculator", "args": {}}]
    score = me.tool_call_accuracy(calls, ["search_law"])
    assert score.right_tool == 0.0
    assert score.right_args == 1.0  # no expected_args -> neutral 1.0
    assert score.score == 0.5


def test_tool_call_accuracy_partial_args():
    calls = [{"name": "search_law", "args": {"query": "x"}}]
    score = me.tool_call_accuracy(
        calls, ["search_law"],
        expected_args={"search_law": {"query": "điều 10"}})
    assert score.right_args == 0.0
    assert score.score == 0.5


def test_context_utilization_hit():
    ctx = ["Bộ luật Dân sự quy định hiệu lực theo Điều 10"]
    ans = "Theo Bộ luật Dân sự quy định hiệu lực theo Điều 10, ..."
    assert me.context_utilization(ans, ctx) == 1.0


def test_context_utilization_miss():
    ctx = ["Hoàn toàn không liên quan văn bản khác ở đây"]
    ans = "Kết quả là cái gì đó rất khác biệt không trùng khớp"
    assert me.context_utilization(ans, ctx) == 0.0


def test_context_utilization_empty():
    assert me.context_utilization("ans", []) == 0.0
    assert me.context_utilization("", ["ctx"]) == 0.0


def test_hallucination_rate():
    scores = [0.9, 0.8, 0.3, 0.1, 0.6]
    # <0.5: 0.3, 0.1 -> 2/5
    assert me.hallucination_rate(scores, threshold=0.5) == 0.4
    assert me.hallucination_rate([]) == 0.0


def test_latency_p99_nearest_rank():
    # 100 samples 1..100, p99 -> rank ceil(99)-1 = 98 -> value 99
    lat = list(range(1, 101))
    assert me.latency_p99(lat) == 99.0


def test_latency_p99_empty():
    assert me.latency_p99([]) == 0.0


def test_noise_sensitivity_robust(monkeypatch):
    """Judge returns same faithfulness under noise -> delta 0, score 1."""
    def fake_evaluate(question, answer, contexts, judge_fn=None):
        class R:
            score = 0.9
        return R()

    import evaluation.metrics_generation as mg
    monkeypatch.setattr(mg, "evaluate_faithfulness", fake_evaluate)
    res = me.noise_sensitivity("q", "a", ["c1"], ["c2"], judge_fn=lambda *a, **k: None)
    assert res.delta == 0.0
    assert res.score == 1.0