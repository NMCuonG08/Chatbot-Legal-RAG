"""DeepEval gate calibration (marker: redteam). Exercises GEval + FaithfulnessMetric
structurally with a mocked model so the offline gate validates the wiring without
a live LLM. Skips when deepeval is not installed.
"""
import pytest

pytestmark = pytest.mark.redteam


class _MockFaithfulness:
    """Mimics deepeval.metrics.FaithfulnessMetric with a deterministic rule."""
    name = "faithfulness"
    score = 0.0
    reason = ""
    def measure(self, test_case):
        output = (test_case.actual_output or "").lower()
        ctx = " ".join(test_case.retrieval_context or []).lower()
        supported = all(w in ctx for w in output.split()[:5]) if output.strip() else False
        self.score = 1.0 if supported else 0.2
        self.reason = "mocked faithfulness"
        return self.score
    def async_measure(self, test_case):
        return self.measure(test_case)


class _MockGEval:
    """Mimics deepeval.metrics.GEval (CoT-before-score) with a length proxy."""
    name = "answer_relevance"
    score = 0.0
    reason = ""
    def __init__(self, criteria=None):
        self.criteria = criteria or "relevance"
    def measure(self, test_case):
        self.score = min(1.0, len(test_case.actual_output or "") / 200.0)
        self.reason = "mocked geval"
        return self.score
    def async_measure(self, test_case):
        return self.measure(test_case)


def test_faithfulness_metric_supported_when_grounded():
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase
    case = LLMTestCase(
        input="Điều 10 nói gì?",
        actual_output="Điều 10 quy định hiệu lực",
        retrieval_context=["Điều 10 quy định hiệu lực của Bộ luật Dân sự"],
    )
    m = _MockFaithfulness()
    assert m.measure(case) == 1.0


def test_faithfulness_metric_low_when_ungrounded():
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase
    case = LLMTestCase(
        input="Điều 10 nói gì?",
        actual_output="Theo cảm nhận thì nên làm vậy",
        retrieval_context=["Điều 10 quy định hiệu lực"],
    )
    m = _MockFaithfulness()
    assert m.measure(case) < 0.5


def test_geval_scores_monotonically_with_length():
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase
    m = _MockGEval()
    short = LLMTestCase(input="q", actual_output="ngắn")
    long = LLMTestCase(input="q", actual_output="x" * 200)
    assert m.measure(short) < m.measure(long)


def test_assert_test_passes_with_mock_metrics():
    """Exercise deepeval.assert_test wiring with mocked metrics (no live model)."""
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase
    from deepeval import assert_test

    case = LLMTestCase(
        input="Điều 10 nói gì?",
        actual_output="Điều 10 quy định hiệu lực",
        retrieval_context=["Điều 10 quy định hiệu lực của Bộ luật Dân sự"],
    )
    assert_test(case, [_MockFaithfulness()])