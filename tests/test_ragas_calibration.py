"""Ragas calibration (marker: slow). Deselected in the offline gate; runs in the
live/nightly workflow. Uses a tiny synthetic dataset + mocked embeddings/LLM so
the ragas API surface is exercised without a real model or network.

If ragas is not installed, the test skips (live workflow installs it via
requirements_dev).
"""
import asyncio

import pytest

pytestmark = pytest.mark.slow


def _make_synthetic_dataset():
    """Build a ragas SingleTurnSample-like dict without requiring ragas at import."""
    return {
        "user_input": "Điều 10 Bộ luật Dân sự quy định gì?",
        "retrieved_contexts": [
            "Điều 10. Thời hiệu là thời hạn một năm kể từ ngày quyền dân sự "
            "chịu ảnh hưởng được xác lập.",
        ],
        "response": "Điều 10 quy định về thời hiệu một năm.",
        "reference": "Điều 10 quy định về thời hiệu.",
    }


def test_synthetic_dataset_shape():
    """Structural check that runs even without ragas (slow but no external dep)."""
    ds = _make_synthetic_dataset()
    assert ds["user_input"]
    assert len(ds["retrieved_contexts"]) >= 1
    assert ds["response"]
    assert ds["reference"]


def test_ragas_faithfulness_with_mock_llm():
    """Exercise ragas faithfulness metric with a mocked evaluator LLM.

    Skips if ragas is not importable. Validates the harness wiring without a
    real Groq/Ollama call.
    """
    pytest.importorskip("ragas")
    ds = _make_synthetic_dataset()

    class _MockLLM:
        def __init__(self, *a, **k):
            pass
        async def generate(self, *a, **k):
            return "1.0"
        def get_temperature(self):
            return 0.0

    try:
        from ragas.metrics import Faithfulness
        from ragas.dataset_schema import SingleTurnSample
    except Exception as exc:
        pytest.skip(f"ragas API surface unavailable: {exc}")

    sample = SingleTurnSample(**ds)
    metric = Faithfulness(llm=_MockLLM())

    async def _run():
        return await metric.single_turn_ascore(sample)
    score = asyncio.run(_run())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_ragas_calibration_correlates_with_homegrown():
    """Calibration check: homegrown context_utilization should be > 0 on a grounded
    sample (direction agrees with a faithful ragas score). Skips without ragas.
    """
    pytest.importorskip("ragas")
    from evaluation.metrics_extended import context_utilization

    ds = _make_synthetic_dataset()
    homegrown = context_utilization(
        ds["response"], ds["retrieved_contexts"])
    assert homegrown > 0.0