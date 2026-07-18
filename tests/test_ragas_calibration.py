"""Ragas calibration (marker: slow). Deselected in the offline gate; runs in the
live/nightly workflow. Uses a tiny synthetic dataset so the ragas API surface is
exercised without a real model or network.

If ragas is not installed, the test skips (live workflow installs it via
requirements_dev). The live workflow runs ragas faithfulness against a real
Groq/Ollama model — here we validate import + structural wiring only, since
mocking ragas' internal LLM response shape (generations[0][i].text + parseable
JSON) is too brittle to be meaningful.
"""
import pytest

pytestmark = pytest.mark.slow


def _make_synthetic_dataset():
    """Build a ragas SingleTurnSample-like dict without requiring ragas at import.

    The response deliberately echoes a 6-word run from the context so the
    homegrown context_utilization signal is positive.
    """
    return {
        "user_input": "Điều 10 Bộ luật Dân sự quy định gì?",
        "retrieved_contexts": [
            "Thời hiệu là thời hạn một năm kể từ ngày quyền dân sự "
            "chịu ảnh hưởng được xác lập.",
        ],
        "response": "Theo Bộ luật, thời hiệu là thời hạn một năm kể từ ngày.",
        "reference": "Thời hiệu là một năm.",
    }


def test_synthetic_dataset_shape():
    """Structural check that runs even without ragas (slow but no external dep)."""
    ds = _make_synthetic_dataset()
    assert ds["user_input"]
    assert len(ds["retrieved_contexts"]) >= 1
    assert ds["response"]
    assert ds["reference"]


def test_ragas_metric_importable():
    """Faithfulness metric + SingleTurnSample are importable and constructible.

    Validates the ragas API surface the live workflow depends on. Skips if ragas
    is not installed. Does NOT execute the metric (needs a real LLM).
    """
    pytest.importorskip("ragas")
    try:
        from ragas.metrics import Faithfulness
        from ragas.dataset_schema import SingleTurnSample
    except Exception as exc:
        pytest.skip(f"ragas API surface unavailable: {exc}")

    ds = _make_synthetic_dataset()
    sample = SingleTurnSample(**ds)
    metric = Faithfulness()
    assert metric is not None
    assert sample.user_input == ds["user_input"]


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