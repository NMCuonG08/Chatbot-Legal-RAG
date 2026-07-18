"""Graph loop-control tests (wall-clock deadline + recursion_limit).

Marker: unit. No services needed — exercises ``_invoke_with_deadline`` and the
config wiring directly. ``run_chat_graph`` itself needs the full graph + DB, so
we test the deadline helper + retry classification in isolation.
"""
import time

import pytest

pytestmark = pytest.mark.unit


def test_invoke_with_deadline_returns_on_fast_call():
    from graph_loop_control import _invoke_with_deadline

    class _Graph:
        def invoke(self, state, config):
            return {"ok": True, "state": state, "recursion_limit": config["recursion_limit"]}

    g = _Graph()
    out = _invoke_with_deadline(g, {"q": 1}, {"recursion_limit": 32}, timeout_s=5.0)
    assert out["ok"] is True
    assert out["recursion_limit"] == 32


def test_invoke_with_deadline_raises_graph_run_timeout_on_slow_call():
    from graph_loop_control import GraphRunTimeout, _invoke_with_deadline

    class _SlowGraph:
        def invoke(self, state, config):
            time.sleep(2.0)
            return {"ok": True}

    with pytest.raises(GraphRunTimeout):
        _invoke_with_deadline(_SlowGraph(), {"q": 1}, {"recursion_limit": 32}, timeout_s=0.2)


def test_graph_run_timeout_is_not_retryable():
    """A hung graph must NOT be retried (would hang again) — propagate to degrade."""
    from retry_utils import is_retryable_exception
    from graph_loop_control import GraphRunTimeout

    assert is_retryable_exception(GraphRunTimeout("hang")) is False


def test_config_recursion_limit_tunable():
    """GRAPH_RECURSION_LIMIT / GRAPH_RUN_TIMEOUT_S exist with sane types."""
    import config

    assert isinstance(config.GRAPH_RECURSION_LIMIT, int)
    assert config.GRAPH_RECURSION_LIMIT >= 1
    assert isinstance(config.GRAPH_RUN_TIMEOUT_S, float)
    assert config.GRAPH_RUN_TIMEOUT_S > 0.0