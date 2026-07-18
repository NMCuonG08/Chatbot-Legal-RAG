"""Unit tests for parallel: config, conversation_id uniqueness, e2e dispatch.

Stubs ``tasks`` / ``agent`` / ``brain`` via ``sys.modules`` so the test never
triggers the heavy import chain (langchain_groq etc.) — only the parallel
orchestration logic is exercised.
"""
import contextvars
import sys
import types

from evaluation.parallel import ParallelConfig, _conversation_id, run_e2e_parallel


def _install_stubs(monkeypatch, run_chat_graph_impl):
    """Inject fake tasks/agent/brain modules into sys.modules."""
    fake_tasks = types.ModuleType("tasks")
    fake_tasks.run_chat_graph = run_chat_graph_impl
    fake_tasks.guardrails_manager = types.SimpleNamespace(initialized=False)

    fake_agent = types.ModuleType("agent")
    fake_agent.agent_tool_calls = contextvars.ContextVar("agent_tool_calls",
                                                         default=None)

    fake_brain = types.ModuleType("brain")
    fake_brain.usage_accumulator = contextvars.ContextVar("usage_accumulator",
                                                          default=None)

    monkeypatch.setitem(sys.modules, "tasks", fake_tasks)
    monkeypatch.setitem(sys.modules, "agent", fake_agent)
    monkeypatch.setitem(sys.modules, "brain", fake_brain)
    return fake_tasks, fake_agent, fake_brain


def _stub_sample(sample_id, question="q"):
    from evaluation.dataset import EvalSample
    return EvalSample(sample_id=sample_id, question=question, gold_context="ctx")


def test_parallel_config_defaults():
    cfg = ParallelConfig()
    assert cfg.max_workers == 8
    assert cfg.judge_concurrency == 4


def test_conversation_id_unique_per_sample():
    a = _conversation_id("run1", "train-0")
    b = _conversation_id("run1", "train-1")
    assert a != b
    assert a.startswith("eval-run1-train-0")


def test_run_e2e_parallel_preserves_order(monkeypatch):
    def _fake_run(history, question, conversation_id=None, user_id=None,
                  run_id=None, role="user"):
        return {"response": "ANS", "route": "legal_rag", "run_id": run_id,
                "sources": [], "tool_calls": [], "verify_score": 0.9,
                "verify_verdict": "supported"}
    _install_stubs(monkeypatch, _fake_run)
    samples = [_stub_sample(f"s{i}") for i in range(6)]
    cfg = ParallelConfig(max_workers=3, judge_concurrency=2)
    results = run_e2e_parallel(samples, cfg, run_id="test-run")
    assert [r.sample_id for r in results] == [f"s{i}" for i in range(6)]
    assert all(r.route == "legal_rag" for r in results)
    assert all(r.error is None for r in results)


def test_run_e2e_parallel_isolates_failure(monkeypatch):
    def _flaky(history, question, conversation_id=None, user_id=None,
               run_id=None, role="user"):
        if "boom" in question:
            raise RuntimeError("boom")
        return {"response": "ok", "route": "legal_rag"}
    _install_stubs(monkeypatch, _flaky)
    samples = [_stub_sample("ok1", "fine"), _stub_sample("bad", "boom")]
    cfg = ParallelConfig(max_workers=2, judge_concurrency=2)
    results = run_e2e_parallel(samples, cfg, run_id="test-run")
    by_id = {r.sample_id: r for r in results}
    assert by_id["ok1"].error is None
    assert by_id["bad"].error is not None
    assert "boom" in by_id["bad"].error