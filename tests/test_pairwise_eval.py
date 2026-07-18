"""Unit tests for pairwise_eval: win count, swap inconsistency, bootstrap CI,
contextvar isolation.

Stubs ``brain`` (LLM contextvars) + ``tasks`` (run_chat_graph) via sys.modules
to avoid the heavy langchain_groq import chain.
"""
import contextvars
import re
import sys
import types

from evaluation.pairwise_eval import (
    AgentConfig,
    PairwiseSampleResult,
    PairwiseSummary,
    pairwise_summary_to_dict,
    run_pairwise_eval,
)
from evaluation.dataset import EvalSample


def _install_stubs(monkeypatch, run_chat_graph_impl):
    fake_brain = types.ModuleType("brain")
    fake_brain.LLM_PROVIDER_CONTEXTVAR = contextvars.ContextVar(
        "LLM_PROVIDER_CONTEXTVAR", default=None)
    fake_brain.LLM_MODEL_CONTEXTVAR = contextvars.ContextVar(
        "LLM_MODEL_CONTEXTVAR", default=None)

    fake_tasks = types.ModuleType("tasks")
    fake_tasks.run_chat_graph = run_chat_graph_impl

    monkeypatch.setitem(sys.modules, "brain", fake_brain)
    monkeypatch.setitem(sys.modules, "tasks", fake_tasks)
    return fake_brain, fake_tasks


def _stub_sample(sample_id, question="q"):
    return EvalSample(sample_id=sample_id, question=question, gold_context="ctx")


# A length-based judge: prefers the longer answer. Parses the two labelled
# answer blocks from the prompt so it is order-invariant (the underlying text
# wins, not the label).
_ANSWER_RE = re.compile(r"Câu trả lời (A|B):\s*(.*?)(?=Câu trả lời |\Z)", re.DOTALL)


def _length_judge(messages):
    user = messages[-1]["content"]
    found = {label: txt.strip() for label, txt in _ANSWER_RE.findall(user)}
    if not found:
        return '{"winner": "tie", "reason": ""}'
    a_len = len(found.get("A", ""))
    b_len = len(found.get("B", ""))
    if a_len == b_len:
        return '{"winner": "tie", "reason": ""}'
    winner = "A" if a_len > b_len else "B"
    return f'{{"winner": "{winner}", "reason": "longer"}}'


def test_pairwise_a_wins_when_a_longer(monkeypatch):
    # variant A (groq) -> long answer; variant B (ollama) -> short answer.
    def _runner_var(history, question):
        from brain import LLM_PROVIDER_CONTEXTVAR
        prov = LLM_PROVIDER_CONTEXTVAR.get()
        return {"response": "B" if prov == "ollama" else "A" * 200}

    _install_stubs(monkeypatch, _runner_var)

    samples = [_stub_sample("s1"), _stub_sample("s2"), _stub_sample("s3")]
    agent_a = AgentConfig(name="A", provider="groq", model="big")
    agent_b = AgentConfig(name="B", provider="ollama", model="small")
    summary = run_pairwise_eval(samples, agent_a, agent_b, _length_judge)
    assert summary.n == 3
    assert summary.a_wins == 3
    assert summary.b_wins == 0
    assert summary.errors == 0
    assert summary.a_win_rate == 1.0
    assert summary.swap_inconsistency_rate == 0.0


def test_pairwise_agent_error_recorded(monkeypatch):
    def _runner(history, question):
        if "boom" in question:
            raise RuntimeError("boom")
        return {"response": "A" * 100}
    _install_stubs(monkeypatch, _runner)
    samples = [_stub_sample("ok", "fine"), _stub_sample("bad", "boom")]
    agent_a = AgentConfig(name="A")
    agent_b = AgentConfig(name="B")
    summary = run_pairwise_eval(samples, agent_a, agent_b, _length_judge)
    assert summary.errors == 1
    # error samples excluded from win-rate denominator
    assert summary.a_wins + summary.b_wins + summary.ties == 1


def test_pairwise_summary_to_dict_roundtrip():
    s = PairwiseSummary(
        n=5, a_wins=3, b_wins=1, ties=1, swap_inconsistency_rate=0.1,
        a_win_rate=0.6, bootstrap_ci_a_win={"lower": 0.2, "upper": 0.9,
                                            "estimate": 0.6},
        p_value_sign_test=0.5, errors=0)
    d = pairwise_summary_to_dict(s)
    assert d["a_wins"] == 3
    assert d["bootstrap_ci_a_win"]["lower"] == 0.2
    assert d["p_value_sign_test"] == 0.5


def test_pairwise_contextvar_isolation(monkeypatch):
    """Pinned provider/model must be set during the call and reset after."""
    seen = []

    def _runner(history, question):
        from brain import LLM_PROVIDER_CONTEXTVAR, LLM_MODEL_CONTEXTVAR
        seen.append((LLM_PROVIDER_CONTEXTVAR.get(),
                     LLM_MODEL_CONTEXTVAR.get()))
        return {"response": "x" * 50}

    fake_brain, _ = _install_stubs(monkeypatch, _runner)
    samples = [_stub_sample("s1")]
    agent_a = AgentConfig(name="A", provider="groq", model="m-a")
    run_pairwise_eval(samples, agent_a, AgentConfig(name="B"), _length_judge)
    # A run (first) saw the pinned values; B run (provider=None) saw defaults.
    assert seen[0] == ("groq", "m-a")
    assert seen[1] == (None, None)
    # after both runs, contextvars are reset to their defaults
    assert fake_brain.LLM_PROVIDER_CONTEXTVAR.get() is None
    assert fake_brain.LLM_MODEL_CONTEXTVAR.get() is None