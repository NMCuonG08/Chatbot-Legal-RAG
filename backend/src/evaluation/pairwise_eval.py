"""Pairwise A/B eval: run two agent configs over the same samples, judge which
answer is better with swap-augmentation, report win rates + swap-inconsistency.

Each ``AgentConfig`` pins a provider+model via ``brain`` contextvars for the
duration of that agent's ``run_chat_graph`` call, so two variants run isolated
in the same process without env mutation.

Public surface:
- ``AgentConfig``, ``PairwiseSampleResult``, ``PairwiseSummary`` — frozen.
- ``run_pairwise_eval`` — run A + B per sample, swap-augment judge, aggregate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from evaluation.judge_panel import swap_augment_pairwise
from evaluation.stats import bootstrap_ci

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    provider: Optional[str] = None  # None -> env default
    model: Optional[str] = None     # None -> env default


@dataclass(frozen=True)
class PairwiseSampleResult:
    sample_id: str
    question: str
    answer_a: str
    answer_b: str
    # +1 A wins, -1 B wins, 0 tie (post swap-augmentation)
    preference: float
    swap_inconsistent: bool  # the two swap runs disagreed -> forced tie
    error: Optional[str] = None


@dataclass(frozen=True)
class PairwiseSummary:
    n: int
    a_wins: int
    b_wins: int
    ties: int
    swap_inconsistency_rate: float
    a_win_rate: float  # a_wins / n_valid
    bootstrap_ci_a_win: Dict[str, float]
    p_value_sign_test: float  # sign test on a_wins vs b_wins
    errors: int


def _run_agent(sample, cfg: AgentConfig, run_chat_graph) -> Dict:
    """Run one agent config under its pinned provider/model contextvars."""
    from brain import LLM_MODEL_CONTEXTVAR, LLM_PROVIDER_CONTEXTVAR
    p_tok = LLM_PROVIDER_CONTEXTVAR.set(cfg.provider) if cfg.provider else None
    m_tok = LLM_MODEL_CONTEXTVAR.set(cfg.model) if cfg.model else None
    try:
        return run_chat_graph([], sample.question)
    finally:
        if p_tok is not None:
            LLM_PROVIDER_CONTEXTVAR.reset(p_tok)
        if m_tok is not None:
            LLM_MODEL_CONTEXTVAR.reset(m_tok)


def run_pairwise_eval(
    samples,
    agent_a: AgentConfig,
    agent_b: AgentConfig,
    judge_fn: Callable,
    *,
    top_k: int = 0,
    run_chat_graph: Optional[Callable] = None,
) -> PairwiseSummary:
    """Run A + B over each sample, swap-augment the pairwise judge, aggregate."""
    if run_chat_graph is None:
        from tasks import run_chat_graph as _rg  # lazy; tests inject a stub
        run_chat_graph = _rg

    results: List[PairwiseSampleResult] = []
    for s in samples:
        try:
            ra = _run_agent(s, agent_a, run_chat_graph)
            rb = _run_agent(s, agent_b, run_chat_graph)
            ans_a = ra.get("response", "")
            ans_b = rb.get("response", "")
        except Exception as exc:
            results.append(PairwiseSampleResult(
                sample_id=s.sample_id, question=s.question,
                answer_a="", answer_b="", preference=0.0,
                swap_inconsistent=False, error=f"{type(exc).__name__}: {exc}"))
            continue
        verdict = swap_augment_pairwise(s.question, ans_a, ans_b, judge_fn)
        inconsistent = (verdict.votes[0].score != verdict.votes[1].score
                        if len(verdict.votes) >= 2 else False)
        results.append(PairwiseSampleResult(
            sample_id=s.sample_id, question=s.question,
            answer_a=ans_a, answer_b=ans_b,
            preference=verdict.score, swap_inconsistent=inconsistent))

    return _summarize_pairwise(results)


def _summarize_pairwise(results: Sequence[PairwiseSampleResult]) -> PairwiseSummary:
    n = len(results)
    valid = [r for r in results if r.error is None]
    a_wins = sum(1 for r in valid if r.preference > 0)
    b_wins = sum(1 for r in valid if r.preference < 0)
    ties = sum(1 for r in valid if r.preference == 0)
    errors = n - len(valid)
    incons = sum(1 for r in valid if r.swap_inconsistent)
    win_indicators = [1.0 if r.preference > 0 else 0.0 for r in valid]
    ci = bootstrap_ci(win_indicators, n_boot=1000, seed=42)
    p_value = _sign_test_pvalue(a_wins, b_wins)
    return PairwiseSummary(
        n=n, a_wins=a_wins, b_wins=b_wins, ties=ties,
        swap_inconsistency_rate=(incons / len(valid)) if valid else 0.0,
        a_win_rate=(a_wins / len(valid)) if valid else 0.0,
        bootstrap_ci_a_win={"lower": ci.lower, "upper": ci.upper,
                            "estimate": ci.estimate},
        p_value_sign_test=p_value, errors=errors,
    )


def _sign_test_pvalue(a_wins: int, b_wins: int) -> float:
    from scipy.stats import binomtest
    non_ties = a_wins + b_wins
    if non_ties == 0:
        return 1.0
    return float(binomtest(a_wins, non_ties, 0.5, alternative="two-sided").pvalue)


def pairwise_summary_to_dict(summary: PairwiseSummary) -> Dict:
    from dataclasses import asdict
    return asdict(summary)


__all__ = [
    "AgentConfig",
    "PairwiseSampleResult",
    "PairwiseSummary",
    "run_pairwise_eval",
    "pairwise_summary_to_dict",
]