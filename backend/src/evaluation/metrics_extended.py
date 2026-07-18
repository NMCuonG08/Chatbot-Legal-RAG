"""Extended metrics: tool-call accuracy, noise sensitivity, context utilization,
hallucination rate, latency p99.

These complement the core retrieval/generation metrics with agentic + reliability
signals. Reuses ``evaluate_faithfulness`` where a judge is needed; the rest are
deterministic.

Public surface:
- ``ToolCallScore``, ``NoiseSensitivityResult`` frozen.
- ``tool_call_accuracy``, ``noise_sensitivity``, ``context_utilization``,
  ``hallucination_rate``, ``latency_p99``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallScore:
    right_tool: float
    right_args: float
    score: float  # 0.5 * right_tool + 0.5 * right_args


@dataclass(frozen=True)
class NoiseSensitivityResult:
    base_faithfulness: float
    noisy_faithfulness: float
    delta: float  # base - noisy; positive = degraded under noise
    score: float  # 1 - |delta| (higher = robust)


def tool_call_accuracy(
    tool_calls: Sequence[dict],
    expected_tools: Sequence[str],
    expected_args: Optional[Dict[str, dict]] = None,
) -> ToolCallScore:
    """Score tool-call correctness.

    ``expected_args`` maps tool name -> expected arg dict (subset match: every
    key in expected must equal the called arg).
    """
    called = []
    for tc in tool_calls or []:
        name = (tc.get("name") or tc.get("tool_name")
                if isinstance(tc, dict) else str(tc))
        args = (tc.get("args") or tc.get("arguments") or {}
                if isinstance(tc, dict) else {})
        called.append((name, args))
    called_names = [c[0] for c in called if c[0]]
    expected = list(expected_tools or [])

    if not expected:
        right_tool = 1.0
    else:
        hits = sum(1 for t in expected if t in called_names)
        right_tool = hits / len(expected)

    if not expected_args:
        right_args = 1.0
    else:
        checked = 0
        matched = 0
        called_by_name = {n: a for n, a in called}
        for t, exp_args in expected_args.items():
            checked += 1
            if t in called_by_name:
                actual = called_by_name[t] or {}
                if all(actual.get(k) == v for k, v in exp_args.items()):
                    matched += 1
        right_args = matched / checked if checked else 1.0

    score = 0.5 * right_tool + 0.5 * right_args
    return ToolCallScore(right_tool=right_tool, right_args=right_args,
                         score=score)


def noise_sensitivity(
    question: str,
    answer: str,
    base_contexts: Sequence[str],
    noisy_contexts: Sequence[str],
    judge_fn,
) -> NoiseSensitivityResult:
    """Ragas-inspired: faithfulness under clean vs noisy (dropped/swapped) docs.

    ``delta`` = base_faithfulness - noisy_faithfulness. A robust answer has
    delta near 0 (faithfulness unchanged when context is perturbed).
    """
    from evaluation.metrics_generation import evaluate_faithfulness
    base = evaluate_faithfulness(question, answer, list(base_contexts),
                                 judge_fn=judge_fn).score
    noisy = evaluate_faithfulness(question, answer, list(noisy_contexts),
                                  judge_fn=judge_fn).score
    delta = base - noisy
    return NoiseSensitivityResult(
        base_faithfulness=base, noisy_faithfulness=noisy,
        delta=delta, score=1.0 - abs(delta),
    )


def context_utilization(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of retrieved contexts whose content is reflected in the answer.

    A context counts as 'utilized' if any 6+ word run from it appears in the
    answer (lowercased). Coarse but deterministic; avoids LLM cost.
    """
    if not contexts:
        return 0.0
    ans_low = (answer or "").lower()
    if not ans_low.strip():
        return 0.0
    used = 0
    for ctx in contexts:
        ctx_low = (ctx or "").lower()
        if not ctx_low.strip():
            continue
        words = ctx_low.split()
        hit = False
        for i in range(0, max(1, len(words) - 5)):
            window = " ".join(words[i:i + 6])
            if window and window in ans_low:
                hit = True
                break
        if hit:
            used += 1
    return used / len(contexts)


def hallucination_rate(
    faithfulness_scores: Sequence[float],
    threshold: float = 0.5,
) -> float:
    """Density of unsupported answers: fraction with faithfulness < threshold."""
    if not faithfulness_scores:
        return 0.0
    bad = sum(1 for f in faithfulness_scores if f < threshold)
    return bad / len(faithfulness_scores)


def latency_p99(latencies_ms: Sequence[float]) -> float:
    """P99 latency via nearest-rank interpolation."""
    if not latencies_ms:
        return 0.0
    s = sorted(latencies_ms)
    rank = max(0, math.ceil(0.99 * len(s)) - 1)
    return float(s[rank])


__all__ = [
    "ToolCallScore",
    "NoiseSensitivityResult",
    "tool_call_accuracy",
    "noise_sensitivity",
    "context_utilization",
    "hallucination_rate",
    "latency_p99",
]