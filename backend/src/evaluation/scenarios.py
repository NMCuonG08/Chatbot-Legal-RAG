"""Task scenarios + tau-bench-style scoring for multi-turn agent eval.

A ``Scenario`` is a deterministic task (goal + success criteria + expected
route/tools/keywords). ``score_scenario`` combines a deterministic action
score (did the agent take the right route/tools?) with an output score
(keyword match + optional judge) into a composite; success when composite >= 0.7.

Public surface:
- ``Scenario``, ``ScenarioScore`` — frozen.
- ``generate_scenarios_from_dataset``, ``score_scenario``.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from evaluation.dataset import EvalSample, load_eval_dataset

logger = logging.getLogger(__name__)

_SUCCESS_THRESHOLD = 0.7
_ACTION_WEIGHT = 0.5
_OUTPUT_WEIGHT = 0.5


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    goal: str
    expected_route: Optional[str] = None
    expected_tools: tuple = ()
    expected_answer_keywords: tuple = ()
    distractors: tuple = ()
    initial_state: str = ""
    max_turns: int = 8
    expected_block: bool = False


@dataclass(frozen=True)
class ScenarioScore:
    scenario_id: str
    r_action: float  # deterministic: route + tools
    r_output: float  # keyword match (+ optional judge)
    r_composite: float
    success: bool
    reached_goal: bool
    n_turns: int
    notes: str = ""


def generate_scenarios_from_dataset(
    path=None, n: int = 30, seed: int = 42,
) -> List[Scenario]:
    """Sample ``n`` scenarios from the eval dataset deterministically.

    Derives expected_route/expected_tool/expected_answer_keywords from the
    annotated EvalSample fields when present; otherwise leaves them None and
    uses the question text as the goal.
    """
    samples = load_eval_dataset(path=path, n_samples=n, seed=seed) if path else \
        load_eval_dataset(n_samples=n, seed=seed)
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    for s in samples:
        kws = tuple(w.strip() for w in (s.expected_answer or "").split(",")
                    if w.strip()) if s.expected_answer else ()
        distractor = ""
        if len(samples) > 1:
            other = rng.choice([x for x in samples if x.sample_id != s.sample_id])
            distractor = other.question
        scenarios.append(Scenario(
            scenario_id=f"scn-{s.sample_id}",
            goal=s.question,
            expected_route=s.expected_route,
            expected_tools=(s.expected_tool,) if s.expected_tool else (),
            expected_answer_keywords=kws,
            distractors=(distractor,) if distractor else (),
            initial_state=s.question,
            expected_block=s.expected_block,
        ))
    return scenarios


def _route_score(log, scenario: Scenario) -> float:
    if not scenario.expected_route:
        return 1.0  # no expectation -> neutral full
    routes = [t.route for t in log.turns if t.route]
    if not routes:
        return 0.0
    hits = sum(1 for r in routes if r == scenario.expected_route)
    return hits / len(routes)


def _tool_score(log, scenario: Scenario) -> float:
    if not scenario.expected_tools:
        return 1.0
    called = set()
    for t in log.turns:
        for tc in t.tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else str(tc)
            if name:
                called.add(name)
    expected = set(scenario.expected_tools)
    if not expected:
        return 1.0
    return len(expected & called) / len(expected)


def _action_score(log, scenario: Scenario) -> float:
    components = []
    if scenario.expected_route:
        components.append(_route_score(log, scenario))
    if scenario.expected_tools:
        components.append(_tool_score(log, scenario))
    if not components:
        return 1.0  # no action expectation -> neutral full
    return sum(components) / len(components)


def _keyword_score(log, scenario: Scenario) -> float:
    if not scenario.expected_answer_keywords:
        return 1.0
    corpus = " ".join(t.agent_reply for t in log.turns).lower()
    if not corpus.strip():
        return 0.0
    hits = sum(1 for kw in scenario.expected_answer_keywords
               if kw.lower() in corpus)
    return hits / len(scenario.expected_answer_keywords)


def score_scenario(
    log,
    scenario: Scenario,
    *,
    judge_fn: Optional[Callable] = None,
) -> ScenarioScore:
    """Score a conversation log against a scenario.

    ``r_action`` is deterministic (route + tool correctness).
    ``r_output`` is keyword match; if ``judge_fn`` is supplied, it is blended
    50/50 with the keyword score (judge grades the final agent reply vs goal).
    ``r_composite`` = 0.5*r_action + 0.5*r_output. Success when >= 0.7.
    """
    r_action = _action_score(log, scenario)
    r_kw = _keyword_score(log, scenario)
    if judge_fn is not None and log.turns:
        final_reply = log.turns[-1].agent_reply
        try:
            raw = judge_fn([
                {"role": "system", "content": "Bạn là giám khảo. Trả lời JSON."},
                {"role": "user", "content": (
                    f"Mục tiêu: {scenario.goal}\nCâu trả lời cuối: {final_reply}\n"
                    f"Trả về DUY NHẤT JSON: {{\"score\": <0.0..1.0>}}")},
            ])
            from evaluation.metrics_generation import _parse_judge_json
            parsed = _parse_judge_json(raw) or {}
            j = float(parsed.get("score", r_kw))
            j = max(0.0, min(1.0, j))
            r_output = 0.5 * r_kw + 0.5 * j
        except Exception as exc:
            logger.warning("scenario judge failed: %s", exc)
            r_output = r_kw
    else:
        r_output = r_kw

    r_composite = _ACTION_WEIGHT * r_action + _OUTPUT_WEIGHT * r_output
    if scenario.expected_block:
        r_composite = r_composite if not log.reached_goal else 0.0
    success = r_composite >= _SUCCESS_THRESHOLD and log.reached_goal
    if scenario.expected_block:
        success = r_composite >= _SUCCESS_THRESHOLD
    return ScenarioScore(
        scenario_id=scenario.scenario_id,
        r_action=r_action, r_output=r_output, r_composite=r_composite,
        success=success, reached_goal=log.reached_goal,
        n_turns=len(log.turns),
        notes="expected_block" if scenario.expected_block else "",
    )


def summarize_scenario_scores(scores: Sequence[ScenarioScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n_scenarios": 0, "success_rate": 0.0,
                "mean_r_action": 0.0, "mean_r_output": 0.0,
                "mean_r_composite": 0.0, "reached_goal_rate": 0.0,
                "mean_turns": 0.0}
    return {
        "n_scenarios": n,
        "success_rate": sum(1 for s in scores if s.success) / n,
        "mean_r_action": sum(s.r_action for s in scores) / n,
        "mean_r_output": sum(s.r_output for s in scores) / n,
        "mean_r_composite": sum(s.r_composite for s in scores) / n,
        "reached_goal_rate": sum(1 for s in scores if s.reached_goal) / n,
        "mean_turns": sum(s.n_turns for s in scores) / n,
    }


__all__ = [
    "Scenario",
    "ScenarioScore",
    "generate_scenarios_from_dataset",
    "score_scenario",
    "summarize_scenario_scores",
]