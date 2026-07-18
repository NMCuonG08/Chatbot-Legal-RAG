"""Parallel eval execution around the synchronous chat graph.

``tasks.run_chat_graph`` is synchronous and uses ``asyncio.run`` internally in
some nodes, so we parallelize with a ``ThreadPoolExecutor``: each worker thread
gets a fresh event loop (no shared running loop), and per-sample
``conversation_id`` is uniquified so the LangGraph checkpointer never collides
across concurrent samples.

A ``threading.Semaphore`` caps concurrent judge LLM calls (Groq rate limits).

Public surface:
- ``ParallelConfig`` — frozen knobs.
- ``run_e2e_parallel`` — concurrent ``run_chat_graph`` for E2E eval.
- ``run_generation_parallel`` — concurrent RAG-path generation eval.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from evaluation.eval_e2e import E2ERunResult, _resolve_expected_route

logger = logging.getLogger(__name__)

# Module-level judge semaphore; resized in run_*_parallel from the config.
_JUDGE_SEMAPHORE: Optional[threading.Semaphore] = None


def set_judge_concurrency(n: int) -> None:
    """Set the global judge-call semaphore capacity for this process."""
    global _JUDGE_SEMAPHORE
    _JUDGE_SEMAPHORE = threading.Semaphore(max(1, n))


def get_judge_semaphore() -> threading.Semaphore:
    global _JUDGE_SEMAPHORE
    if _JUDGE_SEMAPHORE is None:
        _JUDGE_SEMAPHORE = threading.Semaphore(4)
    return _JUDGE_SEMAPHORE


@dataclass(frozen=True)
class ParallelConfig:
    max_workers: int = 8
    judge_concurrency: int = 4
    per_call_timeout_s: float = 120.0


def _conversation_id(run_id: str, sample_id: str) -> str:
    """Unique thread_id per sample so the checkpointer never collides."""
    safe = sample_id.replace("/", "-").replace(" ", "_")
    return f"eval-{run_id}-{safe}"


def _run_one_e2e(sample, run_id: str, history_fn: Optional[Callable]) -> E2ERunResult:
    from tasks import run_chat_graph
    from brain import usage_accumulator
    from agent import agent_tool_calls

    history = history_fn(sample) if history_fn else []
    expected = _resolve_expected_route(sample)
    token_list: List[dict] = []
    tool_calls_list: List[dict] = []
    token_ctx = usage_accumulator.set(token_list)
    tool_ctx = agent_tool_calls.set(tool_calls_list)
    t0 = time.perf_counter()
    err: Optional[str] = None
    answer = ""
    route = ""
    try:
        res = run_chat_graph(
            history, sample.question,
            conversation_id=_conversation_id(run_id, sample.sample_id),
        )
        answer = res.get("response", "")
        route = res.get("route", "")
    except Exception as exc:  # one sample failing must not abort the batch
        err = f"{type(exc).__name__}: {exc}"
        logger.warning("parallel e2e failed for %s: %s", sample.sample_id, exc)
    finally:
        usage_accumulator.reset(token_ctx)
        agent_tool_calls.reset(tool_ctx)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return E2ERunResult(
        sample_id=sample.sample_id, question=sample.question,
        answer=answer, route=route, expected_route=expected,
        latency_ms=elapsed_ms, token_usage=token_list,
        tool_calls=tool_calls_list, error=err,
    )


def run_e2e_parallel(
    samples,
    cfg: ParallelConfig,
    run_id: str,
    history_fn: Optional[Callable] = None,
    progress_every: int = 10,
) -> List[E2ERunResult]:
    """Run ``run_chat_graph`` over samples concurrently, preserving input order."""
    set_judge_concurrency(cfg.judge_concurrency)
    results: List[Optional[E2ERunResult]] = [None] * len(samples)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        futs = {
            ex.submit(_run_one_e2e, s, run_id, history_fn): i
            for i, s in enumerate(samples)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                logger.warning("e2e worker crashed for %s: %s",
                               samples[idx].sample_id, exc)
                results[idx] = E2ERunResult(
                    sample_id=samples[idx].sample_id,
                    question=samples[idx].question, answer="", route="",
                    expected_route=_resolve_expected_route(samples[idx]),
                    latency_ms=0.0, error=f"worker_crash: {exc}",
                )
            done += 1
            if done % progress_every == 0:
                logger.info("parallel e2e: %d/%d", done, len(samples))
    return [r for r in results if r is not None]


def run_generation_parallel(
    samples,
    cfg: ParallelConfig,
    run_id: str,
    judge_fn,
    top_k: int = 5,
    progress_every: int = 10,
) -> list:
    """Concurrent generation eval (RAG path + judge) preserving input order."""
    from evaluation.eval_generation import run_generation_eval

    set_judge_concurrency(cfg.judge_concurrency)
    results: List[Optional[object]] = [None] * len(samples)
    chunk_n = max(1, cfg.max_workers)
    chunks = [samples[i::chunk_n] for i in range(chunk_n)]

    with ThreadPoolExecutor(max_workers=chunk_n) as ex:
        futs = {ex.submit(run_generation_eval, chunk, top_k, judge_fn): chunk
                for chunk in chunks if chunk}
        for fut in as_completed(futs):
            try:
                chunk_results = fut.result()
            except Exception as exc:
                logger.warning("generation chunk crashed: %s", exc)
                continue
            for r in chunk_results:
                sid = getattr(r, "sample_id", None)
                for i, s in enumerate(samples):
                    if s.sample_id == sid:
                        results[i] = r
                        break
    return [r for r in results if r is not None]


def _run_one_scenario(scenario, persona, run_id, max_turns, judge_fn=None):
    """Run one multi-turn scenario: simulate conversation + score it."""
    from tasks import run_chat_graph
    from evaluation.sim_user import simulate_conversation
    from evaluation.scenarios import score_scenario

    conv_id = _conversation_id(run_id, scenario.scenario_id)

    def agent_runner(history, user_msg):
        return run_chat_graph(history, user_msg, conversation_id=conv_id)

    log = simulate_conversation(scenario, agent_runner, persona,
                                max_turns=max_turns)
    return score_scenario(log, scenario, judge_fn=judge_fn)


def run_scenario_parallel(
    scenarios, persona, cfg: ParallelConfig, run_id: str,
    *, max_turns: int = 8, judge_fn=None, progress_every: int = 5,
) -> list:
    """Run scenarios concurrently, preserving input order."""
    set_judge_concurrency(cfg.judge_concurrency)
    from evaluation.scenarios import ScenarioScore
    results: List[Optional[ScenarioScore]] = [None] * len(scenarios)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        futs = {
            ex.submit(_run_one_scenario, sc, persona, run_id, max_turns,
                      judge_fn): i
            for i, sc in enumerate(scenarios)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                logger.warning("scenario worker crashed for %s: %s",
                               scenarios[idx].scenario_id, exc)
                results[idx] = ScenarioScore(
                    scenario_id=scenarios[idx].scenario_id,
                    r_action=0.0, r_output=0.0, r_composite=0.0,
                    success=False, reached_goal=False, n_turns=0,
                    notes=f"worker_crash: {exc}",
                )
            done += 1
            if done % progress_every == 0:
                logger.info("parallel scenario: %d/%d", done, len(scenarios))
    return [r for r in results if r is not None]


__all__ = [
    "ParallelConfig",
    "run_e2e_parallel",
    "run_generation_parallel",
    "run_scenario_parallel",
    "set_judge_concurrency",
    "get_judge_semaphore",
]