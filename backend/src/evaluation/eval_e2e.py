"""End-to-end evaluation: route through the LangGraph chat graph used in production.

Unlike ``eval_generation`` which calls the RAG path directly, this runs the
real ``run_chat_graph`` so the routing decision (legal_rag / agent_tools /
web_search / general_chat) is included in the measurement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class E2ERunResult:
    sample_id: str
    question: str
    answer: str
    route: str
    expected_route: str
    latency_ms: float
    token_usage: List[dict] = field(default_factory=field)
    tool_calls: List[dict] = field(default_factory=field)
    error: Optional[str] = None


def get_expected_route(question: str) -> str:
    """Determine expected route based on legal keywords (calculation/checks vs general legal RAG)."""
    question_lower = question.lower()
    calc_keywords = [
        "tính",
        "kiểm tra",
        "hợp lệ",
        "đủ tuổi",
        "chia",
        "phạt",
        "thời hiệu",
        "thừa kế",
    ]
    if any(kw in question_lower for kw in calc_keywords):
        return "agent_tools"
    return "legal_rag"


def run_e2e_eval(samples, history: Optional[List[dict]] = None) -> List[E2ERunResult]:
    """Run each question through the production chat graph."""
    from tasks import run_chat_graph
    from brain import usage_accumulator
    from agent import agent_tool_calls

    history = history or []
    results: List[E2ERunResult] = []
    for i, sample in enumerate(samples, 1):
        t0 = time.perf_counter()
        actual_route = ""
        expected = get_expected_route(sample.question)
        
        token_list = []
        tool_calls_list = []
        token_ctx = usage_accumulator.set(token_list)
        tool_ctx = agent_tool_calls.set(tool_calls_list)
        
        try:
            graph_res = run_chat_graph(history, sample.question)
            answer = graph_res.get("response", "")
            actual_route = graph_res.get("route", "")
            err = None
        except Exception as exc:
            answer = ""
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("E2E failed for %s: %s", sample.sample_id, exc)
        finally:
            usage_accumulator.reset(token_ctx)
            agent_tool_calls.reset(tool_ctx)
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append(
            E2ERunResult(
                sample_id=sample.sample_id,
                question=sample.question,
                answer=answer,
                route=actual_route,
                expected_route=expected,
                latency_ms=elapsed_ms,
                token_usage=token_list,
                tool_calls=tool_calls_list,
                error=err,
            )
        )
        if i % 5 == 0:
            logger.info("E2E eval: processed %d/%d", i, len(samples))
    return results


def summarize_e2e_results(results: List[E2ERunResult]) -> Dict[str, float]:
    from .metrics_generation import estimate_cost

    n = len(results)
    if n == 0:
        return {"n_queries": 0}
    latencies = [r.latency_ms for r in results]
    success = [r for r in results if not r.error and r.answer]

    # Calculate routing accuracy and aggregate tokens, costs, tool calls
    correct_routes = 0
    route_counts = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    all_tokens = []

    tool_calls_count = 0
    tool_success = 0
    tool_distribution = {}

    for r in results:
        # Route
        route = r.route or "unknown"
        route_counts[route] = route_counts.get(route, 0) + 1

        legal_routes = {"legal_rag", "agent_tools"}
        if r.route == r.expected_route:
            correct_routes += 1
        elif r.route in legal_routes and r.expected_route in legal_routes:
            correct_routes += 1

        # Tokens
        for item in r.token_usage:
            total_prompt_tokens += item.get("prompt_tokens", 0)
            total_completion_tokens += item.get("completion_tokens", 0)
            all_tokens.append(item)

        # Tools
        for call in r.tool_calls:
            tool_calls_count += 1
            name = call.get("tool_name", "unknown")
            tool_distribution[name] = tool_distribution.get(name, 0) + 1
            if call.get("status") == "success":
                tool_success += 1

    total_cost = estimate_cost(all_tokens)
    tool_success_rate = tool_success / tool_calls_count if tool_calls_count > 0 else 1.0

    sorted_lat = sorted(latencies)
    p95_idx = max(0, int(round(0.95 * (n - 1))))
    return {
        "n_queries": n,
        "success_rate": len(success) / n,
        "routing_accuracy": correct_routes / n,
        "latency_ms_mean": sum(latencies) / n,
        "latency_ms_p95": sorted_lat[p95_idx],
        "routing_distribution": route_counts,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "estimated_cost_usd": total_cost,
        "tool_calls_count": tool_calls_count,
        "tool_calls_success_rate": tool_success_rate,
        "tool_calls_distribution": tool_distribution,
    }
