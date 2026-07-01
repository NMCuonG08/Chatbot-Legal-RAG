"""CLI entry point for the RAG evaluation suite.

Examples
--------
Retrieval-only ablation on 50 questions::

    python -m evaluation.run_eval --mode retrieval --n 50

Generation eval (slower, calls judge LLM 3x per sample)::

    python -m evaluation.run_eval --mode generation --n 30

Full report (retrieval + generation + e2e)::

    python -m evaluation.run_eval --mode all --n 30 --output reports/

The script writes ``report.json`` and ``report.md`` into the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend/src is on sys.path so relative-style imports (`from search ...`)
# inside the eval submodules resolve when this script is invoked directly.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(SRC_DIR.parent / ".env")

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    # Fail loudly on unknown types instead of silently serializing a Python
    # repr into the JSON report (which masks bugs in the report structure).
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _format_table(headers, rows) -> str:
    """Render a markdown table given headers and a list of row tuples."""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _retrieval_section(retrieval_results) -> str:
    if not retrieval_results:
        return ""
    headers = ["config", "n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg@10", "lat_mean_ms", "lat_p95_ms"]
    rows = []
    for cfg, run in retrieval_results.items():
        m = run.metrics
        rows.append([
            cfg,
            m.get("n_queries", 0),
            f"{m.get('hit@1', 0):.3f}",
            f"{m.get('hit@3', 0):.3f}",
            f"{m.get('hit@5', 0):.3f}",
            f"{m.get('hit@10', 0):.3f}",
            f"{m.get('mrr', 0):.3f}",
            f"{m.get('ndcg@10', 0):.3f}",
            f"{m.get('latency_ms_mean', 0):.1f}",
            f"{m.get('latency_ms_p95', 0):.1f}",
        ])
    return "## Retrieval ablation\n\n" + _format_table(headers, rows) + "\n"


def _generation_section(gen_summary) -> str:
    if not gen_summary:
        return ""
    headers = ["metric", "value"]
    rows = [
        ("n_queries", gen_summary.get("n_queries", 0)),
        ("faithfulness_mean", f"{gen_summary.get('faithfulness_mean', 0):.3f}"),
        ("answer_relevance_mean", f"{gen_summary.get('answer_relevance_mean', 0):.3f}"),
        ("context_precision_mean", f"{gen_summary.get('context_precision_mean', 0):.3f}"),
        ("latency_ms_mean", f"{gen_summary.get('latency_ms_mean', 0):.1f}"),
        ("latency_ms_p95", f"{gen_summary.get('latency_ms_p95', 0):.1f}"),
    ]
    return "## Generation quality (LLM-as-judge)\n\n" + _format_table(headers, rows) + "\n"


def _e2e_section(e2e_summary) -> str:
    if not e2e_summary:
        return ""
    headers = ["metric", "value"]
    rows = [
        ("n_queries", e2e_summary.get("n_queries", 0)),
        ("success_rate", f"{e2e_summary.get('success_rate', 0):.3f}"),
        ("latency_ms_mean", f"{e2e_summary.get('latency_ms_mean', 0):.1f}"),
        ("latency_ms_p95", f"{e2e_summary.get('latency_ms_p95', 0):.1f}"),
    ]
    return "## End-to-end (chat graph)\n\n" + _format_table(headers, rows) + "\n"


def _operational_section(payload) -> str:
    # Gather token & cost info from generation_summary and/or e2e_summary
    gen_sum = payload.get("generation_summary", {})
    e2e_sum = payload.get("e2e_summary", {})

    rows = []
    if gen_sum and "total_tokens" in gen_sum:
        rows.append([
            "Generation Stage",
            gen_sum.get("total_prompt_tokens", 0),
            gen_sum.get("total_completion_tokens", 0),
            gen_sum.get("total_tokens", 0),
            f"${gen_sum.get('estimated_cost_usd', 0.0):.5f}"
        ])
    if e2e_sum and "total_tokens" in e2e_sum:
        rows.append([
            "E2E Chat Graph Stage",
            e2e_sum.get("total_prompt_tokens", 0),
            e2e_sum.get("total_completion_tokens", 0),
            e2e_sum.get("total_tokens", 0),
            f"${e2e_sum.get('estimated_cost_usd', 0.0):.5f}"
        ])

    if not rows:
        return ""

    headers = ["Stage / Run Mode", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Estimated Cost (USD)"]
    return "## Operational Metrics (Tokens & Cost)\n\n" + _format_table(headers, rows) + "\n"


def _failure_section(failure_summary) -> str:
    if not failure_summary:
        return ""
    headers = ["Failure Mode", "Percentage"]
    rows = []
    for cat, pct in failure_summary.items():
        rows.append([cat.replace("_", " ").title(), f"{pct * 100:.1f}%"])
    return "## Failure Analysis Breakdown\n\n" + _format_table(headers, rows) + "\n"


def _agentic_section(e2e_sum) -> str:
    if not e2e_sum:
        return ""

    out = []

    # 1. Routing metrics
    headers_route = ["Routing Metric", "Value"]
    rows_route = [
        ("Routing Accuracy (Expected vs Actual)", f"{e2e_sum.get('routing_accuracy', 0.0) * 100:.1f}%"),
    ]
    # routing distribution
    dist = e2e_sum.get("routing_distribution", {})
    for route, count in dist.items():
        rows_route.append((f"Route Chosen: {route}", count))

    out.append("### Routing Decisions")
    out.append(_format_table(headers_route, rows_route))
    out.append("")

    # 2. Tool use metrics
    if "tool_calls_count" in e2e_sum and e2e_sum.get("tool_calls_count", 0) > 0:
        headers_tool = ["Agentic Tool Metric", "Value"]
        rows_tool = [
            ("Total Tool Calls", e2e_sum.get("tool_calls_count", 0)),
            ("Tool Calls Success Rate", f"{e2e_sum.get('tool_calls_success_rate', 0.0) * 100:.1f}%"),
        ]
        tool_dist = e2e_sum.get("tool_calls_distribution", {})
        for tool, count in tool_dist.items():
            rows_tool.append((f"Tool Used: {tool}", count))

        out.append("### ReAct Agent Tool Usage")
        out.append(_format_table(headers_tool, rows_tool))
        out.append("")

    return "## Agentic & Routing Metrics\n\n" + "\n".join(out)


def _build_markdown_report(payload: dict) -> str:
    parts = [
        f"# RAG Evaluation Report",
        f"",
        f"- generated_at: {payload['generated_at']}",
        f"- mode: {payload['mode']}",
        f"- n_samples: {payload['n_samples']}",
        f"- seed: {payload['seed']}",
        f"- top_k: {payload['top_k']}",
        f"",
    ]
    parts.append(_retrieval_section(payload.get("retrieval_results")))
    parts.append(_generation_section(payload.get("generation_summary")))
    parts.append(_e2e_section(payload.get("e2e_summary")))
    parts.append(_operational_section(payload))
    parts.append(_failure_section(payload.get("failure_summary")))
    parts.append(_agentic_section(payload.get("e2e_summary")))
    return "\n".join(p for p in parts if p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "generation", "e2e", "all"],
        default="retrieval",
        help="Which evaluation stage(s) to run",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Subset of retrieval configs (vector hybrid hybrid_rerank multi_query full)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_reports",
        help="Directory to write report.json and report.md",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to train.jsonl (defaults to repo data/train.jsonl)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from evaluation.dataset import DEFAULT_DATA_FILE, load_eval_dataset

    data_path = Path(args.data_file) if args.data_file else DEFAULT_DATA_FILE
    samples = load_eval_dataset(path=data_path, n_samples=args.n, seed=args.seed)
    logger.info("Evaluating with %d samples (mode=%s)", len(samples), args.mode)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "n_samples": len(samples),
        "seed": args.seed,
        "top_k": args.top_k,
    }

    retrieval_results = None
    gen_results = None
    e2e_results = None

    if args.mode in ("retrieval", "all"):
        from evaluation.eval_retrieval import run_retrieval_eval

        retrieval_results = run_retrieval_eval(
            samples,
            configs=args.configs,
            top_k=args.top_k,
        )
        payload["retrieval_results"] = {
            cfg: {"metrics": run.metrics, "ranks": run.ranks}
            for cfg, run in retrieval_results.items()
        }
        # Keep dataclass form for markdown rendering only.
        payload["_retrieval_runs"] = retrieval_results

    if args.mode in ("generation", "all"):
        from evaluation.eval_generation import (
            run_generation_eval,
            summarize_generation_results,
        )

        gen_results = run_generation_eval(samples, top_k=min(args.top_k, 5))
        payload["generation_summary"] = summarize_generation_results(gen_results)
        payload["generation_results"] = [
            {
                "sample_id": r.sample_id,
                "question": r.question,
                "answer": r.answer,
                "scores": r.scores,
                "rationales": r.rationales,
                "latency_ms": r.latency_ms,
                "token_usage": r.token_usage,
            }
            for r in gen_results
        ]

    if args.mode in ("e2e", "all"):
        from evaluation.eval_e2e import run_e2e_eval, summarize_e2e_results

        e2e_results = run_e2e_eval(samples)
        payload["e2e_summary"] = summarize_e2e_results(e2e_results)
        payload["e2e_results"] = [
            {
                "sample_id": r.sample_id,
                "question": r.question,
                "answer": r.answer,
                "route": r.route,
                "expected_route": r.expected_route,
                "latency_ms": r.latency_ms,
                "token_usage": r.token_usage,
                "tool_calls": r.tool_calls,
                "error": r.error,
            }
            for r in e2e_results
        ]

    # Failure Classification
    if args.mode in ("generation", "e2e", "all"):
        failures = []
        gen_by_id = {r.sample_id: r for r in (gen_results if gen_results else [])}
        e2e_by_id = {r.sample_id: r for r in (e2e_results if e2e_results else [])}

        from evaluation.failure_analysis import classify_sample_failure, summarize_failures

        for sample in samples:
            g_res = gen_by_id.get(sample.sample_id)
            e_res = e2e_by_id.get(sample.sample_id)

            err = (e_res.error if e_res else None)
            ret_hit = g_res.retrieval_hit if g_res else None
            faith = g_res.scores.get("faithfulness") if g_res else None
            rel = g_res.scores.get("answer_relevance") if g_res else None

            act_route = e_res.route if e_res else None
            exp_route = e_res.expected_route if e_res else None

            cat = classify_sample_failure(
                error=err,
                actual_route=act_route,
                expected_route=exp_route,
                retrieval_hit=ret_hit,
                faithfulness=faith,
                answer_relevance=rel,
            )
            failures.append(cat)

            # Update detail payloads
            if g_res:
                for item in payload["generation_results"]:
                    if item["sample_id"] == sample.sample_id:
                        item["failure_category"] = cat
            if e_res:
                for item in payload["e2e_results"]:
                    if item["sample_id"] == sample.sample_id:
                        item["failure_category"] = cat

        payload["failure_summary"] = summarize_failures(failures)

    # Render markdown using the dataclass-shaped retrieval runs, then drop them.
    md_payload = dict(payload)
    md_payload["retrieval_results"] = payload.pop("_retrieval_runs", None)
    report_md = _build_markdown_report(md_payload)

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=_json_default)
    md_path.write_text(report_md, encoding="utf-8")

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    print("\n" + report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
