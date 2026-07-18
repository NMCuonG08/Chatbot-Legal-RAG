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
        choices=["retrieval", "generation", "e2e", "all", "pairwise",
                 "scenario", "redteam"],
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
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Worker count for parallel eval (>1 = concurrent run_chat_graph).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run id (default: generated uuid).",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help="Override judge provider (groq|ollama). Default: config.JUDGE_PROVIDER.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge model. Default: config.JUDGE_MODEL.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to a baseline run JSON to diff against (regression report).",
    )
    parser.add_argument(
        "--agent-a",
        type=str,
        default=None,
        help='Pairwise mode: JSON config {"name","provider","model"} for variant A.',
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default=None,
        help='Pairwise mode: JSON config {"name","provider","model"} for variant B.',
    )
    parser.add_argument(
        "--scenario-n",
        type=int,
        default=10,
        help="Scenario mode: number of scenarios to generate/run.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="layperson",
        help="Scenario mode: user persona (layperson|business|junior_lawyer).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Scenario mode: max conversation turns per scenario.",
    )
    parser.add_argument(
        "--redteam-category",
        type=str,
        default="all",
        help="Redteam mode: probe category (all|jailbreak_legal|...).",
    )
    parser.add_argument(
        "--slice",
        action="append",
        default=None,
        help="Slice name to report (repeatable): intent|difficulty|language|oos.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from config import (
        EVAL_JUDGE_CONCURRENCY,
        EVAL_MAX_WORKERS,
        JUDGE_MODEL,
        JUDGE_PROVIDER,
        JUDGE_TEMPERATURE,
    )
    from evaluation.dataset import DEFAULT_DATA_FILE, load_eval_dataset
    from evaluation.metrics_generation import get_judge_prompt_hashes
    from evaluation.run_metadata import build_run_metadata, metadata_to_dict

    data_path = Path(args.data_file) if args.data_file else DEFAULT_DATA_FILE
    samples = load_eval_dataset(path=data_path, n_samples=args.n, seed=args.seed)
    logger.info("Evaluating with %d samples (mode=%s)", len(samples), args.mode)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_provider = (args.judge_provider or JUDGE_PROVIDER).lower()
    judge_model = args.judge_model or JUDGE_MODEL
    meta = build_run_metadata(
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_temperature=JUDGE_TEMPERATURE,
        judge_prompt_hash=get_judge_prompt_hashes(),
        run_id=args.run_id,
        extra={"mode": args.mode, "n_samples": len(samples), "seed": args.seed,
               "top_k": args.top_k, "parallel": args.parallel},
    )

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "n_samples": len(samples),
        "seed": args.seed,
        "top_k": args.top_k,
        "run_metadata": metadata_to_dict(meta),
    }

    retrieval_results = None
    gen_results = None
    e2e_results = None
    use_parallel = args.parallel and args.parallel > 1

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

        if use_parallel:
            from evaluation.parallel import ParallelConfig, run_generation_parallel
            from brain import groq_chat_complete
            cfg = ParallelConfig(
                max_workers=min(args.parallel, EVAL_MAX_WORKERS),
                judge_concurrency=EVAL_JUDGE_CONCURRENCY,
            )
            gen_results = run_generation_parallel(
                samples, cfg, meta.run_id, groq_chat_complete,
                top_k=min(args.top_k, 5),
            )
        else:
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

        if use_parallel:
            from evaluation.parallel import ParallelConfig, run_e2e_parallel
            cfg = ParallelConfig(
                max_workers=min(args.parallel, EVAL_MAX_WORKERS),
                judge_concurrency=EVAL_JUDGE_CONCURRENCY,
            )
            e2e_results = run_e2e_parallel(samples, cfg, meta.run_id)
        else:
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

    if args.mode == "pairwise":
        if not args.agent_a or not args.agent_b:
            print("--mode pairwise requires --agent-a and --agent-b JSON configs")
            return 2
        from evaluation.pairwise_eval import (
            AgentConfig,
            pairwise_summary_to_dict,
            run_pairwise_eval,
        )
        from brain import build_judge_fn

        a_cfg = json.loads(args.agent_a)
        b_cfg = json.loads(args.agent_b)
        agent_a = AgentConfig(
            name=a_cfg.get("name", "A"),
            provider=a_cfg.get("provider"),
            model=a_cfg.get("model"),
        )
        agent_b = AgentConfig(
            name=b_cfg.get("name", "B"),
            provider=b_cfg.get("provider"),
            model=b_cfg.get("model"),
        )
        judge_fn = build_judge_fn(judge_provider, judge_model, JUDGE_TEMPERATURE)
        summary = run_pairwise_eval(
            samples, agent_a, agent_b, judge_fn, top_k=args.top_k,
        )
        payload["pairwise_summary"] = pairwise_summary_to_dict(summary)
        payload["pairwise_agents"] = {
            "a": {"name": agent_a.name, "provider": agent_a.provider, "model": agent_a.model},
            "b": {"name": agent_b.name, "provider": agent_b.provider, "model": agent_b.model},
        }

    if args.mode == "scenario":
        from evaluation.scenarios import (
            generate_scenarios_from_dataset,
            summarize_scenario_scores,
        )
        from evaluation.sim_user import build_persona
        from brain import build_judge_fn

        scenarios = generate_scenarios_from_dataset(
            path=data_path, n=args.scenario_n, seed=args.seed)
        persona = build_persona(args.persona)
        judge_fn = build_judge_fn(judge_provider, judge_model, JUDGE_TEMPERATURE)
        if use_parallel:
            from evaluation.parallel import ParallelConfig, run_scenario_parallel
            pcfg = ParallelConfig(
                max_workers=min(args.parallel, EVAL_MAX_WORKERS),
                judge_concurrency=EVAL_JUDGE_CONCURRENCY,
            )
            scores = run_scenario_parallel(
                scenarios, persona, pcfg, meta.run_id,
                max_turns=args.max_turns, judge_fn=judge_fn)
        else:
            from evaluation.parallel import _run_one_scenario
            scores = [
                _run_one_scenario(sc, persona, meta.run_id,
                                  args.max_turns, judge_fn)
                for sc in scenarios
            ]
        payload["scenario_summary"] = summarize_scenario_scores(scores)
        payload["scenario_results"] = [
            {"scenario_id": s.scenario_id, "r_action": s.r_action,
             "r_output": s.r_output, "r_composite": s.r_composite,
             "success": s.success, "reached_goal": s.reached_goal,
             "n_turns": s.n_turns, "notes": s.notes}
            for s in scores
        ]
        payload["scenario_persona"] = args.persona

    if args.mode == "redteam":
        from evaluation.redteam.dataset import (
            CATEGORIES, load_redteam_dataset,
        )
        from evaluation.redteam.metrics import (
            evaluate_redteam, redteam_metrics_to_dict,
        )

        cat = None if args.redteam_category == "all" else args.redteam_category
        if cat and cat not in CATEGORIES:
            print(f"Unknown redteam category: {cat}")
            return 2
        probes = load_redteam_dataset(category=cat)
        if not probes:
            print("No red-team probes loaded (probes.jsonl missing?).")
            return 2
        from tasks import run_chat_graph
        from evaluation.parallel import _conversation_id

        results = []
        for p in probes:
            try:
                res = run_chat_graph([], p.prompt,
                                     conversation_id=_conversation_id(
                                         meta.run_id, p.probe_id))
                results.append({
                    "response": res.get("response", ""),
                    "route": res.get("route"),
                    "blocked": False,
                    "verify_verdict": res.get("verify_verdict"),
                    "escalated": bool(res.get("escalated")),
                })
            except Exception as exc:
                logger.warning("redteam probe %s failed: %s", p.probe_id, exc)
                results.append({"response": "", "route": None, "blocked": True,
                                "verify_verdict": None, "escalated": False})
        metrics = evaluate_redteam(probes, results)
        payload["redteam_summary"] = redteam_metrics_to_dict(metrics)
        payload["redteam_results"] = [
            {"probe_id": p.probe_id, "category": p.category,
             "response": r.get("response", ""), "blocked": r.get("blocked"),
             "route": r.get("route")}
            for p, r in zip(probes, results)
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

    # ----- P5: slicing + extended metrics (opt-in via --slice) -----
    if args.slice:
        from evaluation.slicing import (
            slice_by_difficulty, slice_by_intent, slice_by_language,
            slice_by_oos, summarize_by_slice,
        )
        slice_fns = {
            "intent": slice_by_intent, "difficulty": slice_by_difficulty,
            "language": slice_by_language, "oos": slice_by_oos,
        }
        slices_payload = {}
        for name in args.slice:
            fn = slice_fns.get(name)
            if fn is None:
                continue
            slices_payload[name] = {
                k: {"n": len(v)} for k, v in fn(samples).items()
            }
        payload["slices"] = slices_payload

    # Extended metrics over e2e/gen results when present.
    if e2e_results:
        from evaluation.metrics_extended import hallucination_rate, latency_p99
        latencies = [r.latency_ms for r in e2e_results if r.latency_ms]
        payload["latency_p99_ms"] = latency_p99(latencies)
    if gen_results:
        from evaluation.metrics_extended import hallucination_rate
        faith = [r.scores.get("faithfulness", 0.0) for r in gen_results]
        payload["hallucination_rate"] = hallucination_rate(faith)

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=_json_default)
    md_path.write_text(report_md, encoding="utf-8")

    # Persist a run-keyed copy for regression diffs (eval_reports/runs/<run_id>.json).
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{meta.run_id}.json"
    with run_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    logger.info("Wrote run %s", run_path)

    # Optional regression diff against a baseline run.
    if args.baseline:
        from evaluation.regression import diff_runs, write_regression_report
        report = diff_runs(args.baseline, run_path)
        reg_path = write_regression_report(report, output_dir / "regression_report.md")
        logger.info("Regression gate: %s (wrote %s)", report.gate, reg_path)
        print(f"\nRegression gate: {report.gate}")
        for reason in report.gate_reasons:
            print(f"  - {reason}")

    print("\n" + report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
