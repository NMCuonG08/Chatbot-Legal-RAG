"""Model / configuration benchmark harness for the RAG evaluation suite.

Runs the generation eval across several run-configs (each typically a
different LLM model / provider), collects the per-config summary, and emits a
side-by-side comparison table so model choices can be judged on
faithfulness / answer_relevance / context_precision / latency / cost.

A run-config is a label plus a set of environment-variable overrides applied
only for the duration of that config's run (e.g. ``LLM_MODEL``,
``LLM_PROVIDER``, ``GROQ_API_KEY``). The harness reuses the existing
``run_generation_eval`` / ``summarize_generation_results`` machinery so a
single source of truth drives both the per-run report and this comparison.

Examples
--------
Compare two Groq models on 30 samples::

    python -m evaluation.evaluate_model --n 30 \
        --config llama-8b:LLM_MODEL=llama-3.1-8b-instant \
        --config llama-70b:LLM_MODEL=llama-3.1-70b-versatile \
        --output eval_reports

Compare provider fallback (Groq vs local Vietnamese LLM)::

    python -m evaluation.evaluate_model --n 20 \
        --config groq:LLM_PROVIDER=groq,LLM_MODEL=llama-3.1-8b-instant \
        --config local:LLM_PROVIDER=ollama,LLM_MODEL=vietnamese-legal-llm

The script writes ``models_report.json`` and ``models_report.md`` into the
output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Put backend/src on sys.path so sibling eval modules resolve.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

load_dotenv(SRC_DIR.parent / ".env")

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """One benchmark run: a label plus env overrides applied for its duration."""

    label: str
    env_overrides: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: str) -> "RunConfig":
        """Parse ``label:KEY=VAL,KEY=VAL`` from a CLI --config argument."""
        if ":" not in raw:
            raise argparse.ArgumentTypeError(
                f"Invalid --config '{raw}'. Expected 'label:KEY=VAL,KEY=VAL,...'"
            )
        label, rest = raw.split(":", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"Config label empty in '{raw}'")
        overrides: Dict[str, str] = {}
        if rest.strip():
            for pair in rest.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise argparse.ArgumentTypeError(
                        f"Invalid env override '{pair}' in '{raw}'. Expected KEY=VAL"
                    )
                key, val = pair.split("=", 1)
                overrides[key.strip()] = val.strip()
        return cls(label=label, env_overrides=overrides)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def _apply_env(overrides: Dict[str, str]) -> Dict[str, str]:
    """Apply env overrides, returning the prior values for later restoration."""
    snapshot: Dict[str, str] = {}
    for key, val in overrides.items():
        snapshot[key] = os.environ.get(key, "")
        os.environ[key] = val
    return snapshot


def _restore_env(snapshot: Dict[str, str]) -> None:
    for key, val in snapshot.items():
        if val == "" and key not in os.environ:
            continue
        os.environ[key] = val


def _run_one(config: RunConfig, samples, top_k: int) -> Dict:
    """Run generation eval for one config under its env overrides."""
    logger.info("=== Running config: %s (env=%s) ===", config.label, config.env_overrides)
    snapshot = _apply_env(config.env_overrides)
    try:
        from eval_generation import run_generation_eval, summarize_generation_results

        results = run_generation_eval(samples, top_k=top_k)
        summary = summarize_generation_results(results)
    finally:
        _restore_env(snapshot)

    summary["config_label"] = config.label
    return summary


def _format_table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _comparison_table(summaries: List[Dict]) -> str:
    headers = [
        "config",
        "n",
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "lat_mean_ms",
        "lat_p95_ms",
        "total_tokens",
        "est_cost_usd",
    ]
    rows = []
    for s in summaries:
        rows.append([
            s.get("config_label", ""),
            s.get("n_queries", 0),
            f"{s.get('faithfulness_mean', 0):.3f}",
            f"{s.get('answer_relevance_mean', 0):.3f}",
            f"{s.get('context_precision_mean', 0):.3f}",
            f"{s.get('latency_ms_mean', 0):.1f}",
            f"{s.get('latency_ms_p95', 0):.1f}",
            s.get("total_tokens", 0),
            f"${s.get('estimated_cost_usd', 0.0):.5f}",
        ])
    return "## Model / Config Comparison\n\n" + _format_table(headers, rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RAG eval across model/config variants")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        type=RunConfig.parse,
        help="A run config 'label:KEY=VAL,KEY=VAL,...'. Repeat for each variant.",
    )
    parser.add_argument("--n", type=int, default=30, help="Number of samples per config")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k for generation")
    parser.add_argument(
        "--output",
        type=str,
        default="eval_reports",
        help="Directory to write models_report.json and models_report.md",
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

    from dataset import DEFAULT_DATA_FILE, load_eval_dataset

    data_path = Path(args.data_file) if args.data_file else DEFAULT_DATA_FILE
    samples = load_eval_dataset(path=data_path, n_samples=args.n, seed=args.seed)
    logger.info("Benchmarking %d configs on %d samples", len(args.config), len(samples))

    summaries: List[Dict] = []
    for config in args.config:
        try:
            summary = _run_one(config, samples, top_k=args.top_k)
            summaries.append(summary)
        except Exception as exc:
            logger.exception("Config %s failed: %s", config.label, exc)
            summaries.append({
                "config_label": config.label,
                "error": f"{type(exc).__name__}: {exc}",
                "n_queries": 0,
            })

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "seed": args.seed,
        "top_k": args.top_k,
        "configs": [asdict(c) for c in args.config],
        "summaries": summaries,
    }

    json_path = output_dir / "models_report.json"
    md_path = output_dir / "models_report.md"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)

    md = f"# Model / Config Benchmark\n\ngenerated_at: {payload['generated_at']}\n\n"
    md += _comparison_table(summaries)
    md_path.write_text(md, encoding="utf-8")

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())