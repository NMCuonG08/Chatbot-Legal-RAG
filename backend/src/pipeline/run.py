"""CLI entrypoint for the multi-source pipeline.

Examples
--------
JSONL (legacy train.jsonl)::

    python -m pipeline.run --source-type jsonl --path ../../data/train.jsonl --limit 10

Markdown directory::

    python -m pipeline.run --source-type markdown --path ../../data/legal_md

Multiple sources at once (add more connectors in this script if needed)::

    python -m pipeline.run --source-type jsonl --path ../../data/train.jsonl \
        --no-semantic --collection llm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend/src is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_COLLECTION_NAME  # noqa: E402
from pipeline.connectors import (  # noqa: E402
    HtmlConnector,
    JsonlQaConnector,
    MarkdownConnector,
    PdfConnector,
)
from pipeline.orchestrator import run_pipeline  # noqa: E402
from utils import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def _build_connector(source_type: str, path: Path, limit: int | None):
    if source_type == "jsonl":
        return JsonlQaConnector(path, limit=limit)
    if source_type == "markdown":
        return MarkdownConnector(path, limit=limit)
    if source_type == "html":
        return HtmlConnector(root_dir=path, limit=limit)
    if source_type == "pdf":
        return PdfConnector(path, limit=limit)
    raise ValueError(f"unknown source_type: {source_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multi-source data pipeline")
    parser.add_argument("--source-type", required=True, choices=["jsonl", "markdown", "html", "pdf"])
    parser.add_argument("--path", type=str, required=True, help="file (jsonl) or dir (markdown/html/pdf)")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--limit", type=int, default=None, help="cap docs processed (testing)")
    parser.add_argument("--no-semantic", action="store_true", help="use token chunking instead of semantic")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        logger.error("Path not found: %s", path)
        return 1

    connector = _build_connector(args.source_type, path, args.limit)
    stats = run_pipeline([connector], collection_name=args.collection, use_semantic=not args.no_semantic)
    print(stats)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())