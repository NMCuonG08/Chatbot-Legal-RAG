"""JSONL Q&A connector — wraps the existing ``train.jsonl`` RAG format.

Each line: ``{"question": "...", "context": "..."}`` (legacy ``answer`` key
also accepted). Produces one :class:`RawDocument` per line with
``source_type="json"``. ``content`` keeps the raw JSON line so the parser can
reconstruct ``question + context``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.connectors.base import BaseConnector
from pipeline.schema import RawDocument

logger = logging.getLogger(__name__)

PARSER_VERSION = "1"


class JsonlQaConnector(BaseConnector):
    """Read a JSONL file of {question, context} pairs."""

    source_id = "jsonl_qa"
    source_type = "json"

    def __init__(self, file_path: str | Path, limit: int | None = None):
        self.file_path = Path(file_path)
        self.limit = limit

    def fetch(self) -> list[RawDocument]:
        if not self.file_path.exists():
            logger.error("JSONL source file not found: %s", self.file_path)
            return []

        docs: list[RawDocument] = []
        fetched_at = datetime.now(timezone.utc)
        with self.file_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if self.limit is not None and idx >= self.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d JSON decode error: %s", idx, exc)
                    continue
                question = (data.get("question") or "").strip()
                context = (data.get("context") or data.get("answer") or "").strip()
                if not question or not context:
                    continue
                content = json.dumps(
                    {"question": question, "context": context},
                    ensure_ascii=False,
                )
                doc_id = self._doc_id(content)
                docs.append(
                    RawDocument(
                        doc_id=doc_id,
                        source_id=self.source_id,
                        source_type=self.source_type,
                        content=content,
                        origin_url=str(self.file_path),
                        fetched_at=fetched_at,
                        meta={"line_index": idx},
                    )
                )
        logger.info("JsonlQaConnector fetched %d docs from %s", len(docs), self.file_path)
        return docs

    def _doc_id(self, content: str) -> str:
        return f"{self.source_id}-{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"