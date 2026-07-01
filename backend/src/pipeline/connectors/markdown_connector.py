"""Markdown connector — passthrough source.

Reads ``*.md`` files from a directory (recursively). Markdown is already plain
text, so parsing is a passthrough — the parser keeps it verbatim.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.connectors.base import BaseConnector
from pipeline.schema import RawDocument

logger = logging.getLogger(__name__)


class MarkdownConnector(BaseConnector):
    """Read all ``*.md`` files under a directory."""

    source_id = "markdown_dir"
    source_type = "markdown"

    def __init__(self, root_dir: str | Path, limit: int | None = None):
        self.root_dir = Path(root_dir)
        self.limit = limit

    def fetch(self) -> list[RawDocument]:
        if not self.root_dir.exists():
            logger.error("Markdown source dir not found: %s", self.root_dir)
            return []

        docs: list[RawDocument] = []
        fetched_at = datetime.now(timezone.utc)
        for path in sorted(self.root_dir.rglob("*.md")):
            if self.limit is not None and len(docs) >= self.limit:
                break
            text = path.read_text(encoding="utf-8")
            doc_id = f"{self.source_id}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
            docs.append(
                RawDocument(
                    doc_id=doc_id,
                    source_id=self.source_id,
                    source_type=self.source_type,
                    content=text,
                    origin_url=str(path),
                    fetched_at=fetched_at,
                    meta={"file_name": path.name},
                )
            )
        logger.info("MarkdownConnector fetched %d docs from %s", len(docs), self.root_dir)
        return docs