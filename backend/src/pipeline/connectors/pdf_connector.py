"""PDF connector — read ``*.pdf`` files under a directory.

Text extraction is delegated to the parser (``pypdf`` / ``pdfplumber``); the
connector stores the raw file bytes (base64) in ``content`` so the raw tier is
truly the original document and the pipeline can re-parse later without
re-reading the file.

If no PDF backend is installed, ``fetch`` still succeeds (raw is preserved);
the parser raises a clear error at parse time.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.connectors.base import BaseConnector
from pipeline.schema import RawDocument

logger = logging.getLogger(__name__)


class PdfConnector(BaseConnector):
    """Read all ``*.pdf`` files under a directory."""

    source_id = "pdf_dir"
    source_type = "pdf"

    def __init__(self, root_dir: str | Path, limit: int | None = None):
        self.root_dir = Path(root_dir)
        self.limit = limit

    def fetch(self) -> list[RawDocument]:
        if not self.root_dir.exists():
            logger.error("PDF source dir not found: %s", self.root_dir)
            return []

        docs: list[RawDocument] = []
        fetched_at = datetime.now(timezone.utc)
        for path in sorted(self.root_dir.rglob("*.pdf")):
            if self.limit is not None and len(docs) >= self.limit:
                break
            raw_bytes = path.read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            doc_id = f"{self.source_id}-{digest[:16]}"
            docs.append(
                RawDocument(
                    doc_id=doc_id,
                    source_id=self.source_id,
                    source_type=self.source_type,
                    content=base64.b64encode(raw_bytes).decode("ascii"),
                    origin_url=str(path),
                    fetched_at=fetched_at,
                    meta={"file_name": path.name, "sha256": digest},
                )
            )
        logger.info("PdfConnector fetched %d docs from %s", len(docs), self.root_dir)
        return docs