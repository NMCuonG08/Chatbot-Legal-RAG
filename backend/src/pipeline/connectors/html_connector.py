"""HTML connector — fetch local ``*.html`` files or remote URLs.

Extracting visible text from the HTML happens in :mod:`pipeline.parsers`
(branches on ``source_type``); the connector only fetches raw HTML so the raw
tier stays the original bytes.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pipeline.connectors.base import BaseConnector
from pipeline.schema import RawDocument

logger = logging.getLogger(__name__)


class HtmlConnector(BaseConnector):
    """Read ``*.html`` files under a directory. (URL list support via ``urls=``)."""

    source_id = "html_dir"
    source_type = "html"

    def __init__(
        self,
        root_dir: str | Path | None = None,
        urls: Iterable[str] | None = None,
        limit: int | None = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else None
        self.urls = list(urls) if urls else []
        self.limit = limit

    def fetch(self) -> list[RawDocument]:
        docs: list[RawDocument] = []
        fetched_at = datetime.now(timezone.utc)

        for html, origin, name in self._iter_sources():
            if self.limit is not None and len(docs) >= self.limit:
                break
            doc_id = f"{self.source_id}-{hashlib.sha256(html.encode('utf-8')).hexdigest()[:16]}"
            docs.append(
                RawDocument(
                    doc_id=doc_id,
                    source_id=self.source_id,
                    source_type=self.source_type,
                    content=html,
                    origin_url=origin,
                    fetched_at=fetched_at,
                    meta={"file_name": name},
                )
            )
        logger.info("HtmlConnector fetched %d docs", len(docs))
        return docs

    def _iter_sources(self):
        if self.root_dir and self.root_dir.exists():
            for path in sorted(self.root_dir.rglob("*.html")):
                yield path.read_text(encoding="utf-8"), str(path), path.name
        for url in self.urls:
            try:
                import requests  # local import: optional remote fetch

                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                yield resp.text, url, url.rsplit("/", 1)[-1] or "index.html"
            except Exception as exc:  # noqa: BLE001 — a dead URL must not kill the batch
                logger.warning("HTML fetch failed for %s: %s", url, exc)