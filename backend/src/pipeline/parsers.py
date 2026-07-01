"""Content parsers — branch on ``source_type``, never on source.

The parser never cares *where* a document came from (the connector's job), only
*what kind* of content it is. OCR / tag-stripping / JSON unwrapping are
strategies selected by ``source_type``.

A missing optional dependency (e.g. no PDF backend) raises a clear
``RuntimeError`` at parse time instead of crashing import.
"""

from __future__ import annotations

import base64
import json
import logging
import re

from pipeline.schema import ParsedDocument, RawDocument

logger = logging.getLogger(__name__)

PARSER_VERSION = "1"

# Tags whose text content is not useful for legal retrieval.
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def parse(doc: RawDocument) -> ParsedDocument:
    """Dispatch to the right extractor based on ``doc.source_type``."""
    if doc.source_type == "json":
        text = _parse_json(doc)
        parser_used = "json_qa"
    elif doc.source_type == "markdown":
        text = doc.content
        parser_used = "markdown_passthrough"
    elif doc.source_type == "html":
        text = _parse_html(doc)
        parser_used = "html_regex"
    elif doc.source_type == "pdf":
        text = _parse_pdf(doc)
        parser_used = "pdf_text"
    else:
        raise ValueError(f"Unknown source_type: {doc.source_type!r}")

    if not text.strip():
        raise ValueError(f"Parsed text empty for doc {doc.doc_id}")

    return ParsedDocument(
        doc_id=doc.doc_id,
        source_id=doc.source_id,
        source_type=doc.source_type,
        text=text,
        parser_used=parser_used,
        parser_version=PARSER_VERSION,
        raw_doc_id=doc.doc_id,
        fetched_at=doc.fetched_at,
        meta={**doc.meta, "origin_url": doc.origin_url},
    )


def _parse_json(doc: RawDocument) -> str:
    """Reconstruct ``question + context`` — matches legacy import_data behavior."""
    data = json.loads(doc.content)
    question = (data.get("question") or "").strip()
    context = (data.get("context") or data.get("answer") or "").strip()
    return f"{question} {context}".strip()


def _parse_html(doc: RawDocument) -> str:
    """Strip tags with a regex fallback (bs4 not in requirements)."""
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(doc.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return _WS.sub(" ", soup.get_text(separator=" ")).strip()
    except ImportError:
        text = _SCRIPT_STYLE.sub(" ", doc.content)
        text = _TAG.sub(" ", text)
        return _WS.sub(" ", text).strip()


def _parse_pdf(doc: RawDocument) -> str:
    """Extract text from the base64-encoded PDF bytes."""
    raw_bytes = base64.b64decode(doc.content)
    try:
        import io

        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except ImportError:
        pass
    try:
        import io

        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()
    except ImportError as exc:
        raise RuntimeError(
            "No PDF backend available. Install `pypdf` or `pdfplumber` to parse "
            "PDF sources. Raw content is preserved in the raw tier."
        ) from exc