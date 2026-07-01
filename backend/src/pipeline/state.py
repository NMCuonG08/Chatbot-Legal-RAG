"""Pipeline state store — one shared table for every source.

A document moves through statuses ``fetched → parsed → chunked → embedded``
(or ``failed``). Before processing, the orchestrator checks the status — if a
doc is already ``embedded`` it is skipped (idempotency). One failing document
never blocks another: each is marked independently.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.sql import func

from database import engine
from models import Base, _new_db_session

logger = logging.getLogger(__name__)

# Status values (single source of truth — do not branch the pipeline on these
# elsewhere; they describe lifecycle, not source type).
STATUS_FETCHED = "fetched"
STATUS_PARSED = "parsed"
STATUS_CHUNKED = "chunked"
STATUS_EMBEDDED = "embedded"
STATUS_FAILED = "failed"

# A status at or beyond this point means "done, skip on re-run".
TERMINAL_STATUSES = {STATUS_EMBEDDED}


class PipelineDocument(Base):
    """Shared state row for every document from every connector."""

    __tablename__ = "pipeline_documents"

    doc_id = Column(String(128), primary_key=True)
    source_id = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, index=True)
    error = Column(Text, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


def ensure_pipeline_schema() -> None:
    """Create the ``pipeline_documents`` table if missing."""
    try:
        Base.metadata.create_all(bind=engine, tables=[PipelineDocument.__table__])
    except Exception as exc:  # DB may not be ready during tests / startup
        logger.warning("Pipeline state table not ready: %s", exc)


def get_status(doc_id: str) -> Optional[str]:
    db = _new_db_session()
    try:
        row = db.get(PipelineDocument, doc_id)
        return row.status if row else None
    finally:
        db.close()


def is_done(doc_id: str, target_status: str = STATUS_EMBEDDED) -> bool:
    """True if the doc already reached ``target_status`` (idempotency guard)."""
    current = get_status(doc_id)
    if current is None:
        return False
    if target_status == STATUS_EMBEDDED:
        return current in TERMINAL_STATUSES
    return current == target_status


def mark_status(doc_id: str, source_id: str, status: str, error: Optional[str] = None) -> None:
    """Upsert the status row for a document."""
    db = _new_db_session()
    try:
        row = db.get(PipelineDocument, doc_id)
        if row is None:
            row = PipelineDocument(doc_id=doc_id, source_id=source_id, status=status, error=error)
            db.add(row)
        else:
            row.status = status
            row.source_id = source_id
            row.error = error
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Failed to mark %s -> %s: %s", doc_id, status, exc)
        raise
    finally:
        db.close()


def list_failed(limit: int = 100) -> list[dict]:
    """Return recent failed docs for retry/debugging."""
    from sqlalchemy import select

    db = _new_db_session()
    try:
        rows = db.execute(
            select(PipelineDocument)
            .where(PipelineDocument.status == STATUS_FAILED)
            .order_by(PipelineDocument.updated_at.desc())
            .limit(limit)
        ).scalars().all()
        return [{"doc_id": r.doc_id, "source_id": r.source_id, "error": r.error} for r in rows]
    finally:
        db.close()