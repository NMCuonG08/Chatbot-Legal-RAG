"""Audit logging — durable who-did-what trail.

Writes an ``AuditLog`` row for security-relevant events (auth, chat, tool
calls, admin actions). Best-effort: a logging failure must never break the
request flow, so all exceptions are swallowed and logged.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import AuditLog, _new_db_session

logger = logging.getLogger(__name__)


def log_audit(
    user_id: Optional[str],
    action: str,
    resource: Optional[str] = None,
    ip: Optional[str] = None,
    payload: Optional[Any] = None,
    db: Session = None,
) -> None:
    """Persist one audit event. Best-effort, never raises."""
    session_created = False
    try:
        if db is None:
            db = _new_db_session()
            session_created = True
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            ip=ip,
            payload=payload,
        )
        db.add(entry)
        if session_created:
            db.commit()
    except Exception as exc:
        logger.warning("Audit log write failed (non-fatal): %s", exc)
        if session_created:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if session_created:
            try:
                db.close()
            except Exception:
                pass


def list_audit_entries(
    limit: int = 200,
    offset: int = 0,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
) -> list[dict]:
    """Read recent audit entries (admin endpoint)."""
    db = _new_db_session()
    try:
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc())
        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        rows = db.execute(stmt.offset(offset).limit(limit)).scalars().all()
        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "action": r.action,
                "resource": r.resource,
                "ip": r.ip,
                "payload": r.payload,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]
    finally:
        db.close()


__all__ = ["log_audit", "list_audit_entries"]