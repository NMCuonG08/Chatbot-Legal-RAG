"""Human-in-the-loop approval workflow for sensitive tool calls.

Lifecycle:
1. agent_tools_node (tasks.py) detects a SENSITIVE_TOOLS call by a non-exempt
   role -> ``request_approval`` creates a ``ToolApproval(pending)`` row and the
   node returns a "chờ phê duyệt" response carrying ``approval_id``.
2. An admin resolves it via ``POST /approvals/{id}/decide`` -> ``decide_approval``.
3. On approve, the client re-posts the chat with ``approved_tool_id`` so the
   agent re-runs with that tool explicitly allowed (no full graph suspend).

The "explicitly allowed" set is tracked per-run in Redis (key
``approval:allowed:{run_id}``) so the gate can skip approval on the retry.
Falls back to an in-process dict when Redis is unavailable.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import select

from models import ToolApproval, _new_db_session
from rbac import SENSITIVE_TOOLS, Principal

logger = logging.getLogger(__name__)

# In-process fallback when Redis is unavailable. keyed by run_id -> set(tool_name).
_allowed_fallback: dict[str, set[str]] = {}

APPROVAL_PREFIX = "[CẦN PHÊ DUYỆT] "


def _allow_key(run_id: str) -> str:
    return f"approval:allowed:{run_id}"


def mark_tool_allowed(run_id: str, tool_name: str) -> None:
    """Record that ``tool_name`` has been approved for this run (retry gate)."""
    if not run_id:
        _allowed_fallback.setdefault("_global", set()).add(tool_name)
        return
    try:
        import redis  # local import; redis is an existing dep
        from database import settings as db_settings
        client = redis.from_url(db_settings.redis_url, decode_responses=True)
        client.sadd(_allow_key(run_id), tool_name)
        client.expire(_allow_key(run_id), 3600)
        client.close()
    except Exception:
        _allowed_fallback.setdefault(run_id, set()).add(tool_name)


def is_tool_allowed(run_id: str, tool_name: str) -> bool:
    """True when ``tool_name`` was already approved for this run."""
    if not run_id:
        return tool_name in _allowed_fallback.get("_global", set())
    try:
        import redis
        from database import settings as db_settings
        client = redis.from_url(db_settings.redis_url, decode_responses=True)
        is_member = bool(client.sismember(_allow_key(run_id), tool_name))
        client.close()
        return is_member
    except Exception:
        return tool_name in _allowed_fallback.get(run_id, set())


def request_approval(
    user_id: str,
    tool_name: str,
    args: dict,
    run_id: Optional[str] = None,
) -> ToolApproval:
    """Create a pending approval request for a sensitive tool call."""
    db = _new_db_session()
    try:
        approval = ToolApproval(
            id=uuid.uuid4().hex,
            user_id=user_id,
            tool_name=tool_name,
            args_json=args,
            status="pending",
            run_id=run_id,
        )
        db.add(approval)
        db.commit()
        db.refresh(approval)
        return approval
    finally:
        db.close()


def decide_approval(
    approval_id: str,
    decision: str,
    decided_by: str,
    note: Optional[str] = None,
) -> Optional[ToolApproval]:
    """Approve or reject a pending approval. Returns the updated row or None."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")
    db = _new_db_session()
    try:
        approval = db.get(ToolApproval, approval_id)
        if approval is None:
            return None
        if approval.status != "pending":
            return approval  # already decided — idempotent read
        approval.status = decision
        approval.decided_by = decided_by
        approval.decision_note = note
        from sqlalchemy.sql import func as _func
        approval.decided_at = _func.now()
        db.commit()
        db.refresh(approval)
        if decision == "approved" and approval.run_id:
            mark_tool_allowed(approval.run_id, approval.tool_name)
        return approval
    finally:
        db.close()


def get_approval(approval_id: str) -> Optional[ToolApproval]:
    db = _new_db_session()
    try:
        return db.get(ToolApproval, approval_id)
    finally:
        db.close()


def fetch_pending(limit: int = 100) -> list[dict]:
    db = _new_db_session()
    try:
        rows = db.execute(
            select(ToolApproval)
            .where(ToolApproval.status == "pending")
            .order_by(ToolApproval.created_at.desc())
            .limit(limit)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "tool_name": r.tool_name,
                "args": r.args_json,
                "run_id": r.run_id,
                "status": r.status,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]
    finally:
        db.close()


__all__ = [
    "APPROVAL_PREFIX",
    "mark_tool_allowed",
    "is_tool_allowed",
    "request_approval",
    "decide_approval",
    "get_approval",
    "fetch_pending",
    "evaluate_tool_gate",
    "await_approval_response",
]


def evaluate_tool_gate(
    principal: Optional[Principal],
    anticipated_tool_names,
    run_id: Optional[str] = None,
):
    """Pre-flight approval gate for a ReAct agent run.

    Given the caller's principal and the set of tool names the agent is
    anticipated to call (e.g. from ``filter_tools_for_query``), decide whether
    the run may proceed or must block on human approval:

    - exempt role (admin/lawyer) or no principal -> proceed.
    - for each anticipated sensitive tool not already approved for this run,
      create a pending ``ToolApproval`` and return it.

    Returns ``(decision, approval)`` where decision is ``"proceed"`` (approval
    None) or ``"await_approval"`` (approval = the pending ToolApproval).
    Only the FIRST blocking sensitive tool is surfaced (one approval at a
    time); once an admin approves it and the client re-posts, the gate re-runs
    with that tool now allowed and may surface the next one.
    """
    if principal is None or principal.is_approval_exempt:
        return "proceed", None
    for tool_name in anticipated_tool_names or []:
        if tool_name in SENSITIVE_TOOLS and not is_tool_allowed(run_id, tool_name):
            approval = request_approval(
                user_id=principal.user_id,
                tool_name=tool_name,
                args={},
                run_id=run_id,
            )
            return "await_approval", approval
    return "proceed", None


def await_approval_response(approval: "ToolApproval") -> str:
    """Human-facing Vietnamese message telling the caller a tool call is
    pending approval. Carries the approval_id so the client can poll/decide."""
    return (
        f"{APPROVAL_PREFIX}Lệnh công cụ '{approval.tool_name}' cần được quản trị "
        f"viên phê duyệt trước khi thực thi. Mã phê duyệt: {approval.id}. Vui lòng "
        f"chờ quản trị viên duyệt rồi gửi lại yêu cầu."
    )