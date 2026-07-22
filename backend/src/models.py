import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional
from xml.dom import ValidationErr

from sqlalchemy import JSON, Boolean, Column, DateTime, Index, Integer, String, Text, delete
from sqlalchemy.future import select
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import func

from cache import get_conversation_id
from database import engine, get_db
from utils import setup_logging

class Base(DeclarativeBase):
    pass


def _new_db_session() -> Session:
    return next(get_db())


setup_logging()
logger = logging.getLogger(__name__)


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(50), nullable=False, default="")
    bot_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    message = Column(String)  # Assuming TextField is equivalent to String in SQLAlchemy
    is_request = Column(Boolean, default=True)
    completed = Column(Boolean, default=False)
    sources = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )

    # Audit 6.2 — hot-path indexes. load_conversation filters by
    # conversation_id ALONE (the per-message lookup), and list/delete paths
    # filter by user_id ALONE. A composite (user_id, conversation_id) would NOT
    # serve the conversation_id-only hot path (B-tree leftmost-prefix), so both
    # columns get standalone indexes.
    __table_args__ = (
        Index("idx_chat_conv_conversation_id", "conversation_id"),
        Index("idx_chat_conv_user_id", "user_id"),
    )


def load_conversation(conversation_id: str):
    db = _new_db_session()
    try:
        result = db.execute(
            select(ChatConversation)
            .where(ChatConversation.conversation_id == conversation_id)
            .order_by(ChatConversation.created_at)
        )
        return result.scalars().all()
    finally:
        db.close()


async def read_conversation(conversation_id: str):
    db = _new_db_session()
    try:
        result = db.execute(
            select(ChatConversation).where(
                ChatConversation.conversation_id == conversation_id
            )
        )
        db_conversation = result.scalars().first()
        if db_conversation is None:
            raise ValidationErr("Conversation not found")
        return db_conversation
    finally:
        db.close()


def convert_conversation_to_openai_messages(user_conversations):
    conversation_list = [
        {"role": "system", "content": "You are an amazing virtual assistant"}
    ]

    for conversation in user_conversations:
        role = "assistant" if not conversation.is_request else "user"
        content = str(conversation.message)
        conversation_list.append({"role": role, "content": content})

    logging.info(f"Create conversation to {conversation_list}")

    return conversation_list


def update_chat_conversation(
    bot_id: str, user_id: str, message: str, is_request: bool = True, sources: list = None, conversation_id: str = None
):
    db = _new_db_session()
    if not conversation_id:
        conversation_id = get_conversation_id(bot_id, user_id)

    new_conversation = ChatConversation(
        conversation_id=conversation_id,
        bot_id=bot_id,
        user_id=user_id,
        message=message,
        is_request=is_request,
        completed=not is_request,
        sources=json.dumps(sources) if sources else None,
    )
    # Step 4: Save the ChatConversation instance
    try:
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
    finally:
        db.close()

    logger.info(f"Create message for conversation {conversation_id}")

    return conversation_id


def get_conversation_messages(conversation_id):
    user_conversations = load_conversation(conversation_id)
    return convert_conversation_to_openai_messages(user_conversations)


def list_user_conversations(user_id: str, limit: int = 100, offset: int = 0):
    ensure_database_schema()
    db = _new_db_session()
    try:
        conversations = db.execute(
            select(ChatConversation)
            .where(ChatConversation.user_id == user_id)
            .order_by(ChatConversation.created_at.asc())
        ).scalars().all()

        sessions_map = {}
        for c in conversations:
            cid = c.conversation_id
            if cid not in sessions_map:
                sessions_map[cid] = {
                    "id": c.id,
                    "conversation_id": cid,
                    "bot_id": c.bot_id,
                    "user_id": c.user_id,
                    "question": "",
                    "response": "",
                    "sources": [],
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                    "turns": [],
                }
            sess = sessions_map[cid]
            if c.is_request and not sess["question"]:
                sess["question"] = c.message
            elif not c.is_request and not sess["response"]:
                sess["response"] = c.message
                if getattr(c, "sources", None):
                    try:
                        sess["sources"] = json.loads(c.sources)
                    except Exception:
                        pass
            sess["updated_at"] = c.updated_at

            turn_sources = []
            if not c.is_request and getattr(c, "sources", None):
                try:
                    turn_sources = json.loads(c.sources)
                except Exception:
                    pass

            sess["turns"].append({
                "id": c.id,
                "is_request": c.is_request,
                "message": c.message,
                "sources": turn_sources,
                "created_at": str(c.created_at or ""),
            })

        result = list(sessions_map.values())
        result.sort(key=lambda s: str(s.get("updated_at") or s.get("created_at") or ""), reverse=True)
        return result[offset : offset + limit]
    finally:
        db.close()


def list_unique_session_ids(limit: int = 100) -> list[dict]:
    ensure_database_schema()
    db = _new_db_session()
    try:
        # Create a subquery to find the min(id) representing the first message of each user_id
        subq = (
            select(
                ChatConversation.user_id,
                func.min(ChatConversation.id).label("min_id")
            )
            .group_by(ChatConversation.user_id)
            .subquery()
        )
        
        # Select user_id and message of the first conversation turn, sorted by last active
        query = (
            select(
                ChatConversation.user_id,
                ChatConversation.message
            )
            .join(subq, ChatConversation.id == subq.c.min_id)
            .order_by(ChatConversation.updated_at.desc())
            .limit(limit)
        )
        
        results = db.execute(query).all()
        
        sessions = []
        for r_user_id, r_message in results:
            if not r_user_id:
                continue
            title = r_message[:35] + "..." if r_message else f"Hội thoại {r_user_id[:8]}"
            sessions.append({
                "session_id": r_user_id,
                "title": title
            })
        return sessions
    finally:
        db.close()

def delete_user_conversations(user_id: str):
    """Delete all chat history for a specific user"""
    db = _new_db_session()
    try:
        db.execute(
            delete(ChatConversation).where(ChatConversation.user_id == user_id)
        )
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting user conversations: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def delete_single_conversation(user_id: str, conversation_id: str) -> bool:
    """Delete a single specific conversation session for a user by conversation_id or row id"""
    db = _new_db_session()
    try:
        stmt = delete(ChatConversation).where(
            ChatConversation.user_id == user_id,
            ChatConversation.conversation_id == conversation_id,
        )
        res = db.execute(stmt)
        if (res.rowcount or 0) == 0 and str(conversation_id).isdigit():
            row_id = int(conversation_id)
            target_row = db.query(ChatConversation).filter(ChatConversation.id == row_id).first()
            if target_row and target_row.conversation_id:
                db.execute(
                    delete(ChatConversation).where(
                        ChatConversation.user_id == user_id,
                        ChatConversation.conversation_id == target_row.conversation_id,
                    )
                )
            else:
                db.execute(
                    delete(ChatConversation).where(
                        ChatConversation.user_id == user_id,
                        ChatConversation.id == row_id,
                    )
                )
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting single conversation {conversation_id}: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def delete_all_conversations():
    """Delete all chat conversations for all sessions"""
    db = _new_db_session()
    try:
        db.execute(delete(ChatConversation))
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting all conversations: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def delete_user_episodes(user_id: str) -> bool:
    """Delete all long-term episodic facts (MySQL UserEpisode rows) for a user.

    Mirrors the Qdrant user_episodes purge so a history wipe also removes the
    facts that resurface via get_user_episodes. Without this, deleted-conversation
    facts survive in MySQL and can be re-embedded/surfaced later.
    """
    db = _new_db_session()
    try:
        db.execute(delete(UserEpisode).where(UserEpisode.user_id == user_id))
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting user episodes: {e}")
        db.rollback()
        return False
    finally:
        db.close()


class Document(Base):
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String(2000), nullable=False, default="")
    content = Column(String)  # Assuming TextField is equivalent to String in SQLAlchemy
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(100), nullable=False, index=True)      # e.g., "doc-12" or "train-123"
    chunk_id = Column(String(50), nullable=False, index=True)     # UUID string or integer string
    chunk_hash = Column(String(64), nullable=False)               # MD5 hash of chunk text
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Structured legal metadata (Phase 1 upgrade). All nullable: only chunks
    # whose text actually cites a statute/law get these. Populated at ingest
    # from ``legal_metadata.extract_legal_metadata`` + ``legal_effectivity``.
    law_name = Column(String(300), nullable=True, index=True)
    article_number = Column(Integer, nullable=True, index=True)
    clause_number = Column(Integer, nullable=True)
    point_letter = Column(String(10), nullable=True)
    document_number = Column(Integer, nullable=True, index=True)
    document_year = Column(Integer, nullable=True, index=True)
    document_type = Column(String(20), nullable=True, index=True)
    effectivity_status = Column(String(20), nullable=True, index=True)


class UserEpisode(Base):
    __tablename__ = "user_episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    summary = Column(String)  # Summary of episodic interaction
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserProfile(Base):
    """Structured long-term memory (CoALA semantic layer) — KV profile per user.

    Distinct from ``UserEpisode`` (audit raw facts) and the Qdrant
    ``user_episodes`` collection (semantic recall): this is the cheap,
    directly-addressable structured profile the agent reads via
    ``recall_user_memory_tool``'s ``profile`` key and that
    ``save_episodic_memory_task`` merges into from the extraction prompt's
    ``structured`` field. Auto-created by ``ensure_database_schema`` (create_all).
    """
    __tablename__ = "user_profiles"

    user_id = Column(String(100), primary_key=True)
    name = Column(String(200), nullable=True)
    birth_year = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    location = Column(String(200), nullable=True)
    case_type = Column(String(50), nullable=True, index=True)
    case_summary = Column(Text, nullable=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AgentFeedback(Base):
    """RLHF user-feedback store (Phase 4). Mirror of 👍/👎 on an assistant
    message. Auto-created by ``ensure_database_schema`` (create_all). The
    Qdrant ``rlhf_good_answers`` collection (see ``rlhf_store``) holds the
    good answers used for few-shot injection + rerank up-weighting; this
    table is the durable MySQL audit trail for ALL feedback (good + bad).
    """
    __tablename__ = "agent_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), index=True)
    conversation_id = Column(String(100), index=True)
    message_id = Column(String(100))
    question = Column(String)
    response = Column(String)
    sources_json = Column(JSON)
    rating = Column(String(10))  # "good" | "bad"
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GraphRun(Base):
    """One LangGraph invocation (a single chat turn through the CRAG graph)."""
    __tablename__ = "graph_runs"

    id = Column(String(64), primary_key=True)  # run_id (uuid4 hex)
    thread_id = Column(String(64), nullable=False, index=True)  # = conversation_id
    user_id = Column(String(100), nullable=True)
    question = Column(String)
    route = Column(String(32), nullable=True)
    final_response = Column(String)
    status = Column(String(16), default="running")  # running|completed|error
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    reflection_count = Column(Integer, default=0)
    tool_calls_json = Column(JSON)  # final ReAct tool-call accumulator


class AgentStep(Base):
    """One trace event within a GraphRun (node start/end, tool call, handoff, llm)."""
    __tablename__ = "agent_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, index=True)
    node = Column(String(64), nullable=False)
    step_index = Column(Integer, default=0)
    event_type = Column(String(32), nullable=False)  # node_start|node_end|tool_call|handoff|llm
    payload = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---- Phase 2 — Security: users, audit log, tool approvals ----

class User(Base):
    """Application user with a role (admin|lawyer|user|guest).

    Auth is JWT (see auth.py); password_hash is bcrypt via passlib. ``user_id``
    columns on other tables still store the self-asserted client id for
    backward compatibility, but when JWT auth is enforced the caller's User.id
    (string uuid) is used instead.
    """
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)  # uuid4 hex
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=True, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="user")  # admin|lawyer|user|guest
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """Durable who-did-what trail. Written by audit.log_audit for auth, chat,
    tool, and admin events. Not a replacement for stdout logs — this is the
    queryable, tamper-evident store."""
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), index=True)
    action = Column(String(64), nullable=False, index=True)  # login|register|chat|tool_call|admin|...
    resource = Column(String(255), nullable=True)
    ip = Column(String(64), nullable=True)
    payload = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class ToolApproval(Base):
    """Human-in-the-loop approval request for a sensitive tool call.

    Lifecycle: pending -> approved|rejected. Created by approval.request_approval
    when a non-exempt role triggers a SENSITIVE_TOOLS entry. Resolved by an admin
    via POST /approvals/{id}/decide. On approve, the agent is re-invoked with the
    tool explicitly allowed (see agent_tools_node gate in tasks.py).
    """
    __tablename__ = "tool_approvals"

    id = Column(String(64), primary_key=True)  # uuid4 hex
    user_id = Column(String(100), nullable=False, index=True)
    tool_name = Column(String(128), nullable=False)
    args_json = Column(JSON)
    status = Column(String(16), nullable=False, default="pending", index=True)  # pending|approved|rejected
    run_id = Column(String(64), nullable=True, index=True)
    decided_by = Column(String(100), nullable=True)
    decision_note = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    decided_at = Column(DateTime(timezone=True), nullable=True)


def save_graph_run(run_id: str, thread_id: str, user_id: str | None, question: str,
                   status: str = "running", db: Session = None) -> GraphRun:
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        run = GraphRun(
            id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            question=question,
            status=status,
        )
        db.add(run)
        if session_created:
            db.commit()
            db.refresh(run)
        return run
    finally:
        if session_created:
            db.close()


def update_graph_run(run_id: str, *, status: str | None = None,
                     final_response: str | None = None, route: str | None = None,
                     reflection_count: int | None = None,
                     tool_calls_json=None, db: Session = None) -> None:
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        run = db.get(GraphRun, run_id)
        if run is None:
            return
        if status is not None:
            run.status = status
        if final_response is not None:
            run.final_response = final_response
        if route is not None:
            run.route = route
        if reflection_count is not None:
            run.reflection_count = reflection_count
        if tool_calls_json is not None:
            run.tool_calls_json = tool_calls_json
        if status in ("completed", "error"):
            run.ended_at = func.now()
        if session_created:
            db.commit()
    finally:
        if session_created:
            db.close()


def save_agent_step(run_id: str, node: str, step_index: int, event_type: str,
                    payload, db: Session = None) -> None:
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        step = AgentStep(
            run_id=run_id,
            node=node,
            step_index=step_index,
            event_type=event_type,
            payload=payload,
        )
        db.add(step)
        if session_created:
            db.commit()
    finally:
        if session_created:
            db.close()


def load_agent_steps(run_id: str, db: Session = None) -> list[AgentStep]:
    """Load persisted trace events for a run, ordered by emission sequence.

    Used by the SSE stream endpoint to replay events that were published to
    Redis pub/sub before a client subscribed (the common case when the UI
    opens the trace expander only after the Celery task has finished).
    """
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        return (
            db.query(AgentStep)
            .filter(AgentStep.run_id == run_id)
            .order_by(AgentStep.step_index.asc(), AgentStep.id.asc())
            .all()
        )
    finally:
        if session_created:
            db.close()


def load_session_trace(thread_id: str, db: Session = None) -> list[AgentStep]:
    """Load all trace events for a whole conversation session (multi-turn replay).

    ``run_id`` is per-message, so a multi-turn conversation spans several runs.
    ``GraphRun.thread_id`` (= conversation_id) is the session key. This joins
    ``AgentStep.run_id == GraphRun.id`` filtered by ``thread_id`` and orders by
    run creation then step_index, so the caller can replay an entire session
    as one trace (senior observability: trace_id spans the whole session).

    Returns an empty list if the thread_id has no runs.
    """
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        return (
            db.query(AgentStep)
            .join(GraphRun, AgentStep.run_id == GraphRun.id)
            .filter(GraphRun.thread_id == thread_id)
            .order_by(GraphRun.started_at.asc(), AgentStep.step_index.asc(), AgentStep.id.asc())
            .all()
        )
    finally:
        if session_created:
            db.close()


def save_user_episode(user_id: str, summary: str, db: Session = None):
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        new_episode = UserEpisode(
            user_id=user_id,
            summary=summary
        )
        db.add(new_episode)
        if session_created:
            db.commit()
        return new_episode
    finally:
        if session_created:
            db.close()


def get_user_episodes(user_id: str, limit: int = 20) -> list[UserEpisode]:
    db = _new_db_session()
    try:
        return db.execute(
            select(UserEpisode)
            .where(UserEpisode.user_id == user_id)
            .order_by(UserEpisode.created_at.desc())
            .limit(limit)
        ).scalars().all()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# UserProfile accessors (CoALA structured/semantic memory).
# All merge operations update ONLY non-null fields — a null in the incoming
# payload NEVER overwrites an existing value (idempotent partial update).
# ---------------------------------------------------------------------------

# Whitelist of mergeable fields + their coercion. Anything outside this map is
# ignored so a malformed extraction payload can't write arbitrary columns.
_PROFILE_FIELDS = {
    "name": lambda v: str(v) if v not in (None, "") else None,
    "birth_year": lambda v: int(v) if str(v).strip().isdigit() else None,
    "gender": lambda v: str(v) if v not in (None, "") else None,
    "location": lambda v: str(v) if v not in (None, "") else None,
    "case_type": lambda v: str(v) if v not in (None, "") else None,
    "case_summary": lambda v: str(v) if v not in (None, "") else None,
}


def _coerce_profile_field(field: str, value):
    try:
        return _PROFILE_FIELDS[field](value)
    except Exception:
        return None


def get_user_profile(user_id: str) -> dict | None:
    """Return the user's structured profile as a dict, or None if absent."""
    if not user_id:
        return None
    db = _new_db_session()
    try:
        row = db.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        ).scalars().first()
        if row is None:
            return None
        return {
            "user_id": row.user_id,
            "name": row.name,
            "birth_year": row.birth_year,
            "gender": row.gender,
            "location": row.location,
            "case_type": row.case_type,
            "case_summary": row.case_summary,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
    finally:
        db.close()


def save_user_profile(user_id: str, fields: dict, db: Session = None) -> UserProfile:
    """Upsert a full profile row. ``fields`` keys must be in _PROFILE_FIELDS."""
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        row = db.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        ).scalars().first()
        if row is None:
            row = UserProfile(user_id=user_id)
            db.add(row)
        for field in _PROFILE_FIELDS:
            if field in fields:
                setattr(row, field, _coerce_profile_field(field, fields[field]))
        if session_created:
            db.commit()
        return row
    finally:
        if session_created:
            db.close()


def merge_user_profile(user_id: str, structured: dict, db: Session = None) -> dict | None:
    """Merge ONLY non-null incoming fields into the user's profile.

    Idempotent: a null/missing field in ``structured`` never overwrites an
    existing stored value. Returns the post-merge profile dict (or None if
    ``structured`` had no usable fields and no row existed).
    """
    if not user_id or not isinstance(structured, dict):
        return get_user_profile(user_id)

    # Build a payload containing only fields that have a non-null coerced value.
    payload = {}
    for field in _PROFILE_FIELDS:
        if field in structured:
            coerced = _coerce_profile_field(field, structured[field])
            if coerced is not None:
                payload[field] = coerced

    if not payload:
        # Nothing to merge; return current state unchanged.
        return get_user_profile(user_id)

    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        row = db.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        ).scalars().first()
        if row is None:
            row = UserProfile(user_id=user_id)
            db.add(row)
        for field, value in payload.items():
            setattr(row, field, value)
            # P6c — record which profile field was merged (per-field counter).
            try:
                from memory_metrics import inc_profile_merge
                inc_profile_merge(field)
            except Exception:
                pass
        if session_created:
            db.commit()
    finally:
        if session_created:
            db.close()
    return get_user_profile(user_id)


def get_doc_chunks(doc_id: str) -> list[DocumentChunk]:
    db = _new_db_session()
    try:
        return db.execute(
            select(DocumentChunk).where(DocumentChunk.doc_id == doc_id)
        ).scalars().all()
    finally:
        db.close()


def save_doc_chunk(doc_id: str, chunk_id: str, chunk_hash: str, db: Session = None):
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        existing = db.execute(
            select(DocumentChunk).where(
                DocumentChunk.doc_id == doc_id,
                DocumentChunk.chunk_id == chunk_id
            )
        ).scalars().first()
        if existing:
            existing.chunk_hash = chunk_hash
        else:
            new_chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                chunk_hash=chunk_hash
            )
            db.add(new_chunk)
        if session_created:
            db.commit()
    finally:
        if session_created:
            db.close()


def delete_doc_chunks_by_ids(chunk_ids: list[str], db: Session = None):
    if not chunk_ids:
        return
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        db.execute(
            delete(DocumentChunk).where(DocumentChunk.chunk_id.in_(chunk_ids))
        )
        if session_created:
            db.commit()
    except Exception as e:
        logger.error(f"Failed to delete document chunks from database: {e}")
        if session_created:
            db.rollback()
        raise e
    finally:
        if session_created:
            db.close()


def clear_doc_chunks_by_doc(doc_id: str, db: Session = None):
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        db.execute(
            delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id)
        )
        if session_created:
            db.commit()
    except Exception as e:
        logger.error(f"Failed to clear document chunks: {e}")
        if session_created:
            db.rollback()
        raise e
    finally:
        if session_created:
            db.close()


def ensure_database_schema():
    """Create missing tables for the current database if needed."""
    try:
        Base.metadata.create_all(bind=engine)
        # Check if 'sources' column exists in 'chat_conversations', if not add it
        from sqlalchemy import text
        db = _new_db_session()
        try:
            db.execute(text("SELECT sources FROM chat_conversations LIMIT 1"))
        except Exception:
            db.rollback()
            logger.info("Adding 'sources' column to 'chat_conversations' table...")
            try:
                db.execute(text("ALTER TABLE chat_conversations ADD COLUMN sources TEXT"))
                db.commit()
                logger.info("Successfully added 'sources' column.")
            except Exception as alter_err:
                db.rollback()
                logger.error(f"Failed to add 'sources' column: {alter_err}")
        finally:
            db.close()

        # Phase 1 upgrade — structured legal metadata on document_chunks.
        # create_all does not ALTER existing tables, so add each new column
        # idempotently: probe with a SELECT, ALTER on missing-column error.
        _new_chunk_columns = [
            ("law_name", "VARCHAR(300)"),
            ("article_number", "INTEGER"),
            ("clause_number", "INTEGER"),
            ("point_letter", "VARCHAR(10)"),
            ("document_number", "INTEGER"),
            ("document_year", "INTEGER"),
            ("document_type", "VARCHAR(20)"),
            ("effectivity_status", "VARCHAR(20)"),
        ]
        db = _new_db_session()
        try:
            for col_name, col_type in _new_chunk_columns:
                try:
                    db.execute(
                        text(f"SELECT {col_name} FROM document_chunks LIMIT 1")
                    )
                except Exception:
                    db.rollback()
                    logger.info(
                        "Adding '%s' column to 'document_chunks' table...",
                        col_name,
                    )
                    try:
                        db.execute(
                            text(
                                f"ALTER TABLE document_chunks ADD COLUMN {col_name} {col_type}"
                            )
                        )
                        db.commit()
                        logger.info(
                            "Successfully added '%s' column to document_chunks.",
                            col_name,
                        )
                    except Exception as alter_err:
                        db.rollback()
                        logger.error(
                            "Failed to add '%s' column to document_chunks: %s",
                            col_name,
                            alter_err,
                        )
        finally:
            db.close()
    except Exception as e:
        logger.warning("Database not ready during schema initialization: %s", e)


# ---- Phase 2 — User repository ----
import uuid as _uuid


def create_user(username: str, password_hash: str, role: str = "user",
                email: str | None = None, db: Session = None) -> User:
    """Insert a new user. Raises on duplicate username (caller handles)."""
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        user = User(
            id=_uuid.uuid4().hex,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
        )
        db.add(user)
        if session_created:
            db.commit()
            db.refresh(user)
        return user
    finally:
        if session_created:
            db.close()


def get_user_by_username(username: str, db: Session = None) -> User | None:
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        return db.execute(
            select(User).where(User.username == username)
        ).scalars().first()
    finally:
        if session_created:
            db.close()


def get_user_by_id(user_id: str, db: Session = None) -> User | None:
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        return db.get(User, user_id)
    finally:
        if session_created:
            db.close()


def list_users(limit: int = 100, offset: int = 0, db: Session = None) -> list[User]:
    session_created = False
    if db is None:
        db = _new_db_session()
        session_created = True
    try:
        return db.execute(
            select(User).order_by(User.created_at.desc()).offset(offset).limit(limit)
        ).scalars().all()
    finally:
        if session_created:
            db.close()


# insert document into database
def insert_document(question: str, content: str):
    ensure_database_schema()
    db = _new_db_session()
    # Step 1: Create a new Document instance
    new_doc = Document(
        question=question,
        content=content,
    )
    # Step 2: Save the Document instance
    try:
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
    finally:
        db.close()

    logger.info(f"Create document successfully {new_doc}")

    return new_doc


def list_documents(limit: int = 100, offset: int = 0):
    ensure_database_schema()
    db = _new_db_session()
    try:
        documents = db.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).scalars().all()
        return [
            {
                "id": document.id,
                "question": document.question,
                "content": document.content,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
            }
            for document in documents
        ]
    finally:
        db.close()


# Run once on import after models are defined.
ensure_database_schema()