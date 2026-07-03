import asyncio
import logging
from xml.dom import ValidationErr

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, delete
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
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
    bot_id: str, user_id: str, message: str, is_request: bool = True
):
    db = _new_db_session()
    # Step 1: Create a new ChatConversation instance
    conversation_id = get_conversation_id(bot_id, user_id)

    new_conversation = ChatConversation(
        conversation_id=conversation_id,
        bot_id=bot_id,
        user_id=user_id,
        message=message,
        is_request=is_request,
        completed=not is_request,
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
            .order_by(ChatConversation.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).scalars().all()
        return [
            {
                "id": conversation.id,
                "conversation_id": conversation.conversation_id,
                "bot_id": conversation.bot_id,
                "user_id": conversation.user_id,
                "message": conversation.message,
                "is_request": conversation.is_request,
                "completed": conversation.completed,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
            }
            for conversation in conversations
        ]
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


class UserEpisode(Base):
    __tablename__ = "user_episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    summary = Column(String)  # Summary of episodic interaction
    created_at = Column(DateTime(timezone=True), server_default=func.now())


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
    except Exception as e:
        logger.warning("Database not ready during schema initialization: %s", e)


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