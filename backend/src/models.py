import asyncio
import logging
from xml.dom import ValidationErr

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from cache import get_conversation_id
from database import engine, get_db
from utils import setup_logging

Base = declarative_base()


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
        return (
            db.query(ChatConversation)
            .filter(ChatConversation.conversation_id == conversation_id)
            .order_by(ChatConversation.created_at)
            .all()
        )
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
        conversations = (
            db.query(ChatConversation)
            .filter(ChatConversation.user_id == user_id)
            .order_by(ChatConversation.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
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
        db.query(ChatConversation).filter(ChatConversation.user_id == user_id).delete()
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting user conversations: {e}")
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
        documents = (
            db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
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