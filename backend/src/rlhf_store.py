"""RLHF user-feedback store (Phase 4) — 👍/👎 becomes continual learning signal.

Mirrors the episodic-memory pattern (``save_episodic_memory_task``):
- MySQL ``agent_feedback`` table is the durable audit trail for ALL feedback
  (good + bad), auto-created by ``ensure_database_schema``.
- Qdrant ``rlhf_good_answers`` collection holds the GOOD answers, keyed by the
  question embedding, user-scoped via ``_scope_for`` (never leak a user's 👍
  across users). Used downstream for few-shot injection + rerank up-weighting.

Privacy (carried forward from the semantic-cache + episodic fixes):
- Shared/sentinel user_ids (``anonymous``/``demo-session``/``""``) are rejected
  — feedback requires a real user scope.
- Deterministic uuid5 point id ``f"{user_id}|good|{question}"`` makes the
  upsert idempotent (re-submitting the same 👍 overwrites, not duplicates).
- Dedup: skip the Qdrant write when a near-identical good answer already exists
  for this user (score >= 0.92), to avoid flooding the few-shot pool.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client.models import Distance, VectorParams

from brain import get_embedding
from database import SessionLocal
from models import AgentFeedback
from semantic_cache import _scope_for
from vectorize import add_vector, get_client, search_vector

logger = logging.getLogger(__name__)

RLHF_COLLECTION_NAME = "rlhf_good_answers"
SCOPE_USER_PREFIX = "user:"   # mirrors semantic_cache.SCOPE_USER_PREFIX

# Dedup threshold — a new 👍 whose question is >= this similar to an existing
# good answer for the same user is skipped (avoid few-shot pool flooding).
_DEDUP_THRESHOLD = 0.92
# Few-shot injection threshold — a good answer is injected only when the
# current question is >= this similar (plan default 0.85).
DEFAULT_FEW_SHOT_THRESHOLD = 0.85


def _is_real_user(user_id: Optional[str]) -> bool:
    """Reject sentinel/shared user_ids — feedback needs a real user scope."""
    # Lazy import avoids a tasks <-> rlhf_store top-level import cycle.
    from tasks import _SHARED_USER_SENTINELS
    return (user_id or "").strip() not in _SHARED_USER_SENTINELS


def init_rlhf_collection() -> None:
    """Ensure the ``rlhf_good_answers`` Qdrant collection exists."""
    try:
        collections = [c.name for c in get_client().get_collections().collections]
        if RLHF_COLLECTION_NAME not in collections:
            logger.info("Creating Qdrant rlhf_good_answers collection")
            get_client().create_collection(
                collection_name=RLHF_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.DOT),
            )
            logger.info("✅ rlhf_good_answers collection created")
    except Exception as exc:
        logger.error("Failed to initialize rlhf_good_answers collection: %s", exc)


def save_feedback(
    user_id: str,
    conversation_id: str,
    message_id: str,
    question: str,
    response: str,
    sources: List[Dict[str, Any]],
    rating: str,
) -> str:
    """Persist user feedback (MySQL always; Qdrant for good answers only).

    Args:
        user_id: real user id (sentinels rejected).
        conversation_id, message_id: locators for the rated message.
        question: the user question that produced the response.
        response: the assistant response being rated.
        sources: the RAG source chunks backing the response (for rerank up-weight).
        rating: ``"good"`` (👍) or ``"bad"`` (👎).

    Returns:
        One of ``"saved"`` / ``"rejected_sentinel"`` / ``"skipped_duplicate"``.
        Never raises — failures are logged and degraded to MySQL-only.
    """
    if not _is_real_user(user_id):
        logger.info("RLHF feedback rejected: sentinel/empty user_id.")
        return "rejected_sentinel"
    rating = (rating or "").strip().lower()
    if rating not in ("good", "bad"):
        logger.info("RLHF feedback rejected: invalid rating %r.", rating)
        return "rejected_sentinel"

    # 1. MySQL audit trail — always, for both good + bad.
    try:
        db = SessionLocal()
        try:
            db.add(AgentFeedback(
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=message_id,
                question=question,
                response=response,
                sources_json=sources,
                rating=rating,
            ))
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning("RLHF MySQL write failed (non-blocking): %s", exc)

    # 2. Qdrant good-answers pool — only for 👍, used by few-shot + rerank.
    if rating != "good" or not question or not response:
        return "saved"

    try:
        vector = get_embedding(question)
    except Exception as exc:
        logger.warning("RLHF embedding failed (Qdrant write skipped): %s", exc)
        return "saved"

    scope = _scope_for(user_id, None)

    # Dedup: skip if a near-identical good answer already exists for this user.
    try:
        existing = search_vector(
            collection_name=RLHF_COLLECTION_NAME,
            vector=vector,
            limit=1,
            filters={"scope": scope},
            score_threshold=_DEDUP_THRESHOLD,
        )
        if existing:
            logger.info("RLHF dedup: similar good answer already stored for scope %s.", scope)
            return "skipped_duplicate"
    except Exception as exc:
        logger.warning("RLHF dedup check failed (proceeding to save): %s", exc)

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}|good|{question}"))
    payload = {
        "vector": vector,
        "payload": {
            "user_id": user_id,
            "scope": scope,
            "question": question,
            "response": response,
            "sources": sources,
            "rating": rating,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "content": question,   # compatibility with search_vector payload
        },
    }
    try:
        add_vector(collection_name=RLHF_COLLECTION_NAME, vectors={point_id: payload})
    except Exception as exc:
        logger.warning("RLHF Qdrant write failed (MySQL audit still saved): %s", exc)
    return "saved"


def find_similar_good(
    user_id: Optional[str],
    question: str,
    score_threshold: float = DEFAULT_FEW_SHOT_THRESHOLD,
    limit: int = 1,
) -> Optional[Dict[str, Any]]:
    """Find this user's prior 👍-marked answer semantically near ``question``.

    Args:
        user_id: real user id (sentinels -> None, no cross-user leak).
        question: the current question.
        score_threshold: minimum similarity to surface a good answer.
        limit: max good answers to return.

    Returns:
        The top good-answer dict (``{question, response, sources, score}``) or
        ``None`` when the user is a sentinel, the collection is empty, or no
        match meets the threshold. Never raises.
    """
    if not _is_real_user(user_id) or not question:
        return None
    try:
        vector = get_embedding(question)
    except Exception as exc:
        logger.warning("RLHF find embedding failed: %s", exc)
        return None
    scope = _scope_for(user_id, None)
    try:
        results = search_vector(
            collection_name=RLHF_COLLECTION_NAME,
            vector=vector,
            limit=limit,
            filters={"scope": scope},
            score_threshold=score_threshold,
        )
    except Exception as exc:
        logger.warning("RLHF find search failed: %s", exc)
        return None
    if not results:
        return None
    top = results[0]
    payload = top.get("payload", top)
    return {
        "question": payload.get("question", ""),
        "response": payload.get("response", ""),
        "sources": payload.get("sources", []),
        "score": top.get("score", 0.0),
    }