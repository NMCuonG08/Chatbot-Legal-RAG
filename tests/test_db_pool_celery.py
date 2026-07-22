"""Audit (external 360°) — 6.1 SQL QueuePool, 8.1 Celery worker recycling,
6.2 ChatConversation indexes.

6.1: ``create_engine`` previously set only ``pool_pre_ping=True``, leaving
SQLAlchemy's QueuePool at defaults (pool_size=5, max_overflow=10, no recycle).
Under Celery + API concurrency that overflows and stales connections. Fix:
explicit pool_size=20, max_overflow=40, pool_recycle=1800, pool_pre_ping=True.

8.1: Celery workers run the bge-m3 SentenceTransformer (PyTorch) in-process.
PyTorch's caching allocator + per-task allocations grow resident memory without
bound; with no recycling a worker eventually OOMs. Fix: set
``worker_max_tasks_per_child`` + ``worker_max_memory_per_child`` so a worker
recycles itself after 100 tasks or ~4GB.

6.2: ``ChatConversation`` had NO index on ``conversation_id`` (the hot
``load_conversation`` lookup) or ``user_id`` (list/delete paths) — every load
was a table scan. The audit's proposed composite ``(user_id, conversation_id)``
does NOT serve the conversation_id-only hot path (leftmost-prefix rule), so the
correct fix is separate indexes on both columns.
"""
import pytest

pytestmark = pytest.mark.unit

import database
from database import get_celery_app
from models import ChatConversation


# ---- 6.1: SQLAlchemy QueuePool config -------------------------------------
def test_engine_pool_sized_for_concurrency():
    pool = database.engine.pool
    assert pool.size() == 20, (
        f"pool_size must be 20 (was {pool.size()}); defaults of 5 overflow under load"
    )
    assert getattr(pool, "_max_overflow", None) == 40, (
        "max_overflow must be 40 to absorb burst concurrency"
    )
    assert getattr(pool, "_recycle", None) == 1800, (
        "pool_recycle must be 1800s to drop stale MySQL connections"
    )


def test_engine_pool_pre_ping_enabled():
    # pre_ping issues a cheap SELECT 1 before checkout to drop dead connections.
    assert database.engine.pool._pre_ping is True


# ---- 8.1: Celery worker recycling -----------------------------------------
def test_celery_app_recycles_workers_on_task_count():
    app = get_celery_app("test-app")
    assert app.conf.worker_max_tasks_per_child == 100, (
        "PyTorch allocator grows per task — must recycle after 100 tasks"
    )


def test_celery_app_recycles_workers_on_memory():
    app = get_celery_app("test-app")
    # 4_000_000 KiB == ~4GB resident ceiling per worker.
    assert app.conf.worker_max_memory_per_child == 4_000_000, (
        "must recycle worker before unbounded PyTorch RAM growth OOMs the box"
    )


# ---- 6.2: ChatConversation indexes ----------------------------------------
def test_chat_conversation_has_conversation_id_index():
    """load_conversation filters by conversation_id alone — the hot path. That
    needs its own index; a (user_id, conversation_id) composite would NOT serve
    it (leftmost-prefix)."""
    cols = {idx.columns[0].name for idx in ChatConversation.__table__.indexes
            if len(idx.columns) == 1}
    assert "conversation_id" in cols, (
        "conversation_id needs a standalone index (hot load_conversation lookup)"
    )


def test_chat_conversation_has_user_id_index():
    """list/delete paths filter by user_id alone."""
    cols = {idx.columns[0].name for idx in ChatConversation.__table__.indexes
            if len(idx.columns) == 1}
    assert "user_id" in cols, "user_id needs a standalone index (list/delete paths)"