"""Migration 001 — add hot-path indexes to chat_conversations (audit 6.2).

``ChatConversation`` historically had only the pkey index. ``load_conversation``
filters by ``conversation_id`` alone (the per-message hot path) and list/delete
paths filter by ``user_id`` alone — both were table scans. The model now
declares standalone indexes on both columns (``models.py``); this one-shot
applies them to an EXISTING database (``Base.metadata.create_all`` only creates
missing tables, not indexes on existing tables, so a script is needed).

Idempotent: ``CREATE INDEX IF NOT EXISTS``. Safe to re-run. Uses plain
``CREATE INDEX`` (not ``CONCURRENTLY``) so it works inside the script's
transaction and supports ``IF NOT EXISTS``; on a small chat-conversation table
the brief write lock is acceptable. For a large table, run the equivalent
``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` by hand in psql instead.

Run from repo root:
    python -m backend.scripts.add_chat_conv_indexes          # dry-run preview
    python -m backend.scripts.add_chat_conv_indexes --apply  # create indexes

Env: requires the normal backend env (DATABASE_URL). Read-only until --apply.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make backend/src importable when run as a script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_SRC.parent / ".env", override=True)

from sqlalchemy import text  # noqa: E402
from database import engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migration_001")

# Mirror models.ChatConversation.__table_args__ exactly.
INDEXES = [
    ("idx_chat_conv_conversation_id", "conversation_id"),
    ("idx_chat_conv_user_id", "user_id"),
]


def _existing_indexes(conn) -> set[str]:
    rows = conn.execute(
        text("SELECT indexname FROM pg_indexes WHERE tablename='chat_conversations'")
    ).all()
    return {r[0] for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Add chat_conversation hot-path indexes")
    parser.add_argument("--apply", action="store_true", help="actually run DDL (default: dry-run)")
    args = parser.parse_args()

    with engine.begin() as conn:
        existing = _existing_indexes(conn)
        logger.info("existing chat_conversations indexes: %s", sorted(existing) or "(pkey only)")
        plan = [(name, col) for name, col in INDEXES if name not in existing]
        if not plan:
            logger.info("nothing to do — all indexes already present")
            return 0
        for name, col in plan:
            ddl = (
                f"CREATE INDEX IF NOT EXISTS {name} "
                f"ON chat_conversations ({col})"
            )
            logger.info("%s: %s", "APPLY" if args.apply else "DRY-RUN", ddl)
            if args.apply:
                conn.execute(text(ddl))
        if args.apply:
            after = _existing_indexes(conn)
            logger.info("indexes after: %s", sorted(after))
        else:
            logger.info("dry-run only — re-run with --apply to create")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())