"""P7 — One-shot backfill: build structured UserProfiles from existing episodes.

For each user_id that has UserEpisode rows but no UserProfile, concatenate the
episode summaries, run the ``episodic_extract`` prompt, and merge the parsed
``structured`` block into ``user_profiles`` via ``merge_user_profile`` (idempotent,
null-safe). Re-runnable: a user with an existing profile is left to the merge
semantics (non-null fields only) and not forced.

Run from repo root:
    python -m backend.scripts.backfill_user_profiles          # dry-run preview
    python -m backend.scripts.backfill_user_profiles --apply   # actually merge

Env: requires the normal backend env (LLM key, DB url). Idempotent + safe to
interrupt; per-user commits. Does NOT touch Qdrant or UserEpisode rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make backend/src importable when run as a script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv
load_dotenv(_SRC.parent / ".env", override=True)

from models import (  # noqa: E402
    SessionLocal, UserEpisode, UserProfile, get_user_episodes, merge_user_profile,
)
from sqlalchemy import select  # noqa: E402
from brain import openai_chat_complete  # noqa: E402
from prompt_loader import load_prompt  # noqa: E402
from tasks import strip_legal_citations, _parse_episodic_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_profiles")


def _distinct_user_ids_with_episodes() -> list[str]:
    db = SessionLocal()
    try:
        rows = db.execute(select(UserEpisode.user_id).distinct()).scalars().all()
        return [r for r in rows if r]
    finally:
        db.close()


def _user_ids_without_profile(user_ids: list[str]) -> list[str]:
    if not user_ids:
        return []
    db = SessionLocal()
    try:
        existing = set(db.execute(
            select(UserProfile.user_id).where(UserProfile.user_id.in_(user_ids))
        ).scalars().all())
        return [u for u in user_ids if u not in existing]
    finally:
        db.close()


def _extract_structured(user_id: str) -> dict | None:
    """Concatenate this user's episodes -> extract prompt -> structured dict."""
    episodes = get_user_episodes(user_id, limit=50)
    if not episodes:
        return None
    blob = "\n".join(str(getattr(ep, "summary", "") or "") for ep in episodes)
    blob = strip_legal_citations(blob)
    if not blob.strip():
        return None
    prompt = load_prompt("episodic_extract", user_message=blob)
    raw = openai_chat_complete([
        {"role": "system", "content": "Bạn là chuyên gia trích xuất thông tin cá nhân từ hội thoại pháp luật."},
        {"role": "user", "content": prompt},
    ]).strip()
    parsed = _parse_episodic_json(raw)
    if not parsed:
        return None
    return parsed.get("structured") or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill user_profiles from user_episodes.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually merge (default: dry-run, no writes).")
    parser.add_argument("--user", help="Backfill a single user_id only.")
    args = parser.parse_args()

    user_ids = [args.user] if args.user else _distinct_user_ids_with_episodes()
    if not args.user:
        user_ids = _user_ids_without_profile(user_ids)
    logger.info(f"Users to backfill: {len(user_ids)} {'(APPLY)' if args.apply else '(DRY-RUN)'}")

    merged = 0
    for uid in user_ids:
        try:
            structured = _extract_structured(uid)
        except Exception as exc:
            logger.warning(f"[{uid}] extract failed: {exc}")
            continue
        if not structured:
            logger.info(f"[{uid}] no structured facts, skipping")
            continue
        if args.apply:
            profile = merge_user_profile(uid, structured)
            logger.info(f"[{uid}] merged -> {json.dumps(profile, ensure_ascii=False, default=str)}")
        else:
            logger.info(f"[{uid}] would merge -> {json.dumps(structured, ensure_ascii=False)}")
        merged += 1

    logger.info(f"Done. {merged} users processed ({'applied' if args.apply else 'dry-run'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())