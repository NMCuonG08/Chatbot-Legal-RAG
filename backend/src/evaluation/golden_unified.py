"""Unify the three parallel golden sets into one deduped dataset.

Sources:
- ``run_question_test.QUESTION_SET`` (16, with expected_tool) — backend/src.
- ``eval_router.QUESTION_SET`` (16, route-only) — backend/src. Overlaps the
  first; merged by question text so tools come from run_question_test.
- ``tests/eval_prompts.json`` (14 categories) — quality prompt set.

``unify_golden`` returns a deduped list of ``GoldenItem``. ``write_unified_dataset``
persists JSONL; ``load_unified_dataset`` reads it back; ``to_eval_sample``
adapts a GoldenItem to ``EvalSample`` so the existing eval pipeline consumes it.

Public surface:
- ``GoldenItem`` frozen.
- ``unify_golden``, ``write_unified_dataset``, ``load_unified_dataset``,
  ``to_eval_sample``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GOLDEN_FILE = REPO_ROOT / "data" / "golden_unified.jsonl"
_EVAL_PROMPTS = REPO_ROOT / "tests" / "eval_prompts.json"


@dataclass(frozen=True)
class GoldenItem:
    sample_id: str
    question: str
    expected_route: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_block: bool = False
    category: str = ""
    difficulty: str = "medium"  # easy|medium|hard
    language: str = "vi"
    source: str = ""


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _from_run_question_test() -> List[GoldenItem]:
    items: List[GoldenItem] = []
    try:
        import run_question_test as rqt
    except Exception as exc:
        logger.warning("run_question_test import failed: %s", exc)
        return items
    for qid, q, route, tool in rqt.QUESTION_SET:
        items.append(GoldenItem(
            sample_id=qid, question=q, expected_route=route,
            expected_tool=tool or None, category="acceptance",
            difficulty="medium", source="run_question_test"))
    return items


def _from_eval_router() -> List[GoldenItem]:
    items: List[GoldenItem] = []
    try:
        import eval_router as er
    except Exception as exc:
        logger.warning("eval_router import failed: %s", exc)
        return items
    for qid, q, route in er.QUESTION_SET:
        items.append(GoldenItem(
            sample_id=qid, question=q, expected_route=route,
            category="router", difficulty="easy", source="eval_router"))
    return items


def _from_eval_prompts() -> List[GoldenItem]:
    items: List[GoldenItem] = []
    path = _EVAL_PROMPTS
    if not path.exists():
        logger.warning("eval_prompts.json not found: %s", path)
        return items
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    for cat in data.get("categories", []):
        cat_id = cat.get("id", "")
        for p in cat.get("prompts", []):
            pid = p.get("id", "")
            expect = p.get("expect", "")
            route = None
            for r in ("legal_rag", "agent_tools", "web_search", "general_chat"):
                if r in expect.lower():
                    route = r
                    break
            items.append(GoldenItem(
                sample_id=f"{cat_id}-{pid}",
                question=p.get("q", ""),
                expected_route=route,
                category=cat_id,
                difficulty="medium",
                source="eval_prompts",
            ))
    return items


def unify_golden() -> List[GoldenItem]:
    """Merge the three sources, deduping by normalized question text.

    run_question_test wins over eval_router (it carries expected_tool); the
    eval_prompts source adds category coverage the acceptance set lacks.
    """
    merged: dict = {}
    for item in _from_run_question_test():
        merged[_norm(item.question)] = item
    for item in _from_eval_router():
        key = _norm(item.question)
        if key not in merged:
            merged[key] = item
    for item in _from_eval_prompts():
        key = _norm(item.question)
        if key not in merged:
            merged[key] = item
    return list(merged.values())


def write_unified_dataset(path: Path | str = DEFAULT_GOLDEN_FILE) -> Path:
    """Write the unified golden set to JSONL. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = unify_golden()
    with path.open("w", encoding="utf-8") as fp:
        for it in items:
            fp.write(json.dumps({
                "sample_id": it.sample_id,
                "question": it.question,
                "expected_route": it.expected_route,
                "expected_tool": it.expected_tool,
                "expected_answer": it.expected_answer,
                "expected_block": it.expected_block,
                "category": it.category,
                "difficulty": it.difficulty,
                "language": it.language,
                "source": it.source,
            }, ensure_ascii=False) + "\n")
    logger.info("Wrote %d golden items -> %s", len(items), path)
    return path


def load_unified_dataset(path: Path | str = DEFAULT_GOLDEN_FILE) -> List[GoldenItem]:
    path = Path(path)
    if not path.exists():
        return []
    items: List[GoldenItem] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(GoldenItem(
                sample_id=row["sample_id"],
                question=row["question"],
                expected_route=row.get("expected_route"),
                expected_tool=row.get("expected_tool"),
                expected_answer=row.get("expected_answer"),
                expected_block=bool(row.get("expected_block", False)),
                category=row.get("category", ""),
                difficulty=row.get("difficulty", "medium"),
                language=row.get("language", "vi"),
                source=row.get("source", ""),
            ))
    return items


def to_eval_sample(item: GoldenItem) -> "EvalSample":
    from evaluation.dataset import EvalSample
    return EvalSample(
        sample_id=item.sample_id,
        question=item.question,
        gold_context=item.expected_answer or "",
        expected_route=item.expected_route,
        expected_answer=item.expected_answer,
        expected_tool=item.expected_tool,
        expected_block=item.expected_block,
    )


__all__ = [
    "GoldenItem",
    "unify_golden",
    "write_unified_dataset",
    "load_unified_dataset",
    "to_eval_sample",
]