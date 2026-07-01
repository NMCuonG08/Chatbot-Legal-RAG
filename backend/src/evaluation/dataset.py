"""Load and sample evaluation dataset from train.jsonl.

The training file is a JSONL with shape::

    {"question": "...", "context": "..."}

For evaluation we treat each row as a single ground-truth pair where the
``context`` is the gold passage that should be retrieved when querying with
``question``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# Repo root: backend/src/evaluation/dataset.py -> repo
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "train.jsonl"


@dataclass
class EvalSample:
    """A single evaluation example with ground truth context."""

    sample_id: str
    question: str
    gold_context: str
    # Optional human-provided expected routing label (legal_rag / agent_tools /
    # web_search / general_chat). When absent, eval_e2e falls back to the
    # keyword heuristic in ``get_expected_route``. Annotating train.jsonl rows
    # with an ``expected_route`` field lets routing_accuracy measure against
    # gold labels instead of a noisy proxy.
    expected_route: Optional[str] = None

    @property
    def gold_hash(self) -> int:
        """Stable hash of the gold context for set comparisons."""
        return int(
            hashlib.sha1(self.gold_context.encode("utf-8")).hexdigest()[:16],
            16,
        )


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line: %s", exc)


def load_eval_dataset(
    path: Path | str = DEFAULT_DATA_FILE,
    n_samples: int = 100,
    seed: int = 42,
    min_question_chars: int = 10,
    min_context_chars: int = 50,
) -> List[EvalSample]:
    """Load and sample N evaluation pairs deterministically.

    Parameters
    ----------
    path
        Path to train.jsonl
    n_samples
        Number of examples to sample. Use ``-1`` to load all.
    seed
        Random seed for reproducibility.
    min_question_chars / min_context_chars
        Drop pairs that are too short to give meaningful eval signal.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    logger.info("Loading eval dataset from %s", path)

    raw: List[EvalSample] = []
    for idx, row in enumerate(_iter_jsonl(path)):
        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        if len(question) < min_question_chars or len(context) < min_context_chars:
            continue
        # Optional human routing label; None when the row has no annotation.
        expected_route = (row.get("expected_route") or None)
        raw.append(
            EvalSample(
                sample_id=f"train-{idx}",
                question=question,
                gold_context=context,
                expected_route=expected_route,
            )
        )

    if not raw:
        raise ValueError(f"No valid samples found in {path}")

    logger.info("Loaded %d valid pairs", len(raw))

    if n_samples == -1 or n_samples >= len(raw):
        return raw

    rng = random.Random(seed)
    sampled = rng.sample(raw, n_samples)
    logger.info("Sampled %d eval examples (seed=%d)", len(sampled), seed)
    return sampled


def gold_in_retrieved(sample: EvalSample, retrieved_contents: List[str]) -> int:
    """Return rank (1-indexed) of gold context in retrieved list, or 0 if absent.

    Matching uses two signals:
    1. Exact hash match (gold context is identical to a retrieved chunk).
    2. Substring containment in either direction. The ingestion pipeline splits
       documents into chunks, so the retrieved chunk may be a strict substring
       of the gold context (or vice versa). Either direction is treated as a
       successful retrieval for evaluation purposes.
    """
    gold = sample.gold_context.strip()
    gold_lower = gold.lower()
    # Length-ratio guard: a substring match only counts as a hit when the two
    # texts are of comparable length. Prevents a long retrieved chunk that
    # merely CONTAINS the short gold string (e.g. a statute list) from counting
    # as a correct retrieval when it is topically unrelated.
    LENGTH_RATIO_THRESHOLD = 0.5
    for rank, retrieved in enumerate(retrieved_contents, start=1):
        if not retrieved:
            continue
        retrieved_stripped = retrieved.strip()
        if not retrieved_stripped:
            continue
        if retrieved_stripped == gold:
            return rank
        retrieved_lower = retrieved_stripped.lower()
        # Either direction of substring match counts as a hit, but only when the
        # lengths are comparable (ratio >= threshold) to avoid false positives.
        if retrieved_lower in gold_lower or gold_lower in retrieved_lower:
            min_len = min(len(gold_lower), len(retrieved_lower))
            max_len = max(len(gold_lower), len(retrieved_lower))
            if max_len == 0 or (min_len / max_len) >= LENGTH_RATIO_THRESHOLD:
                return rank
    return 0
