"""Test-set slicing + per-slice summary.

Slices partition the eval sample set by intent/route, difficulty, language, and
out-of-scope membership so regressions can be attributed to a subpopulation
(e.g. "legal_rag slice regressed but oos held").

Public surface:
- ``SliceSpec`` frozen.
- ``slice_by_intent``, ``slice_by_difficulty``, ``slice_by_language``,
  ``slice_by_oos``, ``apply_slices``, ``summarize_by_slice``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from evaluation.dataset import EvalSample


@dataclass(frozen=True)
class SliceSpec:
    name: str
    predicate: Callable[[EvalSample], bool]


_VN_DIACRITIC = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗ"
    r"ơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)


def detect_language(text: str) -> str:
    """Return 'vi' if the text carries Vietnamese diacritics, else 'en'."""
    if not text:
        return "vi"
    if _VN_DIACRITIC.search(text):
        return "vi"
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    return "en" if ascii_letters > 0 else "vi"


def _difficulty(question: str) -> str:
    q = question or ""
    has_ref = bool(re.search(r"điều\s+\d+|nghị định|bộ luật", q, re.IGNORECASE))
    has_numbers = bool(re.search(r"\d", q))
    if len(q) > 120 and (has_ref or has_numbers):
        return "hard"
    if has_ref or has_numbers or len(q) > 60:
        return "medium"
    return "easy"


def slice_by_intent(samples: Sequence[EvalSample]) -> Dict[str, List[EvalSample]]:
    """Group by ``expected_route`` (None -> 'unknown')."""
    out: Dict[str, List[EvalSample]] = {}
    for s in samples:
        key = s.expected_route or "unknown"
        out.setdefault(key, []).append(s)
    return out


def slice_by_difficulty(samples: Sequence[EvalSample]) -> Dict[str, List[EvalSample]]:
    out: Dict[str, List[EvalSample]] = {"easy": [], "medium": [], "hard": []}
    for s in samples:
        out[_difficulty(s.question)].append(s)
    return out


def slice_by_language(samples: Sequence[EvalSample]) -> Dict[str, List[EvalSample]]:
    out: Dict[str, List[EvalSample]] = {"vi": [], "en": []}
    for s in samples:
        out[detect_language(s.question)].append(s)
    return out


def slice_by_oos(samples: Sequence[EvalSample]) -> Dict[str, List[EvalSample]]:
    """OOS = expected_route is None (no legal routing expectation)."""
    out: Dict[str, List[EvalSample]] = {"oos": [], "in_scope": []}
    for s in samples:
        out["oos" if s.expected_route is None else "in_scope"].append(s)
    return out


def apply_slices(samples: Sequence[EvalSample],
                 specs: Sequence[SliceSpec]) -> Dict[str, List[EvalSample]]:
    """Apply named slice specs to the sample set. Returns name -> samples."""
    out: Dict[str, List[EvalSample]] = {}
    for spec in specs:
        out[spec.name] = [s for s in samples if spec.predicate(s)]
    return out


def summarize_by_slice(
    slices: Dict[str, List[EvalSample]],
    metric_fn: Callable[[Sequence[EvalSample]], float],
) -> Dict[str, dict]:
    """Apply ``metric_fn`` (e.g. success_rate) to each slice.

    Returns ``{slice_name: {"n": int, "metric": float}}``.
    """
    summary: Dict[str, dict] = {}
    for name, items in slices.items():
        summary[name] = {"n": len(items), "metric": metric_fn(items)}
    return summary


__all__ = [
    "SliceSpec",
    "slice_by_intent",
    "slice_by_difficulty",
    "slice_by_language",
    "slice_by_oos",
    "apply_slices",
    "summarize_by_slice",
    "detect_language",
]