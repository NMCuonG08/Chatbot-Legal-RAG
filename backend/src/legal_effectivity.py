"""Per-document effectivity classification for Vietnamese legal corpus.

Companion to ``legal_metadata.extract_legal_metadata``. The metadata parser
produces ``law_name`` / ``document_year`` from raw text; this module resolves
those against the curated version table (``legal_corpus_versions``) to attach a
single ``effectivity_status`` string to each Qdrant payload + SQL row:

- ``in_force``          : effective now, not superseded.
- ``not_yet_effective`` : effective date is in the future.
- ``repealed``          : superseded by a newer statute (``replaced_by`` set).
- ``amended``           : still in force but has been amended in place.

Resolution is conservative: when ``law_name`` is unknown to the version table
the statute is assumed ``in_force`` (the corpus is curated from in-force
sources; unknown-name blocking would drop too much data). Callers that need a
stricter answer should call ``get_law_version`` directly.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from legal_corpus_versions import (
    LAW_VERSIONS,
    find_version_by_key,
    find_version_by_name,
)

logger = logging.getLogger(__name__)

# Public enum — also the set of values written to Qdrant payload + SQL column.
EFFECTIVITY_IN_FORCE = "in_force"
EFFECTIVITY_NOT_YET = "not_yet_effective"
EFFECTIVITY_REPEALED = "repealed"
EFFECTIVITY_AMENDED = "amended"

EFFECTIVITY_STATUSES = (
    EFFECTIVITY_IN_FORCE,
    EFFECTIVITY_NOT_YET,
    EFFECTIVITY_REPEALED,
    EFFECTIVITY_AMENDED,
)


def _parse_date(s: str) -> Optional[date]:
    """Parse a ``DD/MM/YYYY`` string into a ``date``. Returns ``None`` on failure."""
    if not s:
        return None
    parts = s.split("/")
    if len(parts) != 3:
        return None
    try:
        return date(int(parts[2]), int(parts[1]), int(parts[0]))
    except (ValueError, IndexError):
        return None


def classify_effectivity(
    law_name: Optional[str],
    document_year: Optional[int] = None,
    as_of: Optional[date] = None,
) -> str:
    """Classify a statute's effectivity as of ``as_of`` (default: today).

    Args:
        law_name:        canonical Vietnamese statute name (from metadata parser).
        document_year:   fallback 4-digit year if the version table has no
                         ``effective_from`` date for this name.
        as_of:           reference date; defaults to ``date.today()``.

    Returns:
        One of ``EFFECTIVITY_STATUSES``. Unknown statutes default to
        ``in_force`` (see module docstring).
    """
    ref = as_of or date.today()
    entry = find_version_by_name(law_name)

    if not entry:
        # Unknown to the curated table: fall back to a year-only guess when the
        # parser captured one. A year far in the future is "not yet effective";
        # otherwise assume in force (conservative).
        if document_year and document_year > ref.year:
            return EFFECTIVITY_NOT_YET
        return EFFECTIVITY_IN_FORCE

    # Superseded by a newer statute -> repealed (whole statute replaced).
    replaced_by = entry.get("replaced_by")
    if replaced_by:
        successor = find_version_by_key(str(replaced_by))
        if successor:
            succ_from = _parse_date(str(successor.get("effective_from", "")))
            # The successor must itself have taken effect for the old one to be
            # considered repealed today.
            if succ_from is None or succ_from <= ref:
                return EFFECTIVITY_REPEALED

    # Amended in place (non-superseding changes) -> amended (still in force).
    amended_by = entry.get("amended_by") or []
    if amended_by:
        return EFFECTIVITY_AMENDED

    # Neither superseded nor amended: check effective date vs reference.
    eff = _parse_date(str(entry.get("effective_from", "")))
    if eff is not None and eff > ref:
        return EFFECTIVITY_NOT_YET

    return EFFECTIVITY_IN_FORCE


def effectivity_for_payload(
    law_name: Optional[str],
    document_year: Optional[int] = None,
    as_of: Optional[date] = None,
) -> str:
    """Ingest-friendly wrapper: never raises, always returns a valid status.

    On any internal error logs a warning and returns ``in_force`` so a buggy
    version-table entry cannot break the ingest pipeline.
    """
    try:
        return classify_effectivity(law_name, document_year, as_of)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "effectivity classification failed for %r (%s): %s — defaulting to in_force",
            law_name, document_year, exc,
        )
        return EFFECTIVITY_IN_FORCE