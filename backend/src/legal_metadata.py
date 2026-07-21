"""Extract structured legal metadata from free Vietnamese legal text.

Used at ingest time to enrich Qdrant payload so retrieval can do exact-field
filtering instead of purely semantic search. Vietnamese legal references follow
a small set of patterns:

- "Điều 418", "Điều 107"             -> article_number
- "khoản 1", "khoản 2"               -> clause_number
- "điểm a", "điểm b khoản 1"         -> point_letter
- "Bộ luật Dân sự 2015"             -> law_name + document_type=code + document_year
- "Luật Đất đai 2024"               -> law_name + document_type=act + document_year
- "Nghị định 10/2022", "Thông tư 80/2020",
  "Nghị quyết 326/2016/UBTVQH14"    -> law_name + document_type + document_number + document_year

The extractor is intentionally conservative: it returns the *first* match of
each kind, or omits the key when no pattern is recognized. Unknown text yields
an empty dict (no ``None`` fields written into Qdrant — keeps payloads clean).

Effectivity status (``in_force`` / ``not_yet_effective`` / ``repealed`` /
``amended``) is *not* derived here — it needs the corpus version table. See
``legal_effectivity.classify_effectivity`` which is called by the ingest path
after this extractor has produced ``law_name`` / ``document_year``.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Normalized document types. Stored as a single-string enum on Qdrant payload
# (and SQL column) so retrieval can filter ``content_type == "decree"`` instead
# of relying on substring heuristics.
DOCUMENT_TYPES = ("code", "act", "decree", "circular", "resolution",
                 "contract", "case_law", "general")

# Match "Điều <digits>" — capture the digits. Handles "Điều 418", "điều 418.",
# "Điều 418/2024" (takes 418).
_ARTICLE_RE = re.compile(r"Điều\s+(\d+)", re.IGNORECASE | re.UNICODE)

# Match "khoản <digits>" — capture the digits. E.g. "khoản 1", "Khoản 2".
_CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE | re.UNICODE)

# Match "điểm <letter>" — capture the letter. Vietnamese legal points use
# lowercase latin letters (a, b, c, ...). Accept diacritic letters too for
# robustness, though real citations use plain latin.
_POINT_RE = re.compile(r"điểm\s+([a-zA-Zà-ỹ])", re.IGNORECASE | re.UNICODE)

# Codes / acts: "Bộ luật Dân sự 2015", "Luật Lao động 2019", "Luật Đất đai 2024".
# ``\w`` with re.UNICODE matches Vietnamese diacritic letters. Capture the
# 4-digit year so callers do not have to re-parse it.
_LAW_RE = re.compile(
    r"((?:Bộ luật|Luật)\s+[\wÀ-ỹ\s]+?)(\d{4})",
    re.UNICODE,
)

# Subordinate instruments carry a numeric id: "Nghị định 10/2022",
# "Thông tư 80/2020", "Nghị quyết 326/2016/UBTVQH14". Capture the instrument
# keyword, the document number, the year, and the optional issuing-body suffix
# (``/UBTVQH14``) so they can be stored as discrete filterable fields AND the
# canonical ``law_name`` preserves the full citation (the suffix disambiguates
# resolutions of the Standing Committee).
_DECREE_RE = re.compile(
    r"(Nghị định|Thông tư|Nghị quyết)\s+(\d+)/(\d{4})(/UBTVQH\d+)?",
    re.UNICODE,
)


def _document_type_for(law_name: str) -> Optional[str]:
    """Map a captured ``law_name`` to a normalized ``document_type``."""
    if not law_name:
        return None
    name = law_name.strip()
    if name.startswith("Bộ luật"):
        return "code"
    if name.startswith("Luật"):
        return "act"
    if name.startswith("Nghị định"):
        return "decree"
    if name.startswith("Thông tư"):
        return "circular"
    if name.startswith("Nghị quyết"):
        return "resolution"
    return None


def extract_legal_metadata(text: str) -> Dict[str, Optional[object]]:
    """Extract structured legal metadata from ``text``.

    Returns a dict with only the keys that were recognized (no ``None``
    placeholders), so callers can ``payload.update(...)`` without polluting
    Qdrant with null fields.

    Recognized keys: ``law_name``, ``article_number``, ``clause_number``,
    ``point_letter``, ``document_number``, ``document_year``, ``document_type``.

    Examples:
        >>> extract_legal_metadata("Điều 418 khoản 2 Bộ luật Dân sự 2015 quy định...")
        {'law_name': 'Bộ luật Dân sự 2015', 'article_number': 418,
        ...  'clause_number': 2, 'document_year': 2015, 'document_type': 'code'}
        >>> extract_legal_metadata("Theo Nghị định 10/2022, lệ phí...")
        {'law_name': 'Nghị định 10/2022', 'document_number': 10,
        ...  'document_year': 2022, 'document_type': 'decree'}
        >>> extract_legal_metadata("Xin chào")
        {}
    """
    if not text:
        return {}

    out: Dict[str, Optional[object]] = {}

    art_m = _ARTICLE_RE.search(text)
    if art_m:
        try:
            out["article_number"] = int(art_m.group(1))
        except ValueError:
            pass

    clause_m = _CLAUSE_RE.search(text)
    if clause_m:
        try:
            out["clause_number"] = int(clause_m.group(1))
        except ValueError:
            pass

    point_m = _POINT_RE.search(text)
    if point_m:
        out["point_letter"] = point_m.group(1).lower()

    # Codes/acts take precedence over decrees when both match the same span
    # (e.g. "Bộ luật Dân sự 2015" is a code, not a decree).
    law_m = _LAW_RE.search(text)
    decree_m = _DECREE_RE.search(text)
    if law_m:
        name = f"{law_m.group(1).strip()} {law_m.group(2)}"
        out["law_name"] = name
        try:
            out["document_year"] = int(law_m.group(2))
        except ValueError:
            pass
        dtype = _document_type_for(name)
        if dtype:
            out["document_type"] = dtype
    elif decree_m:
        name = f"{decree_m.group(1)} {decree_m.group(2)}/{decree_m.group(3)}"
        if decree_m.group(4):  # optional issuing-body suffix e.g. "/UBTVQH14"
            name = f"{name}{decree_m.group(4)}"
        out["law_name"] = name
        try:
            out["document_number"] = int(decree_m.group(2))
        except ValueError:
            pass
        try:
            out["document_year"] = int(decree_m.group(3))
        except ValueError:
            pass
        dtype = _document_type_for(name)
        if dtype:
            out["document_type"] = dtype

    return out