"""Extract structured legal metadata (law_name, article_number) from free text.

Used at ingest time to enrich Qdrant payload so retrieval can do exact-field
filtering instead of purely semantic search. Vietnamese legal references follow
a small set of patterns:

- "Điều 418", "Điều 107"             -> article_number
- "Bộ luật Dân sự 2015"             -> law_name (codes)
- "Luật Đất đai 2024"               -> law_name (acts)
- "Nghị định 10/2022", "Thông tư 80/2020",
  "Nghị quyết 326/2016"            -> law_name (subordinate instruments)

The extractor is intentionally conservative: it returns the *first* match of
each kind, or omits the key when no pattern is recognized. Unknown text yields
an empty dict (no ``None`` fields written into Qdrant — keeps payloads clean).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Match "Điều <digits>" — capture the digits. Handles "Điều 418", "điều 418.",
# "Điều 418/2024" (takes 418).
_ARTICLE_RE = re.compile(r"Điều\s+(\d+)", re.IGNORECASE | re.UNICODE)

# Codes / acts: "Bộ luật Dân sự 2015", "Luật Lao động 2019", "Luật Đất đai 2024".
# ``\w`` with re.UNICODE matches Vietnamese diacritic letters.
_LAW_RE = re.compile(
    r"(?:Bộ luật|Luật)\s+[\wÀ-ỹ\s]+?\d{4}",
    re.UNICODE,
)

# Subordinate instruments carry a numeric id: "Nghị định 10/2022",
# "Thông tư 80/2020", "Nghị quyết 326/2016/UBTVQH14".
_DECREE_RE = re.compile(
    r"(?:Nghị định|Thông tư|Nghị quyết)\s+\d+/\d{4}(?:/UBTVQH\d+)?",
    re.UNICODE,
)


def extract_legal_metadata(text: str) -> Dict[str, Optional[object]]:
    """Extract ``law_name`` and ``article_number`` from ``text``.

    Returns a dict with only the keys that were recognized (no ``None``
    placeholders), so callers can ``payload.update(...)`` without polluting
    Qdrant with null fields.

    Examples:
        >>> extract_legal_metadata("Điều 418 Bộ luật Dân sự 2015 quy định...")
        {'law_name': 'Bộ luật Dân sự 2015', 'article_number': 418}
        >>> extract_legal_metadata("Theo Nghị định 10/2022, lệ phí...")
        {'law_name': 'Nghị định 10/2022'}
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

    # Codes/acts take precedence over decrees when both match the same span
    # (e.g. "Bộ luật Dân sự 2015" is a code, not a decree).
    law_m = _LAW_RE.search(text)
    decree_m = _DECREE_RE.search(text)
    if law_m:
        out["law_name"] = law_m.group(0).strip()
    elif decree_m:
        out["law_name"] = decree_m.group(0).strip()

    return out