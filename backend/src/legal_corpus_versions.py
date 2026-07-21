"""Curated Vietnamese legal corpus version table.

Replaces the previously hardcoded 5-entry ``LAW_VERSIONS`` dict in
``legal_knowledge_tools.py`` with a standalone data module so it can grow
without inflating the tools module and so ``legal_effectivity`` can resolve the
replaces/amended chain.

Each entry is keyed by a stable ``law_key`` (snake_case, year-suffixed) and
records:

- ``full_name``        : the canonical Vietnamese name (used to match against
                        ``law_name`` parsed from corpus text).
- ``effective_from``   : ``DD/MM/YYYY`` date the statute first took effect.
- ``replaces``         : ``full_name`` of the statute this one supersedes
                        (``None`` when it is the first version).
- ``replaced_by``      : ``law_key`` of the statute that supersedes this one
                        (``None`` while still in force). Lets
                        ``classify_effectivity`` mark a statute ``repealed``.
- ``amended_by``       : list of ``full_name`` instruments that amended this
                        statute (non-superseding changes). Triggers the
                        ``amended`` status.
- ``key_articles``     : curated highlight list (display only).

Dates use ``DD/MM/YYYY`` — the Vietnamese convention. Parsers that need a
``datetime`` should split on ``/`` and read ``[2]`` as the year (existing
``get_law_version`` behaviour).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Stable shape of one version entry.
VersionEntry = Dict[str, object]

LAW_VERSIONS: Dict[str, VersionEntry] = {
    # --- Codes ---
    "blds_2015": {
        "full_name": "Bộ luật Dân sự 2015",
        "effective_from": "01/01/2017",
        "replaces": "Bộ luật Dân sự 2005",
        "replaced_by": None,
        "amended_by": ["Luật số 67/2020/QH14 (sửa đổi một số điều liên quan đầu tư)"],
        "key_articles": ["Điều 418 (phạt vi phạm ≤8%)", "Điều 651 (hàng thừa kế)"],
    },
    "blds_2005": {
        "full_name": "Bộ luật Dân sự 2005",
        "effective_from": "01/01/2006",
        "replaces": None,
        "replaced_by": "blds_2015",
        "amended_by": [],
        "key_articles": [],
    },
    "blld_2019": {
        "full_name": "Bộ luật Lao động 2019",
        "effective_from": "01/01/2021",
        "replaces": "Bộ luật Lao động 2012",
        "replaced_by": None,
        "amended_by": [],
        "key_articles": [
            "Điều 48 (trợ cấp thôi việc)",
            "Điều 107 (làm thêm giờ)",
            "Điều 193 (thời hiệu LĐ)",
        ],
    },
    "blld_2012": {
        "full_name": "Bộ luật Lao động 2012",
        "effective_from": "01/05/2013",
        "replaces": "Bộ luật Lao động 1994",
        "replaced_by": "blld_2019",
        "amended_by": [],
        "key_articles": [],
    },
    # --- Acts ---
    "luat_dat_dai_2024": {
        "full_name": "Luật Đất đai 2024",
        "effective_from": "01/01/2025",
        "replaces": "Luật Đất đai 2013",
        "replaced_by": None,
        "amended_by": [],
        "key_articles": [
            "Quyền sử dụng đất",
            "Chuyển nhượng",
            "Thu hồi, bồi thường",
        ],
    },
    "luat_dat_dai_2013": {
        "full_name": "Luật Đất đai 2013",
        "effective_from": "01/07/2014",
        "replaces": "Luật Đất đai 2003",
        "replaced_by": "luat_dat_dai_2024",
        "amended_by": [],
        "key_articles": [],
    },
    "luat_doanh_nghiep_2020": {
        "full_name": "Luật Doanh nghiệp 2020",
        "effective_from": "01/01/2021",
        "replaces": "Luật Doanh nghiệp 2014",
        "replaced_by": None,
        "amended_by": [],
        "key_articles": [
            "Điều 36 (tên doanh nghiệp)",
            "Điều 17 (điều kiện thành lập)",
        ],
    },
    "luat_hngd_2014": {
        "full_name": "Luật Hôn nhân và Gia đình 2014",
        "effective_from": "01/01/2015",
        "replaces": "Luật HNGĐ 2000",
        "replaced_by": None,
        "amended_by": [],
        "key_articles": [
            "Điều 8 (tuổi kết hôn: nam 20, nữ 18)",
            "Điều 82 (cấp dưỡng)",
        ],
    },
}


def find_version_by_name(law_name: Optional[str]) -> Optional[VersionEntry]:
    """Resolve a parsed ``law_name`` to its version entry by ``full_name``.

    Case-insensitive, whitespace-insensitive match. Returns ``None`` when the
    name is unknown to the table (caller should then default to ``in_force``
    rather than blocking ingest).
    """
    if not law_name:
        return None
    target = " ".join(law_name.strip().lower().split())
    for entry in LAW_VERSIONS.values():
        full = entry.get("full_name")
        if full and " ".join(str(full).strip().lower().split()) == target:
            return entry
    return None


def find_version_by_key(law_key: Optional[str]) -> Optional[VersionEntry]:
    """Resolve a ``law_key`` (e.g. ``blds_2015``) to its version entry."""
    if not law_key:
        return None
    return LAW_VERSIONS.get(law_key.strip().lower())


def available_law_keys() -> List[str]:
    """List all known ``law_key`` values (used for error hints)."""
    return list(LAW_VERSIONS.keys())