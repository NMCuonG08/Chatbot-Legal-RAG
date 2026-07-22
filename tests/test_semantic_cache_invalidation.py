"""Audit (external 360°) bug 7.2 — per-law semantic-cache invalidation on ingest.

When a law is re-ingested (incremental mode), cached answers that were grounded
in that law may now be stale (the underlying chunks changed). A full cache wipe
on every ingest kills hit-rate for all other laws; a TTL alone leaves up to 7
days of stale answers. The fix is targeted: stamp the laws an answer cited into
the cache payload (``law_names``), then delete only the points whose ``law_names``
contains the re-ingested law.

These tests pin:
  - ``_extract_law_names`` dedups + drops empty/None.
  - ``set_cached_response`` stores ``law_names`` from sources.
  - ``clear_semantic_cache_by_law`` deletes only matching points (filter on the
    ``law_names`` array), leaving other laws' cache entries intact.
  - ``clear_semantic_cache_by_laws`` batches over a set.
"""
from types import SimpleNamespace

import pytest

import semantic_cache as sc


class _ArrayFilter:
    """Applies a qdrant Filter where ``must`` conditions may target an array
    field (``law_names``) — a MatchValue matches when the value is IN the
    payload array (Qdrant array-field semantics)."""

    @staticmethod
    def matches(payload, f):
        must = getattr(f, "must", None) or []
        should = getattr(f, "should", None) or []
        for c in must:
            key = getattr(c, "key", None)
            val = getattr(getattr(c, "match", None), "value", None)
            field = payload.get(key)
            if isinstance(field, list):
                if val not in field:
                    return False
            elif field != val:
                return False
        if should:
            min_should = getattr(f, "min_should", None)
            min_should = getattr(min_should, "value", min_should)
            if min_should is None:
                min_should = 1
            hit = 0
            for c in should:
                key = getattr(c, "key", None)
                val = getattr(getattr(c, "match", None), "value", None)
                field = payload.get(key)
                if isinstance(field, list) and val in field:
                    hit += 1
                elif field == val:
                    hit += 1
            if hit < min_should:
                return False
        return True


class _FakeQdrant:
    def __init__(self):
        self.points = []

    def upsert(self, collection_name, points):
        for p in points:
            self.points.append(SimpleNamespace(id=p.id, vector=p.vector, payload=p.payload))

    def delete(self, collection_name, points_selector):
        survivors = [p for p in self.points if not _ArrayFilter.matches(p.payload, points_selector)]
        deleted = len(self.points) - len(survivors)
        self.points = survivors
        return SimpleNamespace(operation=SimpleNamespace(deleted_count=deleted))


@pytest.fixture
def fake(monkeypatch):
    fake = _FakeQdrant()
    monkeypatch.setattr(sc, "get_client", lambda: fake)
    monkeypatch.setattr(sc, "get_embedding", lambda text: [0.1] * 1024)
    return fake


# ---- _extract_law_names ---------------------------------------------------
def test_extract_law_names_dedup_and_drop_empty():
    sources = [
        {"law_name": "Bộ luật Dân sự"},
        {"law_name": "  Bộ luật Dân sự  "},
        {"law_name": ""},
        {"law_name": None},
        {"law_name": "Luật Đất đai"},
        {},  # no law_name key
    ]
    out = sc._extract_law_names(sources)
    assert out == ["Bộ luật Dân sự", "Luật Đất đai"]


def test_extract_law_names_empty_when_none_present():
    assert sc._extract_law_names([{"content": "x"}, {"content": "y"}]) == []


# ---- set_cached_response stamps law_names ---------------------------------
def test_set_cached_stores_law_names_from_sources(fake):
    sources = [{"law_name": "Bộ luật Dân sự"}, {"law_name": "Luật Đất đai"}]
    sc.set_cached_response("Q?", "Answer body long enough.", sources, user_id="alice")
    assert fake.points, "entry was written"
    payload = fake.points[-1].payload
    assert payload.get("law_names") == ["Bộ luật Dân sự", "Luật Đất đai"]


def test_set_cached_stores_empty_law_names_when_unknown(fake):
    sc.set_cached_response("Q?", "Answer body long enough.", [{"content": "x"}], user_id="alice")
    payload = fake.points[-1].payload
    assert payload.get("law_names") == []


# ---- clear_semantic_cache_by_law ------------------------------------------
def test_clear_by_law_deletes_only_matching(fake):
    sc.set_cached_response("q1", "Answer about civil law long enough.",
                           [{"law_name": "Bộ luật Dân sự"}], user_id="alice")
    sc.set_cached_response("q2", "Answer about land law long enough.",
                           [{"law_name": "Luật Đất đai"}], user_id="bob")
    assert len(fake.points) == 2

    deleted = sc.clear_semantic_cache_by_law("Bộ luật Dân sự")
    assert deleted == 1
    remaining = [p.payload.get("law_names") for p in fake.points]
    assert remaining == [["Luật Đất đai"]]


def test_clear_by_law_no_match_deletes_nothing(fake):
    sc.set_cached_response("q1", "Answer about land law long enough.",
                           [{"law_name": "Luật Đất đai"}], user_id="bob")
    deleted = sc.clear_semantic_cache_by_law("Bộ luật Hình sự")
    assert deleted == 0
    assert len(fake.points) == 1


def test_clear_by_law_missing_name_is_noop(fake):
    # Defensive: empty/None law name must not wipe the cache.
    sc.set_cached_response("q1", "Answer about land law long enough.",
                           [{"law_name": "Luật Đất đai"}], user_id="bob")
    assert sc.clear_semantic_cache_by_law("") == 0
    assert sc.clear_semantic_cache_by_law(None) == 0
    assert len(fake.points) == 1


# ---- clear_semantic_cache_by_laws (batch) ---------------------------------
def test_clear_by_laws_batch(fake):
    sc.set_cached_response("q1", "Civil answer long enough.",
                           [{"law_name": "Bộ luật Dân sự"}], user_id="a")
    sc.set_cached_response("q2", "Land answer long enough.",
                           [{"law_name": "Luật Đất đai"}], user_id="b")
    sc.set_cached_response("q3", "Criminal answer long enough.",
                           [{"law_name": "Bộ luật Hình sự"}], user_id="c")
    deleted = sc.clear_semantic_cache_by_laws({"Bộ luật Dân sự", "Luật Đất đai"})
    assert deleted == 2
    assert [p.payload["law_names"][0] for p in fake.points] == ["Bộ luật Hình sự"]