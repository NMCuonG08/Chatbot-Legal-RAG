"""Audit (external 360°) bug #3 — HNSW m=0 disables the ANN index.

vectorize.create_collection / wipe_collection previously created the Qdrant
collection with ``HnswConfigDiff(m=0, on_disk=True)`` plus
``OptimizersConfigDiff(indexing_threshold=999999)``. m=0 means no HNSW graph is
built at all, and indexing_threshold=999999 means segments are never indexed —
so every search degrades to a brute-force O(N) scan over disk-resident vectors.
That is catastrophic at corpus scale.

These tests pin the fix: a real HNSW graph (m >= 16, the Qdrant default) and an
indexing threshold below the "never index" sentinel.
"""
import pytest

pytestmark = pytest.mark.unit

import vectorize


class _FakeClient:
    """Records the collection-config kwargs handed to create_collection."""

    def __init__(self):
        self.created = []
        self.deleted = []

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        return None

    def delete_collection(self, **kwargs):
        self.deleted.append(kwargs)
        return None


def _hnsw(kwargs):
    return kwargs["hnsw_config"]


def _opt(kwargs):
    return kwargs["optimizers_config"]


def test_create_collection_builds_hnsw_graph():
    fake = _FakeClient()
    vectorize.set_qdrant_client(fake)
    vectorize.create_collection("legal_chunks")
    assert fake.created, "create_collection must call client.create_collection"
    cfg = fake.created[-1]
    assert _hnsw(cfg).m >= 16, "m=0 disables ANN — must be >= 16 (Qdrant default)"
    assert _opt(cfg).indexing_threshold < 999999, (
        "indexing_threshold=999999 never indexes — must be a real threshold"
    )


def test_wipe_collection_rebuilds_hnsw_graph():
    fake = _FakeClient()
    vectorize.set_qdrant_client(fake)
    vectorize.wipe_collection("legal_chunks")
    assert fake.created, "wipe_collection must recreate the collection"
    cfg = fake.created[-1]
    assert _hnsw(cfg).m >= 16
    assert _opt(cfg).indexing_threshold < 999999