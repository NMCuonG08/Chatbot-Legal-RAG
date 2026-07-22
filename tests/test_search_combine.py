"""Audit (external) — search.py RRF dedup + init thread-safety.

Verifies the two REAL issues from the external deep-dive audit:
1. RRF duplicate-chunk: BM25 text is pyvi-tokenized ("Đất_đai") while vector
   text is raw ("Đất đai"). The dedup hash must normalize whitespace + underscores
   so the same chunk merges under one key instead of appearing twice (2x context).
2. The dedup hash must be deterministic (md5), not Python's process-randomized
   hash(), so a key is stable and collision-resistant.
3. BM25 init/swap globals are guarded by a lock so concurrent cold-start in one
   process cannot race a None retriever.

Bugs 4 (Neo4j CITES*1..3 blowup) and 5 (event-loop RuntimeError) from the same
audit were verified FABRICATED against the real code and are NOT fixed here:
- graph traversal is single-hop with LIMIT in legal_graph_tools.py, no variable-
  length `*1..3` exists anywhere.
- run_async uses asyncio.run deliberately (contextvars propagation for
  @traceable); a persistent loop would regress tracing.
"""
import threading

import pytest

pytestmark = pytest.mark.unit

import search


# ---- lightweight fakes mimicking llama_index NodeWithScore shape ----------
class _FakeNode:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


class _FakeNodeWithScore:
    def __init__(self, text, metadata, score):
        self.node = _FakeNode(text, metadata)
        self.score = score


def _bm25_node(text, question="", doc_id=0, score=0.5, source="bm25"):
    return _FakeNodeWithScore(text, {"question": question, "doc_id": doc_id,
                                     "source": source}, score)


def _vec_doc(text, question="", doc_id=0, score=0.5, source="vector"):
    return {"content": text, "question": question, "doc_id": doc_id,
            "source": source, "similarity_score": score}


# ---------------------------------------------------------------------------
# Bug 1 + 3: normalized, deterministic dedup hash
# ---------------------------------------------------------------------------
def test_norm_hash_equivalence_tokenized_vs_raw():
    # BM25 tokenized text vs vector raw text must hash to the SAME key.
    a = search._content_norm_hash("Đất_đai và sổ_đỏ", "Q?")
    b = search._content_norm_hash("Đất đai và sổ đỏ", "Q?")
    assert a == b


def test_norm_hash_strips_prepended_question():
    h1 = search._content_norm_hash("Câu hỏi đây Nội dung chunk", "Câu hỏi đây")
    h2 = search._content_norm_hash("Nội dung chunk", "Câu hỏi đây")
    assert h1 == h2


def test_norm_hash_is_deterministic_string():
    h = search._content_norm_hash("Điều 100 Bộ luật Dân sự", "")
    assert isinstance(h, str)
    assert h == search._content_norm_hash("Điều 100 Bộ luật Dân sự", "")


def test_norm_hash_case_insensitive():
    assert search._content_norm_hash("Đất Đai", "") == \
           search._content_norm_hash("đất đai", "")


# ---------------------------------------------------------------------------
# Bug 1: RRF merges tokenized BM25 chunk + raw vector chunk (no duplicate)
# ---------------------------------------------------------------------------
def test_combine_merges_underscored_bm25_with_raw_vector_no_docid():
    """doc_id falsy on both sides -> hash path. Underscore mismatch must NOT
    produce two chunks; RRF overlap must merge them into ONE hybrid doc."""
    chunk = "Quyền sở hữu đất đai và sổ đỏ"
    bm25 = [_bm25_node("Quyền sở hữu đất_đai và sổ_đỏ", question="Q",
                       doc_id=0, score=0.9)]
    vector = [_vec_doc(chunk, question="Q", doc_id=0, score=0.8)]

    out = search.combine_search_results(bm25, vector, "Q")
    assert len(out) == 1, f"expected 1 merged doc, got {len(out)} (duplicate chunk bug)"
    merged = out[0]
    assert merged["search_method"] == "hybrid"
    assert "bm25_score" in merged and "vector_score" in merged


def test_combine_merges_same_docid_even_when_text_differs():
    """doc_id truthy -> id_key path; merges regardless of tokenization."""
    bm25 = [_bm25_node("Đất_đai", question="Q", doc_id="abc-1", score=0.9)]
    vector = [_vec_doc("Đất đai", question="Q", doc_id="abc-1", score=0.8)]
    out = search.combine_search_results(bm25, vector, "Q")
    assert len(out) == 1
    assert out[0]["search_method"] == "hybrid"


def test_combine_keeps_disjoint_chunks_separate():
    bm25 = [_bm25_node("Nội dung A", question="Q", doc_id=0, score=0.9)]
    vector = [_vec_doc("Nội dung B hoàn toàn khác", question="Q", doc_id=0,
                       score=0.8)]
    out = search.combine_search_results(bm25, vector, "Q")
    assert len(out) == 2  # genuinely different -> no false merge


# ---------------------------------------------------------------------------
# Bug 2: init globals guarded by a lock
# ---------------------------------------------------------------------------
def test_init_lock_exists():
    assert isinstance(search._init_lock,
                      (type(threading.Lock()), type(threading.RLock())))


def test_initialize_search_index_acquires_init_lock(monkeypatch):
    """The init path must hold _init_lock while writing globals. Verified by
    wrapping the lock with a recorder."""
    acquired = {"n": 0}
    real_lock = search._init_lock

    class _Recording:
        def __enter__(self):
            acquired["n"] += 1
            return real_lock.__enter__()

        def __exit__(self, *a):
            return real_lock.__exit__(*a)

    monkeypatch.setattr(search, "_init_lock", _Recording())
    monkeypatch.setattr(search, "_search_engine_initialized", False)
    search.initialize_search_index([])  # empty docs -> early return, still locked
    assert acquired["n"] >= 1, "initialize_search_index must hold _init_lock"