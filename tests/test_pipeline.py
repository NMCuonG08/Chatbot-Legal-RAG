"""Pipeline test suite.

Covers schema immutability, parsers (json/markdown/html/unknown/empty),
chunker lineage, three-tier storage idempotency, state store with a fake
DB session, embedder upsert/dedup with mocked heavy deps, every connector,
and the orchestrator loop (happy path, idempotency skip, per-doc isolation).
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline import state, storage
from pipeline.chunker import CHUNK_CONFIG_VERSION, chunk
from pipeline.connectors import (
    HtmlConnector,
    JsonlQaConnector,
    MarkdownConnector,
    PdfConnector,
)
from pipeline.connectors.base import BaseConnector
from pipeline.embedder import EMBED_MODEL_TAG, embed_chunks
from pipeline.orchestrator import run_pipeline
from pipeline.parsers import PARSER_VERSION, parse
from pipeline.schema import ChunkedDocument, ParsedDocument, RawDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lake_dir(tmp_path, monkeypatch):
    """Isolate the storage lake under a tmp dir for each test."""
    lake = tmp_path / "lake"
    monkeypatch.setenv("PIPELINE_LAKE_DIR", str(lake))
    return lake


@pytest.fixture()
def frozen_now() -> datetime:
    return datetime(2026, 7, 1, tzinfo=timezone.utc)


@pytest.fixture()
def raw_doc(frozen_now) -> RawDocument:
    return RawDocument(
        doc_id="doc-1",
        source_id="jsonl_qa",
        source_type="json",
        content=json.dumps({"question": "What is a contract?", "context": "A contract is an agreement."}),
        origin_url="file://train.jsonl",
        fetched_at=frozen_now,
        meta={"line_index": 0},
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_raw_document_is_frozen(raw_doc):
    # Arrange / Act / Assert
    with pytest.raises((AttributeError, TypeError)):
        raw_doc.doc_id = "changed"


def test_schema_defaults_are_empty_dicts():
    # Arrange
    now = datetime.now(timezone.utc)
    # Act
    raw = RawDocument(doc_id="d", source_id="s", source_type="json", content="x", origin_url="u", fetched_at=now)
    parsed = ParsedDocument(
        doc_id="d", source_id="s", source_type="json", text="t", parser_used="p",
        parser_version="1", raw_doc_id="d", fetched_at=now,
    )
    chunked = ChunkedDocument(doc_id="d", source_id="s", chunks=(), chunk_config="c", embed_model="", parsed_doc_id="d")
    # Assert
    assert raw.meta == {}
    assert parsed.meta == {}
    assert chunked.meta == {}
    assert chunked.chunks == ()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_json_reconstructs_question_and_context(raw_doc):
    # Act
    parsed = parse(raw_doc)
    # Assert
    assert parsed.parser_used == "json_qa"
    assert parsed.parser_version == PARSER_VERSION
    assert "What is a contract?" in parsed.text
    assert "A contract is an agreement." in parsed.text
    assert parsed.raw_doc_id == raw_doc.doc_id


def test_parse_json_accepts_legacy_answer_key(frozen_now):
    # Arrange
    doc = RawDocument(
        doc_id="d", source_id="s", source_type="json",
        content=json.dumps({"question": "Q?", "answer": "A."}),
        origin_url="u", fetched_at=frozen_now,
    )
    # Act
    parsed = parse(doc)
    # Assert
    assert "Q?" in parsed.text and "A." in parsed.text


def test_parse_markdown_is_passthrough(frozen_now):
    # Arrange
    md = "# Heading\n\nBody text."
    doc = RawDocument(doc_id="d", source_id="s", source_type="markdown", content=md, origin_url="u", fetched_at=frozen_now)
    # Act
    parsed = parse(doc)
    # Assert
    assert parsed.text == md
    assert parsed.parser_used == "markdown_passthrough"


def test_parse_html_strips_tags(frozen_now):
    # Arrange
    html = "<html><body><p>Hello <b>world</b></p><script>bad()</script></body></html>"
    doc = RawDocument(doc_id="d", source_id="s", source_type="html", content=html, origin_url="u", fetched_at=frozen_now)
    # Act
    parsed = parse(doc)
    # Assert
    assert "Hello" in parsed.text
    assert "world" in parsed.text
    assert "bad" not in parsed.text
    assert "<" not in parsed.text
    assert parsed.parser_used == "html_regex"


def test_parse_unknown_source_type_raises(frozen_now):
    # Arrange
    doc = RawDocument(doc_id="d", source_id="s", source_type="xml", content="<x/>", origin_url="u", fetched_at=frozen_now)
    # Act / Assert
    with pytest.raises(ValueError, match="Unknown source_type"):
        parse(doc)


def test_parse_empty_text_raises(frozen_now):
    # Arrange — json object with empty fields yields empty text
    doc = RawDocument(
        doc_id="d", source_id="s", source_type="json",
        content=json.dumps({"question": "", "context": ""}),
        origin_url="u", fetched_at=frozen_now,
    )
    # Act / Assert
    with pytest.raises(ValueError, match="Parsed text empty"):
        parse(doc)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def test_chunk_builds_lineage_and_config_suffix(frozen_now, monkeypatch):
    # Arrange
    parsed = ParsedDocument(
        doc_id="d", source_id="jsonl_qa", source_type="json", text="t",
        parser_used="json_qa", parser_version="1", raw_doc_id="d", fetched_at=frozen_now,
        meta={"origin_url": "u"},
    )

    class _Node:
        def __init__(self, text):
            self.text = text

    fake_splitter = types.ModuleType("splitter")
    fake_splitter.split_document = lambda text, use_semantic=True: [_Node("a"), _Node("b")]
    monkeypatch.setitem(sys.modules, "splitter", fake_splitter)

    # Act
    chunked = chunk(parsed, use_semantic=True)

    # Assert
    assert chunked.chunks == ("a", "b")
    assert chunked.chunk_config == CHUNK_CONFIG_VERSION + "+semantic"
    assert chunked.parsed_doc_id == parsed.doc_id
    assert chunked.embed_model == ""
    assert chunked.meta["parser_used"] == "json_qa"


def test_chunk_token_config_when_semantic_off(frozen_now, monkeypatch):
    # Arrange
    parsed = ParsedDocument(
        doc_id="d", source_id="s", source_type="markdown", text="t",
        parser_used="markdown_passthrough", parser_version="1", raw_doc_id="d", fetched_at=frozen_now,
    )
    fake_splitter = types.ModuleType("splitter")
    fake_splitter.split_document = lambda text, use_semantic=True: []
    monkeypatch.setitem(sys.modules, "splitter", fake_splitter)

    # Act
    chunked = chunk(parsed, use_semantic=False)

    # Assert
    assert chunked.chunk_config.endswith("+token")


# ---------------------------------------------------------------------------
# Storage (uses lake_dir fixture for isolation)
# ---------------------------------------------------------------------------


def test_persist_raw_is_write_once_idempotent(lake_dir, raw_doc):
    # Act
    first = storage.persist_raw(raw_doc)
    size_first = first.stat().st_size
    second = storage.persist_raw(raw_doc)
    # Assert
    assert first == second
    assert second.stat().st_size == size_first
    assert storage.raw_dir(raw_doc.source_id).exists()


def test_persist_parsed_writes_lineage_json(lake_dir, frozen_now):
    # Arrange
    parsed = ParsedDocument(
        doc_id="d", source_id="jsonl_qa", source_type="json", text="hello",
        parser_used="json_qa", parser_version="1", raw_doc_id="d", fetched_at=frozen_now,
        meta={"origin_url": "u"},
    )
    # Act
    path = storage.persist_parsed(parsed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Assert
    assert payload["raw_doc_id"] == "d"
    assert payload["parser_used"] == "json_qa"
    assert payload["text"] == "hello"


def test_persist_serving_writes_chunks_and_lineage(lake_dir):
    # Arrange
    chunked = ChunkedDocument(
        doc_id="d", source_id="jsonl_qa", chunks=("a", "b"),
        chunk_config="semantic_v1+semantic", embed_model="custom_embedding_v1",
        parsed_doc_id="d", meta={"question": "Q?"},
    )
    # Act
    path = storage.persist_serving(chunked)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Assert
    assert payload["chunks"] == ["a", "b"]
    assert payload["parsed_doc_id"] == "d"
    assert payload["embed_model"] == "custom_embedding_v1"


def test_persist_raw_sanitizes_path_nasty_doc_id(lake_dir, frozen_now):
    # Arrange — doc_id with path separators must not escape the raw dir
    raw = RawDocument(
        doc_id="../../etc/passwd", source_id="s", source_type="markdown",
        content="x", origin_url="u", fetched_at=frozen_now,
    )
    # Act
    path = storage.persist_raw(raw)
    # Assert — file stays inside raw_dir; no path separators leaked into name
    assert path.parent == storage.raw_dir("s")
    assert "/" not in path.name
    assert "\\" not in path.name


# ---------------------------------------------------------------------------
# State store (fake DB session)
# ---------------------------------------------------------------------------


class _FakeSession:
    """Minimal in-memory session matching the SQLAlchemy calls state.py makes."""

    def __init__(self):
        self.store: dict[str, object] = {}

    def get(self, entity, pk):
        return self.store.get(pk)

    def add(self, row):
        self.store[row.doc_id] = row

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture()
def fake_db(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(state, "_new_db_session", lambda: session)
    monkeypatch.setattr(state, "ensure_pipeline_schema", lambda: None)
    return session


def test_state_is_done_false_when_no_row(fake_db):
    assert state.is_done("missing") is False


def test_state_mark_then_is_done_true(fake_db):
    # Act
    state.mark_status("d1", "jsonl_qa", state.STATUS_EMBEDDED)
    # Assert
    assert state.get_status("d1") == state.STATUS_EMBEDDED
    assert state.is_done("d1") is True


def test_state_mark_failed_records_error(fake_db):
    # Act
    state.mark_status("d2", "pdf_dir", state.STATUS_FAILED, error="boom")
    # Assert
    assert state.get_status("d2") == state.STATUS_FAILED
    assert state.is_done("d2") is False
    row = fake_db.store["d2"]
    assert row.error == "boom"


def test_state_mark_upserts_existing_row(fake_db):
    # Arrange
    state.mark_status("d3", "jsonl_qa", state.STATUS_FETCHED)
    # Act
    state.mark_status("d3", "jsonl_qa", state.STATUS_PARSED)
    # Assert
    assert state.get_status("d3") == state.STATUS_PARSED
    assert len(fake_db.store) == 1


# ---------------------------------------------------------------------------
# Embedder (mocked heavy deps: custom_embedding, vectorize, models)
# ---------------------------------------------------------------------------


def _install_embedder_mocks(monkeypatch, existing_chunks=None, captured=None):
    """Inject fake custom_embedding/vectorize/models modules."""
    existing_chunks = existing_chunks or []

    fake_custom_embedding = types.ModuleType("custom_embedding")
    fake_custom_embedding.get_custom_embedding = lambda texts: [[0.1, 0.2]] * len(texts)
    monkeypatch.setitem(sys.modules, "custom_embedding", fake_custom_embedding)

    captured["upserts"] = []
    captured["deletes"] = []
    captured["saved"] = []
    captured["deleted_chunks"] = []
    fake_vectorize = types.ModuleType("vectorize")

    def _add_vector(collection_name, vectors):
        captured["upserts"].append((collection_name, vectors))

    def _del_vectors(collection_name, ids):
        captured["deletes"].append((collection_name, list(ids)))

    fake_vectorize.add_vector = _add_vector
    fake_vectorize.delete_vectors_by_ids = _del_vectors
    monkeypatch.setitem(sys.modules, "vectorize", fake_vectorize)

    fake_models = types.ModuleType("models")
    fake_models.get_doc_chunks = lambda doc_id: list(existing_chunks)

    def _save(doc_id, cid, chash):
        captured["saved"].append((doc_id, cid, chash))

    def _del_chunks(ids):
        captured["deleted_chunks"] = list(ids)

    fake_models.save_doc_chunk = _save
    fake_models.delete_doc_chunks_by_ids = _del_chunks
    monkeypatch.setitem(sys.modules, "models", fake_models)


def test_embed_chunks_returns_model_tag_and_upserts_new(lake_dir, monkeypatch):
    # Arrange
    chunked = ChunkedDocument(
        doc_id="d", source_id="jsonl_qa", chunks=("a", "b"),
        chunk_config="semantic_v1+semantic", embed_model="", parsed_doc_id="d",
        meta={"question": "Q?"},
    )
    captured: dict = {}
    _install_embedder_mocks(monkeypatch, captured=captured)

    # Act
    result = embed_chunks(chunked, collection_name="legal_collection")

    # Assert
    assert result.embed_model == EMBED_MODEL_TAG
    assert len(captured["upserts"]) == 1
    coll, vectors = captured["upserts"][0]
    assert coll == "legal_collection"
    assert len(vectors) == 2
    assert len(captured["saved"]) == 2


def test_embed_chunks_skips_when_no_chunks(lake_dir, monkeypatch):
    # Arrange
    chunked = ChunkedDocument(
        doc_id="d", source_id="jsonl_qa", chunks=(),
        chunk_config="semantic_v1+semantic", embed_model="", parsed_doc_id="d",
    )
    captured: dict = {}
    _install_embedder_mocks(monkeypatch, captured=captured)

    # Act
    result = embed_chunks(chunked, collection_name="legal_collection")

    # Assert
    assert result.embed_model == EMBED_MODEL_TAG
    assert captured["upserts"] == []
    assert captured["saved"] == []


def test_embed_chunks_deletes_orphans_and_skips_unchanged(lake_dir, monkeypatch):
    # Arrange — one existing chunk matches new cid/hash → skip; one orphan → delete
    chunked = ChunkedDocument(
        doc_id="d", source_id="jsonl_qa", chunks=("keep",),
        chunk_config="semantic_v1+semantic", embed_model="", parsed_doc_id="d",
        meta={"question": "Q?"},
    )
    keep_cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc_d_chunk_0"))
    keep_hash = hashlib.md5(b"keep").hexdigest()

    class _Existing:
        def __init__(self, cid, chash):
            self.chunk_id = cid
            self.chunk_hash = chash

    existing = [_Existing(keep_cid, keep_hash), _Existing("orphan-cid", "deadbeef")]
    captured: dict = {}
    _install_embedder_mocks(monkeypatch, existing_chunks=existing, captured=captured)

    # Act
    embed_chunks(chunked, collection_name="legal_collection")

    # Assert
    assert captured["deleted_chunks"] == ["orphan-cid"]
    assert ("legal_collection", ["orphan-cid"]) in captured["deletes"]
    # unchanged chunk not re-upserted
    assert captured["saved"] == []


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------


def test_jsonl_connector_fetches_and_skips_bad_lines(tmp_path):
    # Arrange
    f = tmp_path / "train.jsonl"
    lines = [
        json.dumps({"question": "Q1", "context": "C1"}),
        "not-json",
        json.dumps({"question": "", "context": "empty-q"}),
        json.dumps({"question": "Q2", "answer": "C2"}),
    ]
    f.write_text("\n".join(lines), encoding="utf-8")
    # Act
    docs = JsonlQaConnector(f).fetch()
    # Assert
    assert len(docs) == 2
    assert docs[0].source_type == "json"
    assert all(d.source_id == "jsonl_qa" for d in docs)
    assert docs[1].meta["line_index"] == 3


def test_jsonl_connector_respects_limit(tmp_path):
    # Arrange
    f = tmp_path / "train.jsonl"
    f.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "context": f"C{i}"}) for i in range(5)),
        encoding="utf-8",
    )
    # Act
    docs = JsonlQaConnector(f, limit=2).fetch()
    # Assert
    assert len(docs) == 2


def test_jsonl_connector_missing_file_returns_empty(tmp_path):
    # Act
    docs = JsonlQaConnector(tmp_path / "nope.jsonl").fetch()
    # Assert
    assert docs == []


def test_jsonl_connector_doc_id_is_content_hash(tmp_path):
    # Arrange
    f = tmp_path / "train.jsonl"
    payload = json.dumps({"question": "Q", "context": "C"})
    f.write_text(payload + "\n", encoding="utf-8")
    expected = "jsonl_qa-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    # Act
    docs = JsonlQaConnector(f).fetch()
    # Assert
    assert docs[0].doc_id == expected


def test_markdown_connector_reads_md_files(tmp_path):
    # Arrange
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")
    # Act
    docs = MarkdownConnector(tmp_path).fetch()
    # Assert
    assert len(docs) == 2
    assert all(d.source_type == "markdown" for d in docs)
    assert {d.meta["file_name"] for d in docs} == {"a.md", "b.md"}


def test_markdown_connector_missing_dir_returns_empty(tmp_path):
    assert MarkdownConnector(tmp_path / "nope").fetch() == []


def test_html_connector_reads_html_files(tmp_path):
    # Arrange
    (tmp_path / "page.html").write_text("<p>Hi</p>", encoding="utf-8")
    (tmp_path / "skip.md").write_text("nope", encoding="utf-8")
    # Act
    docs = HtmlConnector(root_dir=tmp_path).fetch()
    # Assert
    assert len(docs) == 1
    assert docs[0].source_type == "html"
    assert "<p>Hi</p>" in docs[0].content


def test_html_connector_dead_url_does_not_raise(tmp_path):
    # Arrange — unreachable URL; requests not needed if it errors out
    docs = HtmlConnector(urls=["http://127.0.0.1:1/nope"]).fetch()
    # Assert
    assert docs == []


def test_pdf_connector_missing_dir_returns_empty(tmp_path):
    assert PdfConnector(tmp_path / "nope").fetch() == []


def test_pdf_connector_encodes_bytes_as_base64(tmp_path):
    # Arrange
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    # Act
    docs = PdfConnector(tmp_path).fetch()
    # Assert
    assert len(docs) == 1
    assert docs[0].source_type == "pdf"
    assert base64.b64decode(docs[0].content) == b"%PDF-1.4 fake"
    assert docs[0].meta["sha256"] == hashlib.sha256(b"%PDF-1.4 fake").hexdigest()


# ---------------------------------------------------------------------------
# Orchestrator (mocked state/storage/embedder; real parse+chunk)
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(self, text):
        self.text = text


@pytest.fixture()
def patched_orchestrator(monkeypatch, fake_db, lake_dir):
    """Wire real state (fake_db) + real storage (lake_dir) + fake splitter + fake embedder deps."""
    fake_splitter = types.ModuleType("splitter")
    fake_splitter.split_document = lambda text, use_semantic=True: [_FakeNode(text)]
    monkeypatch.setitem(sys.modules, "splitter", fake_splitter)

    fake_custom_embedding = types.ModuleType("custom_embedding")
    fake_custom_embedding.get_custom_embedding = lambda texts: [[0.1]] * len(texts)
    monkeypatch.setitem(sys.modules, "custom_embedding", fake_custom_embedding)

    captured: dict = {"upserts": []}
    fake_vectorize = types.ModuleType("vectorize")
    fake_vectorize.add_vector = lambda collection_name, vectors: captured["upserts"].append((collection_name, vectors))
    fake_vectorize.delete_vectors_by_ids = lambda collection_name, ids: None
    monkeypatch.setitem(sys.modules, "vectorize", fake_vectorize)

    fake_models = types.ModuleType("models")
    fake_models.get_doc_chunks = lambda doc_id: []
    fake_models.save_doc_chunk = lambda doc_id, cid, chash: None
    fake_models.delete_doc_chunks_by_ids = lambda ids: None
    monkeypatch.setitem(sys.modules, "models", fake_models)
    return captured


def test_run_pipeline_happy_path(patched_orchestrator, tmp_path):
    # Arrange
    f = tmp_path / "train.jsonl"
    f.write_text(json.dumps({"question": "Q", "context": "C"}) + "\n", encoding="utf-8")
    connector = JsonlQaConnector(f)

    # Act
    stats = run_pipeline([connector], collection_name="legal_collection")

    # Assert
    assert stats["fetched"] == 1
    assert stats["parsed"] == 1
    assert stats["chunked"] == 1
    assert stats["embedded"] == 1
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    assert len(patched_orchestrator["upserts"]) == 1


def test_run_pipeline_idempotency_skips_already_embedded(patched_orchestrator, tmp_path):
    # Arrange
    f = tmp_path / "train.jsonl"
    payload = json.dumps({"question": "Q", "context": "C"})
    f.write_text(payload + "\n", encoding="utf-8")
    docs = JsonlQaConnector(f).fetch()
    state.mark_status(docs[0].doc_id, docs[0].source_id, state.STATUS_EMBEDDED)

    # Act
    stats = run_pipeline([JsonlQaConnector(f)], collection_name="legal_collection")

    # Assert
    assert stats["skipped"] == 1
    assert stats["embedded"] == 0
    assert patched_orchestrator["upserts"] == []


def test_run_pipeline_isolates_per_doc_failure(patched_orchestrator, tmp_path, monkeypatch):
    # Arrange — two lines; force the parser to blow up on the first only.
    f = tmp_path / "train.jsonl"
    f.write_text(
        json.dumps({"question": "good", "context": "good"}) + "\n"
        + json.dumps({"question": "also", "context": "also"}) + "\n",
        encoding="utf-8",
    )
    real_parse = parse

    calls = {"n": 0}

    def _flaky(doc):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom on first doc")
        return real_parse(doc)

    monkeypatch.setattr("pipeline.orchestrator.parse", _flaky)
    # Act
    stats = run_pipeline([JsonlQaConnector(f)], collection_name="legal_collection")
    # Assert
    assert stats["failed"] == 1
    assert stats["embedded"] == 1


def test_run_pipeline_connector_fetch_failure_does_not_halt(patched_orchestrator, tmp_path):
    # Arrange — first connector raises on fetch, second succeeds.
    f = tmp_path / "train.jsonl"
    f.write_text(json.dumps({"question": "Q", "context": "C"}) + "\n", encoding="utf-8")

    class _Broken(BaseConnector):
        source_id = "broken"
        source_type = "json"

        def fetch(self):
            raise RuntimeError("source down")

    # Act
    stats = run_pipeline([_Broken(), JsonlQaConnector(f)], collection_name="legal_collection")
    # Assert
    assert stats["embedded"] == 1
    assert stats["failed"] == 0


def test_run_pipeline_limit_caps_docs(patched_orchestrator, tmp_path):
    # Arrange
    f = tmp_path / "train.jsonl"
    f.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "context": f"C{i}"}) for i in range(5)),
        encoding="utf-8",
    )
    # Act
    stats = run_pipeline([JsonlQaConnector(f)], collection_name="legal_collection", limit=2)
    # Assert — at most one doc's worth of stages before cap triggers
    assert stats["embedded"] <= 1