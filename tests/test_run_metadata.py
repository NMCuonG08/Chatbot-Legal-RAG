"""Unit tests for run_metadata: hashing, env scrubbing, git capture, assembly."""
import dataclasses

from evaluation.run_metadata import (
    EVAL_VERSION,
    RunMetadata,
    build_run_metadata,
    capture_env,
    compute_prompt_hash,
    metadata_to_dict,
)


def test_compute_prompt_hash_stable():
    h1 = compute_prompt_hash(["a", "b"])
    h2 = compute_prompt_hash(["a", "b"])
    assert h1 == h2 and len(h1) == 64


def test_compute_prompt_hash_order_boundary():
    # Boundary byte prevents ["ab","c"] colliding with ["a","bc"].
    assert compute_prompt_hash(["ab", "c"]) != compute_prompt_hash(["a", "bc"])


def test_capture_env_records_secret_as_bool_only(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value-123")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    snap = capture_env()
    assert snap["GROQ_API_KEY"] is True  # presence only, value never stored
    assert snap["LLM_MODEL"] == "llama-3.3-70b-versatile"
    assert "super-secret-value-123" not in str(snap)


def test_capture_env_ignores_unrelated_vars(monkeypatch):
    monkeypatch.setenv("RANDOM_UNRELATED_VAR", "x")
    snap = capture_env()
    assert "RANDOM_UNRELATED_VAR" not in snap


def test_build_run_metadata_immutable():
    meta = build_run_metadata(
        judge_provider="groq", judge_model="llama-3.1-8b-instant",
        judge_temperature=0.0, judge_prompt_hash="abc",
        run_id="fixed-id",
    )
    assert isinstance(meta, RunMetadata)
    assert meta.run_id == "fixed-id"
    assert meta.eval_version == EVAL_VERSION
    assert meta.judge_provider == "groq"
    try:
        meta.run_id = "other"  # type: ignore[misc]
        raise AssertionError("RunMetadata should be frozen")
    except dataclasses.FrozenInstanceError:
        pass


def test_metadata_to_dict_roundtrips_jsonable():
    import json
    meta = build_run_metadata(
        judge_provider="groq", judge_model="m",
        judge_temperature=0.0, judge_prompt_hash="h",
    )
    d = metadata_to_dict(meta)
    s = json.dumps(d, default=str)
    assert "run_id" in json.loads(s)