"""Security layer tests: path-traversal guard + admin API-key dependency.

Isolated from the heavy tasks.py import chain (which depends on a broken
langchain_groq version in the user env). Tests only security.py + FastAPI
TestClient on a tiny app that mounts the protected deps.
"""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from security import (
    get_legal_collection_name,
    require_api_key,
    resolve_safe_data_path,
)


@pytest.fixture()
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPORT_DATA_DIR", str(tmp_path))
    (tmp_path / "train.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "secret.txt").write_text("LEAK", encoding="utf-8")
    return tmp_path


def test_resolve_safe_data_path_rejects_traversal(isolated_data_dir):
    with pytest.raises(Exception) as exc:
        resolve_safe_data_path("../../etc/passwd")
    assert exc.value.status_code == 400


def test_resolve_safe_data_path_rejects_absolute_outside(isolated_data_dir):
    with pytest.raises(Exception):
        resolve_safe_data_path(str(isolated_data_dir.parent / "outside.jsonl"))


def test_resolve_safe_data_path_accepts_relative_inside(isolated_data_dir):
    p = resolve_safe_data_path("train.jsonl")
    assert p.exists()
    assert p.name == "train.jsonl"


def test_resolve_safe_data_path_default_file(isolated_data_dir):
    p = resolve_safe_data_path(None)
    assert p.name == "train.jsonl"


def test_resolve_safe_data_path_missing_file_404(isolated_data_dir):
    with pytest.raises(Exception) as exc:
        resolve_safe_data_path("nonexistent.jsonl")
    assert exc.value.status_code == 404


# ===== Collection name validation =====


def test_collection_name_valid():
    assert get_legal_collection_name("llm_v2") == "llm_v2"


def test_collection_name_rejects_uppercase():
    with pytest.raises(Exception):
        get_legal_collection_name("LLM")


def test_collection_name_rejects_special_chars():
    with pytest.raises(Exception):
        get_legal_collection_name("llm;drop")


def test_collection_name_rejects_empty():
    with pytest.raises(Exception):
        get_legal_collection_name("")


# ===== Admin API key dependency =====


def _make_app(monkeypatch, *, admin_key, unsafe):
    monkeypatch.setenv("ADMIN_API_KEY", admin_key or "")
    monkeypatch.setenv("ALLOW_UNSAFE_ADMIN", "1" if unsafe else "0")
    app = FastAPI()

    @app.post("/admin")
    async def admin(_: None = Depends(require_api_key)):
        return {"ok": True}

    return TestClient(app)


def test_admin_endpoint_refused_without_config(monkeypatch):
    client = _make_app(monkeypatch, admin_key="", unsafe=False)
    r = client.post("/admin")
    assert r.status_code == 503


def test_admin_endpoint_rejects_wrong_key(monkeypatch):
    client = _make_app(monkeypatch, admin_key="secret-key", unsafe=False)
    r = client.post("/admin", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_admin_endpoint_accepts_correct_key(monkeypatch):
    client = _make_app(monkeypatch, admin_key="secret-key", unsafe=False)
    r = client.post("/admin", headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_admin_endpoint_unsafe_dev_bypass(monkeypatch):
    client = _make_app(monkeypatch, admin_key="", unsafe=True)
    r = client.post("/admin")
    assert r.status_code == 200