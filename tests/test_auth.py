"""Phase 2 — auth module tests (password hashing + JWT encode/decode).

No DB needed: tests only ``auth.py`` crypto helpers with a synthetic
``JWT_SECRET`` via the ``jwt_secret`` fixture.
"""
import pytest

import auth
from auth import (
    create_access_token,
    decode_token,
    extract_bearer,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password_roundtrip(jwt_secret):
    h = hash_password("s3cret-pass")
    assert h != "s3cret-pass"
    assert verify_password("s3cret-pass", h) is True
    assert verify_password("wrong", h) is False


def test_create_and_decode_token_roundtrip(jwt_secret):
    token = create_access_token(
        subject="user-123", claims={"username": "alice", "role": "admin"}
    )
    payload = decode_token(token)
    assert payload["sub"] == "user-123"
    assert payload["username"] == "alice"
    assert payload["role"] == "admin"
    assert "exp" in payload and "iat" in payload


def test_decode_token_rejects_tampered(jwt_secret):
    token = create_access_token(subject="user-123", claims={"role": "user"})
    tampered = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
    with pytest.raises(Exception):
        decode_token(tampered)


def test_extract_bearer_strips_scheme():
    assert extract_bearer("Bearer abc.def.ghi") == "abc.def.ghi"
    assert extract_bearer("bearer abc.def.ghi") == "abc.def.ghi"
    # No scheme prefix -> rejected (must use "Bearer <token>").
    assert extract_bearer("abc.def.ghi") is None
    assert extract_bearer(None) is None
    assert extract_bearer("") is None


def test_auth_not_configured_without_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ALLOW_UNSAFE_AUTH", raising=False)
    assert auth.auth_configured() is False


def test_auth_unsafe_fallback_random_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("ALLOW_UNSAFE_AUTH", "1")
    assert auth.auth_configured() is True
    token = create_access_token(subject="x", claims={})
    assert decode_token(token)["sub"] == "x"