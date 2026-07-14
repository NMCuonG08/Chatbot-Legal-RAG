"""Authentication primitives: password hashing + JWT issuance/verification.

Env:
- JWT_SECRET  : HMAC secret (required in production; if unset, auth endpoints
                refuse unless ALLOW_UNSAFE_AUTH=1, which falls back to a
                process-local random secret so tokens never validate across
                restarts — dev only).
- JWT_ALG     : HS256 (default).
- JWT_EXP_MIN : access-token lifetime in minutes (default 60).

Uses passlib bcrypt for password hashing and python-jose for JWT. SQLAlchemy
User model lives in models.py; this module only handles crypto + token strings.
"""
from __future__ import annotations

import logging
import os
import secrets as _secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = os.getenv("JWT_ALG", "HS256")
_EXP_MIN = int(os.getenv("JWT_EXP_MIN", "60"))

# Dev fallback secret: generated ONCE at import time so encode/decode within
# the same process agree. Tokens still invalidate across restarts (dev only).
_UNSAFE_SECRET: Optional[str] = None


def _get_secret() -> str:
    global _UNSAFE_SECRET
    secret = os.getenv("JWT_SECRET")
    if secret:
        return secret
    if os.getenv("ALLOW_UNSAFE_AUTH", "0") == "1":
        # Dev fallback: random per-process secret. Tokens invalidate on restart.
        if _UNSAFE_SECRET is None:
            _UNSAFE_SECRET = _secrets.token_urlsafe(32)
        return _UNSAFE_SECRET
    raise RuntimeError(
        "JWT_SECRET chưa cấu hình. Thiết lập biến môi trường, hoặc set "
        "ALLOW_UNSAFE_AUTH=1 cho môi trường dev."
    )


def auth_configured() -> bool:
    """True when JWT_SECRET is set (auth endpoints safe to serve)."""
    return bool(os.getenv("JWT_SECRET")) or os.getenv("ALLOW_UNSAFE_AUTH", "0") == "1"


def hash_password(password: str) -> str:
    return _pwd_ctx.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return _pwd_ctx.verify(password, password_hash)
    except (ValueError, TypeError) as exc:
        logger.warning("Password verify failed: %s", exc)
        return False


def create_access_token(subject: str, claims: Optional[Dict[str, Any]] = None) -> str:
    """Issue a JWT for ``subject`` (user id string) with optional extra claims."""
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=_EXP_MIN),
    }
    if claims:
        payload.update(claims)
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode + verify a JWT. Raises jose.JWTError on failure."""
    return jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])


def extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


__all__ = [
    "ALGORITHM",
    "auth_configured",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_token",
    "extract_bearer",
]