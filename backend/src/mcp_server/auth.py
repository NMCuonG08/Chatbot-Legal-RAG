"""Static bearer-key auth for the HTTP (streamable-http) MCP transport.

One shared key in ``MCP_API_KEY`` authorizes every client — the simplest
scheme that survives deployment (set one env var, hand the key out). stdio
transport skips auth entirely (local process, no network surface).

Env:
- MCP_API_KEY      : required when serving HTTP. The single shared key.
                     When unset, HTTP serving refuses unless
                     MCP_ALLOW_NO_AUTH=1 (dev only — logs a loud warning).
- MCP_ALLOW_NO_AUTH: when "1", serve HTTP with no key check (dev only).

The middleware checks ``Authorization: Bearer <key>`` against a
``secrets.compare_digest`` constant-time compare and returns 401 JSON on
mismatch/missing. It deliberately does NOT touch auth.py/JWT — this is a
transport-level gate, orthogonal to per-user app auth.
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

import json
import logging
import os
import secrets
from typing import Optional

logger = logging.getLogger(__name__)

_UNSET_WARNING = (
    "MCP_API_KEY chưa cấu hình — HTTP server đang chạy KHÔNG auth. "
    "Chỉ dùng cho dev. Set MCP_API_KEY cho production."
)

_warned = False


def _configured_key() -> Optional[str]:
    return os.getenv("MCP_API_KEY") or None


def auth_required() -> bool:
    """True when HTTP serving must enforce the bearer key."""
    if _configured_key():
        return True
    return os.getenv("MCP_ALLOW_NO_AUTH", "0") != "1"


class BearerAuthMiddleware:
    """Reject requests whose Bearer token != MCP_API_KEY (constant-time).

    Implemented as a pure ASGI middleware to support SSE / streaming responses
    without triggering Starlette BaseHTTPMiddleware bugs.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _warned
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        key = _configured_key()
        if not key:
            # No key configured. Refuse unless dev bypass is explicit.
            if os.getenv("MCP_ALLOW_NO_AUTH", "0") == "1":
                if not _warned:
                    logger.warning(_UNSET_WARNING)
                    _warned = True
                await self.app(scope, receive, send)
                return
            await self._send_unauthorized(send, "server_auth_not_configured: set MCP_API_KEY")
            return

        # Extract authorization header
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("utf-8")

        token = _extract_bearer(auth_header)
        if token is None:
            await self._send_unauthorized(send, "missing_bearer_token")
            return
        if not secrets.compare_digest(token, key):
            await self._send_unauthorized(send, "invalid_bearer_token")
            return

        await self.app(scope, receive, send)

    async def _send_unauthorized(self, send, detail: str):
        body = json.dumps({"error": "unauthorized", "detail": detail}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


__all__ = ["BearerAuthMiddleware", "auth_required", "_configured_key"]