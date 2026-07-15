"""CLI entrypoint for the legal-tools MCP server.

Usage:
    python -m src.mcp_server --transport stdio
    python -m src.mcp_server --transport http --host 0.0.0.0 --port 8100

- stdio : local tool transport (Claude Desktop / Claude Code). No auth.
- http  : streamable-http (remote/production). Bearer-key gated by
          MCP_API_KEY (see mcp_server.auth). stdio skips auth entirely.
"""
from __future__ import annotations

import argparse
import logging
import os

from .auth import BearerAuthMiddleware
from .server import mcp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src.mcp_server", description="Legal tools MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport: stdio (default) or http (streamable-http).",
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"), help="HTTP host (http only).")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_HTTP_PORT", "8100")),
        help="HTTP port (http only). Default 8100 / MCP_HTTP_PORT env.",
    )
    return parser.parse_args()


def _build_http_app():
    """Wrap the FastMCP streamable-http app with the bearer-key middleware.

    Returns the ASGI app suitable for ``uvicorn.run(...)``. Auth is
    enforced on every HTTP request via BearerAuthMiddleware; stdio is unaffected.
    """
    mcp_app = mcp.streamable_http_app()
    return BearerAuthMiddleware(mcp_app)


def main() -> None:
    args = _parse_args()
    if args.transport == "stdio":
        mcp.run()  # defaults to stdio; no auth (local process)
        return

    # HTTP: serve the auth-wrapped Starlette app via uvicorn.
    import uvicorn

    if not os.getenv("MCP_API_KEY") and os.getenv("MCP_ALLOW_NO_AUTH", "0") != "1":
        raise RuntimeError(
            "MCP_API_KEY chưa cấu hình — refuse to serve HTTP unauthenticated. "
            "Set MCP_API_KEY, hoặc MCP_ALLOW_NO_AUTH=1 cho dev."
        )
    if not os.getenv("MCP_API_KEY"):
        logger.warning("Serving HTTP with NO auth (MCP_ALLOW_NO_AUTH=1) — dev only.")
    app = _build_http_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()