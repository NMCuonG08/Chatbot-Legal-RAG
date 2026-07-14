"""CLI entrypoint for the legal-tools MCP server.

Usage:
    python -m src.mcp_server --transport stdio
    python -m src.mcp_server --transport http --host 0.0.0.0 --port 8100

- stdio : local tool transport (Claude Desktop / Claude Code).
- http  : streamable-http (remote/production, runs an HTTP server).
"""
from __future__ import annotations

import argparse
import os

from .server import mcp


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


def main() -> None:
    args = _parse_args()
    if args.transport == "stdio":
        mcp.run()  # defaults to stdio
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()