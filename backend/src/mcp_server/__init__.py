"""MCP server exposing Vietnamese legal tools.

Two transports:
- stdio  : `python -m mcp_server --transport stdio` (run from backend/src)
- http   : `python -m mcp_server --transport http --port 8100` (streamable-http)

Reuses the raw (Dict-returning) implementations in the legal_* modules; does
NOT wrap the LlamaIndex FunctionTool wrappers (those return JSON strings and
would double-encode). Retrieval/graph tools are best-effort: if Qdrant/Neo4j
is unavailable the tool returns a JSON error instead of crashing the server.
"""
from .server import mcp  # noqa: F401  re-export for `mcp dev mcp_server.server:mcp`

__all__ = ["mcp"]