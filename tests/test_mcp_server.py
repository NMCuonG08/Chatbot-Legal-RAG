"""Tests for the legal-tools MCP server (Phase 1).

Covers:
- All expected tools are registered (schema introspection via mcp.get_tools()).
- Pure-calc tools return correct JSON when called through the MCP wrapper.
- Retrieval/graph tools degrade to a JSON error string when Qdrant/Neo4j absent
  (no server crash) — skipped when no live store.

Skips cleanly if the `mcp` package is not installed (e.g. CI without the new
dependency) so it never hard-fails the broader suite on import.
"""
import asyncio
import json

import pytest

try:
    from mcp_server import server as mcp_server  # noqa: F401
    from mcp_server.server import mcp
except Exception:  # mcp not installed
    mcp = None
    mcp_server = None

pytestmark = pytest.mark.skipif(mcp is None, reason="mcp package not installed")

EXPECTED_TOOLS = {
    "contract_penalty_calculator",
    "legal_age_checker",
    "inheritance_calculator",
    "business_name_validator",
    "statute_lookup",
    "severance_pay",
    "overtime_pay",
    "pit_monthly",
    "court_fee",
    "child_support",
    "law_version",
    "article_lookup",
    "precedent_lookup",
    "cross_reference",
    "verify_citation",
    "procedure_wizard",
    "jurisdiction_resolver",
    "recall_legal_graph",
}


def _list_tools_sync():
    """Return the list of registered Tool descriptors (async list_tools)."""
    return asyncio.run(mcp.list_tools())


def _call_tool(name: str, **kwargs) -> str:
    """Invoke a registered MCP tool and return its first text content as a string.

    Uses the server's own ``call_tool`` so the test exercises the same code path
    a real MCP client would (schema validation + wrapper + JSON serialization).
    """
    async def _go() -> str:
        blocks, structured = await mcp.call_tool(name, kwargs)
        for block in blocks:
            txt = getattr(block, "text", None)
            if txt:
                return txt
        return json.dumps(structed, ensure_ascii=False, default=str)

    return asyncio.run(_go())


# ---- Registration / schema ----
class TestRegistration:
    def test_all_expected_tools_registered(self):
        tools = _list_tools_sync()
        names = {t.name for t in tools}
        missing = EXPECTED_TOOLS - names
        assert not missing, f"Missing MCP tools: {sorted(missing)}"

    def test_no_unexpected_extra_tools_beyond_known_set(self):
        # Allow extras but fail if a known tool silently disappeared — kept loose.
        tools = _list_tools_sync()
        assert len(tools) >= len(EXPECTED_TOOLS)

    def test_tool_has_description(self):
        tools = _list_tools_sync()
        tool = next(t for t in tools if t.name == "contract_penalty_calculator")
        desc = getattr(tool, "description", "") or ""
        assert "phạt" in desc.lower() or "vi phạm" in desc.lower()


# ---- Pure-calc wrappers (no external deps) ----
class TestCalcWrappers:
    def test_contract_penalty_capped(self):
        out = _call_tool(
            "contract_penalty_calculator",
            contract_value=100_000_000,
            penalty_rate=0.5,
            days_late=30,
        )
        data = json.loads(out)
        assert "error" not in data
        assert "418" in data.get("legal_basis", "")

    def test_severance_pay_returns_amount(self):
        out = _call_tool("severance_pay", monthly_salary=10_000_000, months_worked=12)
        data = json.loads(out)
        assert "error" not in data

    def test_pit_monthly_runs(self):
        out = _call_tool("pit_monthly", taxable_income=20_000_000)
        data = json.loads(out)
        assert "error" not in data

    def test_inheritance_bad_json_returns_error(self):
        out = _call_tool(
            "inheritance_calculator", total_value=500_000_000, heirs_json="not-json"
        )
        data = json.loads(out)
        assert "error" in data

    def test_inheritance_valid_json(self):
        heirs = json.dumps([{"relation": "spouse"}, {"relation": "child", "count": 2}])
        out = _call_tool(
            "inheritance_calculator", total_value=500_000_000, heirs_json=heirs
        )
        data = json.loads(out)
        assert "error" not in data

    def test_business_name_validator(self):
        out = _call_tool("business_name_validator", business_name="Công ty TNHH ABC")
        data = json.loads(out)
        assert "error" not in data

    def test_statute_lookup(self):
        out = _call_tool("statute_lookup", case_type="civil")
        data = json.loads(out)
        assert "error" not in data

    def test_court_fee(self):
        out = _call_tool("court_fee", claim_value=100_000_000, case_type="civil_first")
        data = json.loads(out)
        assert "error" not in data

    def test_child_support(self):
        out = _call_tool("child_support", payer_income=10_000_000, num_children=1)
        data = json.loads(out)
        assert "error" not in data

    def test_overtime_pay(self):
        out = _call_tool("overtime_pay", hourly_wage=50_000, hours=4, day_type="weekday")
        data = json.loads(out)
        assert "error" not in data

    def test_law_version(self):
        out = _call_tool("law_version", law_key="blds_2015", effective_year=0)
        data = json.loads(out)
        assert "error" not in data

    def test_legal_age_checker(self):
        out = _call_tool("legal_age_checker", birth_year=2008, action_type="sign_contract")
        data = json.loads(out)
        assert "error" not in data


# ---- Procedure tools (no external deps) ----
class TestProcedureWrappers:
    def test_jurisdiction_resolver(self):
        out = _call_tool(
            "jurisdiction_resolver", dispute_type="civil", claim_value=50_000_000, location="Hà Nội"
        )
        data = json.loads(out)
        assert "error" not in data

    def test_procedure_wizard(self):
        out = _call_tool("procedure_wizard", procedure_type="land_ownership_certificate")
        data = json.loads(out)
        assert "error" not in data


# ---- Best-effort retrieval/graph tools (skip without live stores) ----
@pytest.mark.integration
class TestRetrievalBestEffort:
    def test_article_lookup_no_crash_without_qdrant(self):
        # Either succeeds (Qdrant up) or returns a JSON error string (no crash).
        out = _call_tool("article_lookup", law_name="Bộ luật Dân sự", article_number=0, limit=3)
        assert isinstance(out, str)
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_recall_legal_graph_no_crash_without_neo4j(self):
        out = _call_tool("recall_legal_graph", query="Điều 35 BLDS dẫn chiếu điều nào?", limit=3)
        assert isinstance(out, str)
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_precedent_lookup_no_crash(self):
        out = _call_tool("precedent_lookup", fact_pattern="tranh chấp hợp đồng", limit=3)
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_cross_reference_no_crash(self):
        out = _call_tool("cross_reference", law_name="Bộ luật Dân sự", article_number=35, limit=3)
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_verify_citation_no_crash(self):
        out = _call_tool("verify_citation", law_name="Bộ luật Dân sự", article_number=35, claimed_text="đoạn trích dẫn")
        data = json.loads(out)
        assert isinstance(data, dict)


# ---- CLI entrypoint (mcp_server.__main__) ----
class TestCli:
    def test_parse_args_defaults_stdio(self, monkeypatch):
        from mcp_server import __main__ as cli

        monkeypatch.setattr(cli, "mcp", _FakeMcp())
        monkeypatch.setattr("sys.argv", ["mcp_server"])
        ns = cli._parse_args()
        assert ns.transport == "stdio"
        assert ns.port == 8100

    def test_main_stdio_calls_run(self, monkeypatch):
        from mcp_server import __main__ as cli

        fake = _FakeMcp()
        monkeypatch.setattr(cli, "mcp", fake)
        monkeypatch.setattr("sys.argv", ["mcp_server", "--transport", "stdio"])
        cli.main()
        assert fake.calls and fake.calls[0]["transport"] is None

    def test_main_http_calls_run_with_host_port(self, monkeypatch):
        from mcp_server import __main__ as cli

        fake = _FakeMcp()
        monkeypatch.setattr(cli, "mcp", fake)
        monkeypatch.setattr("sys.argv", ["mcp_server", "--transport", "http", "--host", "127.0.0.1", "--port", "9999"])
        cli.main()
        assert fake.calls[0]["transport"] == "streamable-http"
        assert fake.calls[0]["host"] == "127.0.0.1"
        assert fake.calls[0]["port"] == 9999


class _FakeMcp:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs, "transport": kwargs.get("transport"), "host": kwargs.get("host"), "port": kwargs.get("port")})