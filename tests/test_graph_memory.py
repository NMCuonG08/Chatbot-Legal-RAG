"""Phase 3 unit tests — Neo4j legal knowledge graph (ingest + recall tool).

Uses the ``set_graph_client`` test seam with a fake driver so no live Neo4j
is required. Verifies:
- ``add_to_graph`` MERGE behavior + skip-when-metadata-incomplete.
- Best-effort: down/absent graph never raises (regression guard).
- ``recall_legal_graph`` parsing + traversal + empty-fallback.
- Tool registration surface (FunctionTool present in all_tools).
"""
import json

import agent_tool_wrappers
import graph_db
import legal_graph_ingest
import legal_graph_tools
from llama_index.core.tools import FunctionTool


# ---- fake driver ----

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self._state["writes"].append((cypher, params))
        # Return canned rows for read queries matching the law/article.
        law = params.get("law", "").lower()
        art = params.get("art")
        rows = []
        for r in self._state.get("rows", []):
            if law and law in r.get("law", "").lower():
                if art is None or r.get("number") == art:
                    rows.append(r)
        return _FakeResult(rows)


class _FakeDriver:
    def __init__(self):
        self.state = {"writes": [], "rows": []}

    def session(self):
        return _FakeSession(self.state)


# ---- ingest ----

def test_add_to_graph_merges_statute_article():
    drv = _FakeDriver()
    graph_db.set_graph_client(drv)
    try:
        meta = {"law_name": "Bộ luật Dân sự 2015", "article_number": 35}
        ok = legal_graph_ingest.add_to_graph("cid-1", "Điều 35 quy định...", meta)
        assert ok is True
        assert len(drv.state["writes"]) == 1
        cypher, params = drv.state["writes"][0]
        assert "MERGE (st:Statute" in cypher
        assert "MERGE (a:Article" in cypher
        assert "HAS_ARTICLE" in cypher
        assert params["law"] == "Bộ luật Dân sự 2015"
        assert params["art"] == 35
        assert params["year"] == 2015
        assert params["text"] == "Điều 35 quy định..."
    finally:
        graph_db.set_graph_client(None)


def test_add_to_graph_skips_when_metadata_incomplete():
    drv = _FakeDriver()
    graph_db.set_graph_client(drv)
    try:
        assert legal_graph_ingest.add_to_graph("c", "text", {}) is False
        assert legal_graph_ingest.add_to_graph("c", "text", {"law_name": "Luật X"}) is False
        assert legal_graph_ingest.add_to_graph("c", "text", {"article_number": 5}) is False
        assert drv.state["writes"] == []   # nothing attempted
    finally:
        graph_db.set_graph_client(None)


def test_add_to_graph_truncates_long_text():
    drv = _FakeDriver()
    graph_db.set_graph_client(drv)
    try:
        long_text = "x" * 5000
        legal_graph_ingest.add_to_graph("c", long_text, {"law_name": "Luật Y 2020", "article_number": 1})
        _, params = drv.state["writes"][0]
        assert len(params["text"]) == 2000   # _MAX_ARTICLE_TEXT
    finally:
        graph_db.set_graph_client(None)


def test_add_to_graph_down_graph_never_raises():
    graph_db.set_graph_client(None)   # disabled
    # No driver -> False, no exception.
    ok = legal_graph_ingest.add_to_graph("c", "text", {"law_name": "Luật Z", "article_number": 9})
    assert ok is False


def test_add_to_graph_driver_error_swallowed():
    class BoomDriver:
        def session(self):
            raise ConnectionError("neo4j down")
    graph_db.set_graph_client(BoomDriver())
    try:
        ok = legal_graph_ingest.add_to_graph("c", "text", {"law_name": "Luật A", "article_number": 2})
        assert ok is False   # swallowed, not raised
    finally:
        graph_db.set_graph_client(None)


def test_add_batch_to_graph_counts_writes():
    drv = _FakeDriver()
    graph_db.set_graph_client(drv)
    try:
        drv.state["rows"] = []  # writes only
        items = [
            ("c1", "t1", {"law_name": "Luật A 2020", "article_number": 1}),
            ("c2", "t2", {"law_name": "Luật B 2020", "article_number": 2}),
            ("c3", "t3", {"law_name": "Luật C"}),   # incomplete -> skipped
        ]
        written = legal_graph_ingest.add_batch_to_graph(items)
        assert written == 2
        assert len(drv.state["writes"]) == 2
    finally:
        graph_db.set_graph_client(None)


# ---- recall tool ----

def test_recall_legal_graph_parses_and_traverses():
    drv = _FakeDriver()
    drv.state["rows"] = [
        {"number": 35, "text": "nội dung điều 35", "law": "Bộ luật Dân sự 2015"},
        {"number": 36, "text": "nội dung điều 36", "law": "Bộ luật Dân sự 2015"},
        {"number": 35, "text": "khác", "law": "Luật Khác 2010"},
    ]
    graph_db.set_graph_client(drv)
    try:
        result = legal_graph_tools.recall_legal_graph("Điều 35 Bộ luật Dân sự nói gì?", limit=5)
        assert result["law"] == "Bộ luật Dân sự"
        assert result["article"] == 35
        nums = [a["number"] for a in result["articles"]]
        assert 35 in nums
        # Article-specific query filters to that article within matching law.
        assert all(a["law"] == "Bộ luật Dân sự 2015" for a in result["articles"])
    finally:
        graph_db.set_graph_client(None)


def test_recall_legal_graph_no_law_returns_empty():
    graph_db.set_graph_client(_FakeDriver())
    try:
        result = legal_graph_tools.recall_legal_graph("chào bạn thế nào?", limit=5)
        assert result["articles"] == []
        assert result["law"] == ""
    finally:
        graph_db.set_graph_client(None)


def test_recall_legal_graph_down_graph_returns_empty():
    graph_db.set_graph_client(None)
    result = legal_graph_tools.recall_legal_graph("Điều 35 Bộ luật Dân sự", limit=5)
    assert result["articles"] == []   # never raises


def test_recall_legal_graph_tool_serializes_json():
    drv = _FakeDriver()
    drv.state["rows"] = [{"number": 35, "text": "điều 35", "law": "Bộ luật Dân sự 2015"}]
    graph_db.set_graph_client(drv)
    try:
        out = legal_graph_tools.recall_legal_graph_tool("Điều 35 Bộ luật Dân sự", limit=5)
        parsed = json.loads(out)
        assert "articles" in parsed
        assert parsed["articles"][0]["number"] == 35
    finally:
        graph_db.set_graph_client(None)


def test_recall_legal_graph_tool_error_returns_json_error():
    class BoomDriver:
        def session(self):
            raise ConnectionError("down")
    graph_db.set_graph_client(BoomDriver())
    try:
        out = legal_graph_tools.recall_legal_graph_tool("Điều 35 Bộ luật Dân sự", limit=5)
        parsed = json.loads(out)
        # execute_read swallows -> empty articles (not error key); but if any
        # other path raises, the wrapper returns {"error": ...}. Accept either
        # graceful empty result or error envelope — both are non-raising.
        assert "articles" in parsed
    finally:
        graph_db.set_graph_client(None)


# ---- registration surface ----

def test_recall_legal_graph_func_tool_registered():
    tool = getattr(agent_tool_wrappers, "recall_legal_graph_func_tool", None)
    assert tool is not None
    assert isinstance(tool, FunctionTool)
    assert tool in agent_tool_wrappers.all_tools