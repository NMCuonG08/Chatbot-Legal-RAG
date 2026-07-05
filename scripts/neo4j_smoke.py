"""Phase 3 Neo4j smoke — ingest sample statute chunks, recall via graph tool.

Run from backend/src so `graph_db` / `legal_graph_*` / `legal_metadata` import.
Requires NEO4J_URI/USER/PASSWORD env and a live Neo4j on bolt://localhost:7687.
"""
import json
import os
import sys

# Backend src on path (run: python ../../scripts/neo4j_smoke.py from backend/src,
# or set PYTHONPATH). Simplest: insert backend/src.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "backend", "src"))
sys.path.insert(0, SRC)

from legal_metadata import extract_legal_metadata          # noqa: E402
from legal_graph_ingest import add_to_graph, add_batch_to_graph  # noqa: E402
from legal_graph_tools import recall_legal_graph, recall_legal_graph_tool  # noqa: E402
from graph_db import get_graph_client, execute_read       # noqa: E402

# 5 sample chunks: 3 articles of BLDS 2015 + 2 of NĐ 10/2022.
SAMPLES = [
    ("c-1", "Điều 35 Bộ luật Dân sự 2015 quy định quyền cá nhân đối với hình ảnh. "
            "Mỗi cá nhân có quyền đối với hình ảnh của mình."),
    ("c-2", "Điều 418 Bộ luật Dân sự 2015 quy định thời hiệu khởi kiện. Thời hiệu "
            "lao động là 03 năm kể từ ngày quyền bị xâm phạm."),
    ("c-3", "Điều 133 Bộ luật Dân sự 2015 quy định điều kiện có hiệu lực của giao "
            "dịch dân sự. Giao dịch có hiệu lực khi đương sự có năng lực hành vi."),
    ("c-4", "Theo Nghị định 10/2022, lệ phí trước bạ đối với hợp đồng tặng cho "
            "bất động sản được quy định tại điều 7."),
    ("c-5", "Điều 7 Nghị định 10/2022 quy định mức thu lệ phí trước bạ."),
]


def main():
    print("== Neo4j smoke ==")
    print("driver:", get_graph_client())

    # 1. extract metadata + ingest
    print("\n[1] Ingest 5 sample chunks (extract_legal_metadata -> add_to_graph):")
    items = []
    for cid, text in SAMPLES:
        meta = extract_legal_metadata(text)
        print(f"  {cid}: meta={meta}")
        items.append((cid, text, meta))
    written = add_batch_to_graph(items)
    print(f"  written={written} (idempotent MERGE)")

    # 2. re-ingest -> idempotent (no dup)
    print("\n[2] Re-ingest same batch (idempotency check):")
    written2 = add_batch_to_graph(items)
    print(f"  written={written2}")

    # 3. raw counts
    print("\n[3] Raw counts:")
    n_statutes = execute_read("MATCH (s:Statute) RETURN count(s) AS c")[0]["c"]
    n_articles = execute_read("MATCH (a:Article) RETURN count(a) AS c")[0]["c"]
    n_edges = execute_read("MATCH ()-[r:HAS_ARTICLE]->() RETURN count(r) AS c")[0]["c"]
    print(f"  Statutes={n_statutes}  Articles={n_articles}  HAS_ARTICLE edges={n_edges}")

    # 4. recall by law + article (multi-hop)
    print("\n[4] recall_legal_graph('Điều 35 Bộ luật Dân sự nói gì?'):")
    res = recall_legal_graph("Điều 35 Bộ luật Dân sự nói gì?")
    print("  parsed law =", repr(res["law"]), " article =", res["article"])
    for a in res["articles"]:
        print(f"  -> Điều {a['number']} ({a['law']}): {a['text'][:80]}...")

    # 5. recall by law only (enumerate articles of a statute)
    print("\n[5] recall_legal_graph('Bộ luật Dân sự có các điều nào?'):")
    res2 = recall_legal_graph("Bộ luật Dân sự có các điều nào?")
    print("  parsed law =", repr(res2["law"]), " article =", res2["article"])
    print(f"  returned {len(res2['articles'])} articles: "
          f"{[a['number'] for a in res2['articles']]}")

    # 6. agent-facing JSON tool
    print("\n[6] recall_legal_graph_tool JSON (first 400 chars):")
    print("  " + recall_legal_graph_tool("Điều 418 Bộ luật Dân sự 2015")[:400])

    print("\n== SMOKE OK ==")


if __name__ == "__main__":
    main()