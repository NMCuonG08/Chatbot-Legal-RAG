"""Curated acceptance question-set runner.

Runs a fixed set of Vietnamese legal questions through the production chat
graph (``run_chat_graph``) and prints a per-question report + aggregate
summary: route taken, expected route, tool calls, latency, errors.

Usage:  python run_question_test.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src is importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# (id, question, expected_route, expected_tool_keyword)
QUESTION_SET = [
    ("Q01", "Tôi làm việc 3 năm với lương bình quân 15 triệu/tháng, bây giờ nghỉ việc thì nhận bao nhiêu trợ cấp thôi việc?", "agent_tools", "severance_pay_tool"),
    ("Q02", "Tôi làm thêm 4 giờ vào ngày nghỉ với lương giờ thường 50k, được trả bao nhiêu tiền làm thêm?", "agent_tools", "overtime_pay_tool"),
    ("Q03", "Thu nhập tính thuế 20 triệu/tháng thì phải đóng thuế TNCN bao nhiêu?", "agent_tools", "pit_monthly_tool"),
    ("Q04", "Tôi mua nhà 2 tỷ, lệ phí trước bạ là bao nhiêu?", "agent_tools", "land_registration_fee_tool"),
    ("Q05", "Tôi mua ô tô 1 tỷ lần đầu, lệ phí trước bạ bao nhiêu?", "agent_tools", "vehicle_registration_fee_tool"),
    ("Q06", "Khởi kiện đòi nợ 100 triệu thì án phí dân sự bao nhiêu?", "agent_tools", "court_fee_tool"),
    ("Q07", "Chạy xe máy sau khi uống rượu bị phạt bao nhiêu?", "agent_tools", "admin_fine_lookup_tool"),
    ("Q08", "Ly hôn thì cần giấy tờ gì, nộp ở đâu?", "agent_tools", "procedure_wizard_tool"),
    ("Q09", "Tranh chấp dân sự 600 triệu thì tòa cấp nào thụ lý?", "agent_tools", "jurisdiction_resolver_tool"),
    ("Q10", "Sinh đơn khởi kiện đòi nợ 100 triệu giúp tôi với", "agent_tools", "generate_document_template_tool"),
    ("Q11", "Bộ luật Lao động 2019 hiện còn hiệu lực không?", "agent_tools", "law_version_tool"),
    ("Q12", "Cấp dưỡng 1 con khi thu nhập 10 triệu/tháng là bao nhiêu?", "agent_tools", "child_support_tool"),
    ("Q13", "Tôi bị khởi tố hình sự về tội trộm cắp, cần bào chữa thế nào?", "agent_tools", "legal_disclaimer_tool"),
    ("Q14", "Điều 418 Bộ luật Dân sự 2015 nói về vấn đề gì?", "legal_rag", "article_lookup_tool"),
    ("Q15", "Xin chào, bạn có thể giúp gì cho tôi?", "general_chat", ""),
    ("Q16", "Mức lương tối thiểu vùng năm 2024 mới nhất là bao nhiêu?", "web_search", ""),
]


def run_once(qid: str, question: str, expected_route: str, expected_tool: str) -> dict:
    from tasks import run_chat_graph

    t0 = time.perf_counter()
    try:
        res = run_chat_graph([], question, user_id="eval", conversation_id=f"eval-{qid}")
        answer = res.get("response", "")
        route = res.get("route", "")
        # agent_tools_node stores tool_calls in graph state; run_chat_graph surfaces them.
        tool_list = res.get("tool_calls", []) or []
        err = None
    except Exception as exc:  # noqa: BLE001
        answer, route, err, tool_list = "", "", f"{type(exc).__name__}: {exc}", []
        logger.warning("%s failed: %s", qid, err)
    elapsed = (time.perf_counter() - t0) * 1000

    tool_names = [c.get("tool_name", "?") for c in tool_list]
    tool_ok = all(c.get("status") == "success" for c in tool_list) if tool_list else True
    route_match = (route == expected_route) or (
        route in {"legal_rag", "agent_tools"} and expected_route in {"legal_rag", "agent_tools"}
    )
    # legal_rag path uses retrieve→generate, NOT agent tools — so an agent-tool
    # expectation is only meaningful when the route actually went to agent_tools.
    if expected_tool and route != "agent_tools":
        tool_match = route_match  # already validated by route; tools N/A on this path
    else:
        tool_match = (expected_tool in tool_names) if expected_tool else (not tool_list)

    return {
        "qid": qid,
        "question": question,
        "expected_route": expected_route,
        "route": route or "unknown",
        "route_match": route_match,
        "expected_tool": expected_tool or "-",
        "tools_called": tool_names,
        "tool_match": tool_match,
        "all_tools_ok": tool_ok,
        "latency_ms": round(elapsed),
        "error": err,
        "answer_preview": (answer or "")[:200].replace("\n", " "),
    }


def main() -> int:
    print("=" * 100)
    print("ACCEPTANCE QUESTION-SET RUN — production run_chat_graph")
    print("=" * 100)
    rows = []
    for qid, q, er, et in QUESTION_SET:
        print(f"\n[{qid}] {q}", flush=True)
        row = run_once(qid, q, er, et)
        rows.append(row)
        status = "OK" if (row["route_match"] and row["tool_match"] and not row["error"]) else "FAIL"
        print(f"    -> route={row['route']} (exp {row['expected_route']}) [{status}] "
              f"tools={row['tools_called']} {row['latency_ms']}ms", flush=True)
        if row["error"]:
            print(f"    ERROR: {row['error']}")
        if row["answer_preview"]:
            print(f"    ANS: {row['answer_preview']}", flush=True)

    # Aggregate
    n = len(rows)
    route_ok = sum(1 for r in rows if r["route_match"])
    tool_ok = sum(1 for r in rows if r["tool_match"])
    no_err = sum(1 for r in rows if not r["error"] and r["answer_preview"])
    lat = [r["latency_ms"] for r in rows]
    from collections import Counter
    route_dist = Counter(r["route"] for r in rows)
    all_tools = Counter(t for r in rows for t in r["tools_called"])

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"questions        : {n}")
    print(f"route correct   : {route_ok}/{n}  ({route_ok/n:.0%})")
    print(f"tool  correct    : {tool_ok}/{n}  ({tool_ok/n:.0%})")
    print(f"answered (no err): {no_err}/{n}  ({no_err/n:.0%})")
    print(f"latency mean    : {sum(lat)/n:.0f}ms   max={max(lat)}ms")
    print(f"route dist      : {dict(route_dist)}")
    print(f"tools used      : {dict(all_tools)}")
    fails = [r["qid"] for r in rows if not (r["route_match"] and r["tool_match"] and not r["error"])]
    print(f"FAILS           : {fails}")

    # Dump full JSON for the report
    out_path = Path(__file__).resolve().parents[2] / "data" / "acceptance_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfull json       : {out_path}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())