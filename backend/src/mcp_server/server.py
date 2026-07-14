"""FastMCP server exposing Vietnamese legal tools.

Tool schemas are derived by FastMCP from the function signatures + docstrings,
so each wrapper carries a Vietnamese docstring (that becomes the MCP tool
description shown to clients like Claude Desktop / the MCP inspector).

Wrappers call the raw ``Dict``-returning implementations in ``legal_tools``,
``legal_knowledge_tools``, ``legal_retrieval_tools``, ``legal_procedure_tools``,
and ``legal_graph_tools``. They deliberately do NOT reuse the LlamaIndex
``FunctionTool`` wrappers in ``agent_tool_wrappers`` (those already JSON-encode
into ``str``); calling them here would double-encode.

Retrieval/graph tools need live Qdrant/Neo4j. They are imported lazily inside
the wrapper so a missing store degrades to a per-call JSON error rather than
crashing the server at import time. Pure calc tools have no external deps.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

mcp = FastMCP("legal-tools-vn")


def _to_json(result: Any) -> str:
    """Serialize a tool result to a compact JSON string (MCP tools return text)."""
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"serialize_failed: {exc}"}, ensure_ascii=False)


def _err(msg: str, **extra: Any) -> str:
    payload = {"error": msg}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pure calc tools (no external deps) — legal_tools.py
# ---------------------------------------------------------------------------
@mcp.tool()
def contract_penalty_calculator(contract_value: float, penalty_rate: float, days_late: int) -> str:
    """Tính tiền phạt vi phạm hợp đồng theo Bộ luật Dân sự 2015 (Điều 418).

    Args:
        contract_value: Giá trị hợp đồng (VNĐ).
        penalty_rate: Tỷ lệ phạt (%/ngày, thường 0.05–0.3).
        days_late: Số ngày chậm trễ.
    """
    from legal_tools import calculate_contract_penalty

    return _to_json(calculate_contract_penalty(contract_value, penalty_rate, days_late))


@mcp.tool()
def legal_age_checker(birth_year: int, action_type: str = "sign_contract", gender: str = "") -> str:
    """Kiểm tra độ tuổi pháp lý để thực hiện hành vi (ký hợp đồng, kết hôn...).

    Args:
        birth_year: Năm sinh của người đó.
        action_type: Loại hành vi: sign_contract | marry | civil_act.
        gender: Giới tính (nam/nữ) — ảnh hưởng độ tuổi kết hôn.
    """
    from legal_tools import check_legal_entity_age

    return _to_json(check_legal_entity_age(birth_year, action_type, gender))


@mcp.tool()
def inheritance_calculator(total_value: float, heirs_json: str) -> str:
    """Tính phần thừa kế theo pháp luật Việt Nam.

    Args:
        total_value: Tổng giá trị khối di sản (VNĐ).
        heirs_json: JSON string — danh sách người thừa kế. Mỗi phần tử là
            dict có khóa "relation" (ví dụ "spouse", "child", "parent") và
            tuỳ chọn "count". Ví dụ: [{"relation": "spouse"}, {"relation": "child", "count": 2}].
    """
    from legal_tools import calculate_inheritance_share

    try:
        heirs = json.loads(heirs_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return _err(f"heirs_json không hợp lệ: {exc}")
    if not isinstance(heirs, list):
        return _err("heirs_json phải là một mảng (JSON array).")
    return _to_json(calculate_inheritance_share(total_value, heirs))


@mcp.tool()
def business_name_validator(business_name: str) -> str:
    """Kiểm tra tên doanh nghiệp có vi phạm quy định đặt tên hay không.

    Args:
        business_name: Tên doanh nghiệp cần kiểm tra.
    """
    from legal_tools import check_business_name_rules

    return _to_json(check_business_name_rules(business_name))


@mcp.tool()
def statute_lookup(case_type: str) -> str:
    """Tra cứu thời hiệu khởi kiện theo loại án.

    Args:
        case_type: Loại án (ví dụ: civil_first, civil_appellate, administrative, criminal).
    """
    from legal_tools import get_statute_of_limitations

    return _to_json(get_statute_of_limitations(case_type))


# ---------------------------------------------------------------------------
# Curated data tools — legal_knowledge_tools.py
# ---------------------------------------------------------------------------
@mcp.tool()
def severance_pay(monthly_salary: float, months_worked: int) -> str:
    """Tính trợ cấp thôi việc theo Bộ luật Lao động 2019.

    Args:
        monthly_salary: Mức lương tháng gần nhất (VNĐ).
        months_worked: Tổng số tháng làm việc thực tế.
    """
    from legal_knowledge_tools import calculate_severance_pay

    return _to_json(calculate_severance_pay(monthly_salary, months_worked))


@mcp.tool()
def overtime_pay(hourly_wage: float, hours: float, day_type: str = "weekday") -> str:
    """Tính tiền làm thêm giờ theo Bộ luật Lao động 2019.

    Args:
        hourly_wage: Mức lương giờ (VNĐ).
        hours: Số giờ làm thêm.
        day_type: Loại ngày: weekday | weekend | holiday.
    """
    from legal_knowledge_tools import calculate_overtime_pay

    return _to_json(calculate_overtime_pay(hourly_wage, hours, day_type))


@mcp.tool()
def pit_monthly(taxable_income: float) -> str:
    """Tính thuế thu nhập cá nhân (PIT) theo tháng theo Luật Thuế TNCN.

    Args:
        taxable_income: Thu nhập tính thuế tháng (VNĐ, sau khi trừ giảm trừ).
    """
    from legal_knowledge_tools import calculate_pit_monthly

    return _to_json(calculate_pit_monthly(taxable_income))


@mcp.tool()
def court_fee(claim_value: float, case_type: str = "civil_first") -> str:
    """Tính án phí sơ thẩm theo giá trị tranh chấp.

    Args:
        claim_value: Giá trị yêu cầu/vật chứng tranh chấp (VNĐ).
        case_type: Loại án: civil_first | civil_appellate | administrative.
    """
    from legal_knowledge_tools import calculate_court_fee

    return _to_json(calculate_court_fee(claim_value, case_type))


@mcp.tool()
def child_support(payer_income: float, num_children: int = 1) -> str:
    """Tính mức cấp dưỡng cho con theo Luật Hôn nhân & Gia đình.

    Args:
        payer_income: Thu nhập của người có nghĩa vụ cấp dưỡng (VNĐ/tháng).
        num_children: Số con được nhận cấp dưỡng.
    """
    from legal_knowledge_tools import calculate_child_support

    return _to_json(calculate_child_support(payer_income, num_children))


@mcp.tool()
def law_version(law_key: str, effective_year: int = 0) -> str:
    """Tra cứu phiên bản/hiệu lực của một văn bản luật.

    Args:
        law_key: Khóa văn bản (ví dụ: "BLDS", "BLLD").
        effective_year: Năm hiệu lực cần tra (0 = mặc định hiện hành).
    """
    from legal_knowledge_tools import get_law_version

    return _to_json(get_law_version(law_key, effective_year or None))


# ---------------------------------------------------------------------------
# Retrieval-backed tools (Qdrant) — legal_retrieval_tools.py — best-effort
# ---------------------------------------------------------------------------
@mcp.tool()
def article_lookup(law_name: str, article_number: int = 0, limit: int = 5) -> str:
    """Tra cứu điều luật theo tên luật (và tuỳ chọn số điều) qua Qdrant.

    Args:
        law_name: Tên luật (ví dụ "Bộ luật Dân sự").
        article_number: Số điều cần tra (0 = không lọc theo số).
        limit: Số kết quả tối đa.
    """
    try:
        from legal_retrieval_tools import lookup_article
    except Exception as exc:  # import-time deps missing
        return _err("lookup_article_unavailable", detail=str(exc))
    try:
        return _to_json(lookup_article(law_name, article_number or None, limit))
    except Exception as exc:
        return _err("article_lookup_failed", detail=str(exc))


@mcp.tool()
def precedent_lookup(fact_pattern: str, limit: int = 5) -> str:
    """Tra cứu án lệ/tiền lệ theo mô tả tình tiết sự việc.

    Args:
        fact_pattern: Mô tả tình tiết sự việc cần tìm án lệ.
        limit: Số kết quả tối đa.
    """
    try:
        from legal_retrieval_tools import precedent_lookup as _fn
    except Exception as exc:
        return _err("precedent_lookup_unavailable", detail=str(exc))
    try:
        return _to_json(_fn(fact_pattern, limit))
    except Exception as exc:
        return _err("precedent_lookup_failed", detail=str(exc))


@mcp.tool()
def cross_reference(law_name: str, article_number: int, limit: int = 5) -> str:
    """Tìm các điều luật dẫn chiếu tới điều luật đã cho.

    Args:
        law_name: Tên luật chứa điều cần tra.
        article_number: Số điều cần tìm dẫn chiếu.
        limit: Số kết quả tối đa.
    """
    try:
        from legal_retrieval_tools import cross_reference as _fn
    except Exception as exc:
        return _err("cross_reference_unavailable", detail=str(exc))
    try:
        return _to_json(_fn(law_name, article_number, limit))
    except Exception as exc:
        return _err("cross_reference_failed", detail=str(exc))


@mcp.tool()
def verify_citation(law_name: str, article_number: int, claimed_text: str) -> str:
    """Kiểm tra một đoạn trích dẫn có khớp với văn bản luật thật hay không.

    Args:
        law_name: Tên luật được trích dẫn.
        article_number: Số điều được trích dẫn.
        claimed_text: Đoạn văn bản được tuyên bố là trích từ điều luật.
    """
    try:
        from legal_retrieval_tools import verify_citation as _fn
    except Exception as exc:
        return _err("verify_citation_unavailable", detail=str(exc))
    try:
        return _to_json(_fn(law_name, article_number, claimed_text))
    except Exception as exc:
        return _err("verify_citation_failed", detail=str(exc))


# ---------------------------------------------------------------------------
# Procedure tools — legal_procedure_tools.py
# ---------------------------------------------------------------------------
@mcp.tool()
def procedure_wizard(procedure_type: str) -> str:
    """Trả về các bước của một thủ tục pháp lý cụ thể.

    Args:
        procedure_type: Mã thủ tục (ví dụ: "land_registration", "business_registration").
    """
    from legal_procedure_tools import procedure_wizard as _fn

    return _to_json(_fn(procedure_type))


@mcp.tool()
def jurisdiction_resolver(dispute_type: str, claim_value: float = 0, location: str = "") -> str:
    """Xác định cơ quan/thẩm quyền giải quyết tranh chấp.

    Args:
        dispute_type: Loại tranh chấp (civil | administrative | commercial ...).
        claim_value: Giá trị tranh chấp (VNĐ), 0 nếu không xác định.
        location: Địa bàn (tỉnh/thành) để xác định thẩm quyền theo lãnh thổ.
    """
    from legal_procedure_tools import jurisdiction_resolver as _fn

    return _to_json(_fn(dispute_type, claim_value or None, location))


# ---------------------------------------------------------------------------
# Neo4j graph tool — legal_graph_tools.py — best-effort (additive)
# ---------------------------------------------------------------------------
@mcp.tool()
def recall_legal_graph(query: str, limit: int = 5) -> str:
    """Truy hồi đa bước trên đồ thị tri thức pháp luật (Neo4j): dẫn chiếu, hiệu lực.

    Args:
        query: Câu hỏi dạng tự do (ví dụ "Điều 35 BLDS dẫn chiếu điều nào?").
        limit: Số nút Article trả về tối đa.
    """
    try:
        from legal_graph_tools import recall_legal_graph as _fn
    except Exception as exc:
        return _err("legal_graph_unavailable", detail=str(exc))
    try:
        return _to_json(_fn(query, limit))
    except Exception as exc:
        return _err("recall_legal_graph_failed", detail=str(exc))


if __name__ == "__main__":  # pragma: no cover
    mcp.run()