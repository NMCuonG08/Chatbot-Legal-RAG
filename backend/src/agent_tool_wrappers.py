"""Tool wrappers + FunctionTool instances for the ReAct agent.

Each wrapper fn validates inputs, calls the underlying legal/search util, and
returns a JSON string. ``@track_tool_call`` records each call into the
``agent_tool_calls`` contextvar for the current run.

Extracted from ``agent.py`` to keep ``agent.py`` focused on agent runtime
(LLM, memory, prompt, execution). ``agent.py`` imports ``all_tools`` from here.
"""

import json
import logging

from llama_index.core.tools import FunctionTool

from agent_tool_tracking import track_tool_call
from legal_tools import (
    calculate_contract_penalty,
    calculate_inheritance_share,
    check_business_name_rules,
    check_legal_entity_age,
    get_statute_of_limitations,
)
from legal_retrieval_tools import (
    cross_reference,
    lookup_article,
    precedent_lookup,
    verify_citation,
)
from legal_knowledge_tools import (
    calculate_child_support,
    calculate_court_fee,
    calculate_land_registration_fee,
    calculate_overtime_pay,
    calculate_pit_monthly,
    calculate_severance_pay,
    calculate_vehicle_registration_fee,
    get_law_version,
    legal_disclaimer_check,
    lookup_administrative_fine,
)
from legal_procedure_tools import (
    generate_document_template,
    jurisdiction_resolver,
    procedure_wizard,
)
from search import search_engine
from tavily_tool import tavily_qna, tavily_search_legal

logger = logging.getLogger(__name__)


# ===== LEGAL CALCULATION TOOLS =====


@track_tool_call
def contract_penalty_calculator(
    contract_value: float, penalty_rate: float, days_late: int
) -> str:
    """
    Tính tiền phạt vi phạm hợp đồng theo Bộ luật Dân sự Việt Nam.

    Args:
        contract_value: Giá trị hợp đồng (VNĐ), ví dụ: 100000000 (100 triệu)
        penalty_rate: Tỷ lệ phạt theo hợp đồng (% mỗi ngày), ví dụ: 0.1 (0.1%/ngày)
        days_late: Số ngày chậm trễ, ví dụ: 30

    Returns:
        Kết quả tính toán tiền phạt chi tiết
    """
    # Tool Guardrail: Validate value range to prevent negative values or massive integers
    if contract_value <= 0 or contract_value > 10**12:
        return json.dumps({"error": "Giá trị hợp đồng không hợp lệ (phải từ 0 đến 1,000 tỷ VNĐ)"}, ensure_ascii=False)
    if penalty_rate < 0 or penalty_rate > 100:
        return json.dumps({"error": "Tỷ lệ phạt không hợp lệ (phải từ 0% đến 100%)"}, ensure_ascii=False)
    if days_late < 0 or days_late > 3650:
        return json.dumps({"error": "Số ngày chậm trễ không hợp lệ (phải từ 0 đến 3650 ngày - 10 năm)"}, ensure_ascii=False)

    result = calculate_contract_penalty(contract_value, penalty_rate, days_late)
    return json.dumps(result, ensure_ascii=False, indent=2)


@track_tool_call
def legal_age_checker(
    birth_year: int, action_type: str = "sign_contract", gender: str = ""
) -> str:
    """
    Kiểm tra tuổi pháp lý để thực hiện hành vi dân sự.

    Args:
        birth_year: Năm sinh, ví dụ: 2005
        action_type: Loại hành vi, có thể là: "sign_contract" (ký hợp đồng), "marriage" (kết hôn), "work" (làm việc), "criminal_responsibility" (chịu trách nhiệm hình sự)
        gender: Giới tính "male" hoặc "female". BẮT BUỘC khi action_type="marriage" vì nam phải đủ 20 tuổi, nữ phải đủ 18 tuổi (Điều 8 Luật HNGĐ 2014).

    Returns:
        Thông tin về khả năng pháp lý và căn cứ pháp luật
    """
    # Tool Guardrail: Validate parameters
    current_year = 2026
    if birth_year < 1900 or birth_year > current_year:
        return json.dumps({"error": f"Năm sinh không hợp lệ (phải từ 1900 đến {current_year})"}, ensure_ascii=False)

    valid_actions = ["sign_contract", "marriage", "work", "criminal_responsibility"]
    if action_type not in valid_actions:
        return json.dumps({"error": f"Loại hành vi không hợp lệ. Các loại hợp lệ: {', '.join(valid_actions)}"}, ensure_ascii=False)

    if action_type == "marriage":
        gender_norm = (gender or "").strip().lower()
        if gender_norm not in ("male", "female"):
            return json.dumps(
                {"error": "Khi kiểm tra tuổi kết hôn cần gender='male' hoặc gender='female' vì nam đủ 20, nữ đủ 18 tuổi (Điều 8 Luật HNGĐ 2014)."},
                ensure_ascii=False,
            )

    result = check_legal_entity_age(birth_year, action_type, gender=gender)
    return json.dumps(result, ensure_ascii=False, indent=2)


@track_tool_call
def inheritance_calculator(total_value: float, heirs_json: str) -> str:
    """
    Tính phần thừa kế theo pháp luật Việt Nam (hàng thừa kế thứ nhất).

    Args:
        total_value: Tổng giá trị tài sản thừa kế (VNĐ), ví dụ: 500000000 (500 triệu)
        heirs_json: Danh sách người thừa kế dạng JSON string, ví dụ: '[{"name":"Nguyễn Văn A","relation":"child","is_minor":false},{"name":"Trần Thị B","relation":"spouse","is_minor":false}]'

    Returns:
        Phân chia tài sản thừa kế cho từng người
    """
    # Tool Guardrail: Validate range
    if total_value <= 0 or total_value > 10**12:
        return json.dumps({"error": "Tổng giá trị tài sản không hợp lệ (phải từ 0 đến 1,000 tỷ VNĐ)"}, ensure_ascii=False)

    try:
        heirs = json.loads(heirs_json)
        result = calculate_inheritance_share(total_value, heirs)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return json.dumps(
            {"error": "heirs_json không đúng định dạng JSON"}, ensure_ascii=False
        )


@track_tool_call
def business_name_validator(business_name: str) -> str:
    """
    Kiểm tra tên doanh nghiệp có hợp lệ theo Luật Doanh nghiệp Việt Nam.

    Args:
        business_name: Tên doanh nghiệp cần kiểm tra, ví dụ: "Công ty TNHH ABC"

    Returns:
        Kết quả kiểm tra tính hợp lệ và các lưu ý
    """
    # Tool Guardrail: Validate input length
    if not business_name or len(business_name.strip()) == 0:
        return json.dumps({"error": "Tên doanh nghiệp không được để trống"}, ensure_ascii=False)
    if len(business_name) > 200:
        return json.dumps({"error": "Tên doanh nghiệp quá dài (tối đa 200 ký tự)"}, ensure_ascii=False)

    result = check_business_name_rules(business_name)
    return json.dumps(result, ensure_ascii=False, indent=2)


@track_tool_call
def statute_lookup(case_type: str) -> str:
    """
    Tra cứu thời hiệu khởi kiện theo pháp luật Việt Nam.

    Args:
        case_type: Loại vụ việc, có thể là: "civil" (dân sự), "labor" (lao động), "administrative" (hành chính), "criminal" (hình sự)

    Returns:
        Thông tin về thời hiệu và căn cứ pháp lý
    """
    # Tool Guardrail: Validate type
    valid_cases = ["civil", "labor", "administrative", "criminal"]
    if case_type not in valid_cases:
        return json.dumps({"error": f"Loại vụ việc không hợp lệ. Các loại hợp lệ: {', '.join(valid_cases)}"}, ensure_ascii=False)

    result = get_statute_of_limitations(case_type)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ===== WEB SEARCH TOOLS =====


@track_tool_call
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Tìm kiếm thông tin pháp luật trên internet sử dụng Google Search.
    Dùng khi cần tìm tin tức, văn bản pháp luật mới, hoặc thông tin cập nhật.

    Args:
        query: Từ khóa tìm kiếm, ví dụ: "Luật Đất đai 2024 sửa đổi"
        max_results: Số kết quả tối đa (mặc định 5)

    Returns:
        Kết quả tìm kiếm với tiêu đề, link và nội dung tóm tắt
    """
    try:
        return search_engine(query, top_k=max_results)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Lỗi tìm kiếm: {str(e)}"


@track_tool_call
def tavily_search_tool(query: str, max_results: int = 5) -> str:
    """
    Tìm kiếm thông tin pháp luật sử dụng Tavily AI (tìm kiếm thông minh với AI).
    Tavily cung cấp kết quả tốt hơn và tóm tắt tự động cho câu hỏi pháp lý.

    Args:
        query: Câu hỏi hoặc từ khóa tìm kiếm, ví dụ: "Quy định mới về BHXH 2024"
        max_results: Số kết quả tối đa (mặc định 5)

    Returns:
        Kết quả tìm kiếm với tóm tắt AI và các nguồn liên quan
    """
    try:
        return tavily_search_legal(query, max_results=max_results)
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Tavily không khả dụng: {str(e)}"


@track_tool_call
def quick_answer_tool(question: str) -> str:
    """
    Trả lời nhanh câu hỏi bằng tìm kiếm web (Tavily Q&A).
    Dùng cho câu hỏi sự kiện, thống kê, hoặc thông tin cập nhật cần web search.

    Args:
        question: Câu hỏi cần trả lời, ví dụ: "Mức lương tối thiểu vùng 1 năm 2024"

    Returns:
        Câu trả lời trực tiếp từ web search
    """
    try:
        return tavily_qna(question)
    except Exception as e:
        logger.error(f"Quick answer error: {e}")
        return f"Không thể trả lời: {str(e)}"


@track_tool_call
def get_current_time() -> str:
    """
    Lấy thời gian và năm hiện tại của hệ thống.
    Dùng khi cần biết năm nay là năm nào để tính tuổi pháp lý hoặc kiểm tra thời gian.

    Returns:
        Chuỗi chứa thông tin thời gian hiện tại dưới dạng JSON
    """
    from datetime import datetime
    now = datetime.now()
    result = {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "current_year": now.year,
        "current_date": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ===== RETRIEVAL-BACKED LEGAL TOOLS =====


@track_tool_call
def article_lookup_tool(law_name: str, article_number: int = 0, limit: int = 5) -> str:
    """
    Tra cứu nội dung một điều của văn bản luật trong kho dữ liệu nội bộ.

    Dùng khi user hỏi "Điều X Luật Y nói gì?". Trả về đoạn văn bản khớp nhất.
    Lưu ý: kết quả semantic, chưa đảm bảo đúng điều nếu corpus thiếu metadata —
    agent NÊN gọi verify_citation_tool trước khi trích dẫn chính xác cho user.

    Args:
        law_name: Tên luật, ví dụ "Bộ luật Dân sự 2015".
        article_number: Số điều, ví dụ 418. Truyền 0 nếu tra cả luật.
        limit: Số kết quả tối đa (mặc định 5).
    """
    try:
        art = article_number if article_number and article_number > 0 else None
        result = lookup_article(law_name, art, limit=limit)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"article_lookup_tool error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def precedent_lookup_tool(fact_pattern: str, limit: int = 5) -> str:
    """
    Tra cứu án lệ (case law) liên quan đến một tình huống.

    Dùng khi user hỏi "trường hợp của tôi bị xử sao" / "có án lệ nào…".

    Args:
        fact_pattern: Tình huống thực tế, ví dụ "vay tiền không trả có giấy vay".
        limit: Số kết quả.
    """
    try:
        result = precedent_lookup(fact_pattern, limit=limit)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"precedent_lookup_tool error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def cross_reference_tool(law_name: str, article_number: int, limit: int = 5) -> str:
    """
    Tìm các điều/văn bản dẫn chiếu ĐẾN một điều luật cụ thể.

    Args:
        law_name: Tên luật, ví dụ "Bộ luật Dân sự 2015".
        article_number: Số điều, ví dụ 418.
        limit: Số kết quả.
    """
    try:
        result = cross_reference(law_name, article_number, limit=limit)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"cross_reference_tool error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def verify_citation_tool(law_name: str, article_number: int, claimed_text: str) -> str:
    """
    Xác minh một trích dẫn pháp luật có khớp văn bản thật trong corpus không.

    Anti-hallucination: TRƯỚC khi khẳng định "Điều X Luật Y nói Z" với user,
    hãy gọi tool này với Z làm claimed_text. Nếu verdict != 'consistent',
    không khẳng định; đề nghị web_search.

    Args:
        law_name: Tên luật, ví dụ "Bộ luật Dân sự 2015".
        article_number: Số điều, ví dụ 418.
        claimed_text: Nội dung agent KHẲNG ĐỊNH điều đó nói.
    """
    try:
        result = verify_citation(law_name, article_number, claimed_text)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"verify_citation_tool error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ===== KNOWLEDGE / DATA-DRIVEN LEGAL TOOLS =====


@track_tool_call
def severance_pay_tool(monthly_salary: float, months_worked: int) -> str:
    """
    Tính trợ cấp thôi việc theo Điều 48 Bộ luật Lao động 2019.

    Args:
        monthly_salary: Lương bình quân 6 tháng cuối (VNĐ), ví dụ 15000000.
        months_worked: Tổng số tháng làm việc, ví dụ 36.
    """
    try:
        result = calculate_severance_pay(monthly_salary, months_worked)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def overtime_pay_tool(hourly_wage: float, hours: float, day_type: str = "weekday") -> str:
    """
    Tính tiền làm thêm giờ theo Điều 107 Bộ luật Lao động 2019.

    Args:
        hourly_wage: Lương giờ thường (VNĐ), ví dụ 50000.
        hours: Số giờ làm thêm, ví dụ 4.
        day_type: weekday | rest_day | holiday.
    """
    try:
        result = calculate_overtime_pay(hourly_wage, hours, day_type=day_type)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def pit_monthly_tool(taxable_income: float) -> str:
    """
    Tính thuế TNCN lương tháng theo biểu lũy tiến (Luật Thuế TNCN + TT 111/2013).

    Args:
        taxable_income: Thu nhập tính thuế tháng (sau giảm trừ 11tr bản thân + 4.4tr/người phụ thuộc), VNĐ.
    """
    try:
        result = calculate_pit_monthly(taxable_income)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def land_registration_fee_tool(property_value: float, is_first_home: bool = True) -> str:
    """
    Tính lệ phí trước bạ nhà đất (ND 10/2022; TT 80/2020). Mức 0,5%.

    Args:
        property_value: Giá trị nhà/đất theo giá nhà nước (VNĐ).
        is_first_home: Lần đầu cấp sổ không.
    """
    try:
        result = calculate_land_registration_fee(property_value, is_first_home=is_first_home)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def vehicle_registration_fee_tool(vehicle_value: float, vehicle_type: str = "car", is_first_time: bool = True) -> str:
    """
    Tính lệ phí trước bạ xe (ND 10/2022; TT 80/2020).

    Args:
        vehicle_value: Giá trị xe (VNĐ).
        vehicle_type: car | motorcycle | truck.
        is_first_time: Lần đầu đăng ký không.
    """
    try:
        result = calculate_vehicle_registration_fee(vehicle_value, vehicle_type=vehicle_type, is_first_time=is_first_time)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def court_fee_tool(claim_value: float, case_type: str = "civil_first") -> str:
    """
    Tính án phí dân sự theo NQ 326/2016/UBTVQH14.

    Args:
        claim_value: Giá trị yêu cầu (VNĐ). 0 nếu không có giá.
        case_type: civil_first | civil_appeal | no_value.
    """
    try:
        result = calculate_court_fee(claim_value, case_type=case_type)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def admin_fine_lookup_tool(violation_type: str) -> str:
    """
    Tra mức phạt vi phạm hành chính phổ biến (giao thông, kinh doanh, đất đai).

    Args:
        violation_type: Một trong: traffic_no_license, traffic_alcohol_car_low,
            traffic_alcohol_motorbike, business_unregistered, land_violation_encroachment.
    """
    try:
        result = lookup_administrative_fine(violation_type)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def child_support_tool(payer_income: float, num_children: int = 1) -> str:
    """
    Ước lượng mức cấp dưỡng nuôi con sau ly hôn (Đ82 Luật HNGĐ 2014).

    Args:
        payer_income: Thu nhập tháng của bên có nghĩa vụ (VNĐ).
        num_children: Số con được cấp dưỡng.
    """
    try:
        result = calculate_child_support(payer_income, num_children=num_children)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def procedure_wizard_tool(procedure_type: str) -> str:
    """
    Hướng dẫn thủ tục pháp lý: hồ sơ, phí, cơ quan thụ lý, bước, thời hạn.

    Args:
        procedure_type: marriage_registration | business_registration | lawsuit_filing_civil
            | land_ownership_certificate | administrative_complaint | divorce.
    """
    try:
        result = procedure_wizard(procedure_type)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def jurisdiction_resolver_tool(dispute_type: str, claim_value: float = 0, location: str = "") -> str:
    """
    Xác định tòa/cơ quan có thẩm quyền thụ lý tranh chấp.

    Args:
        dispute_type: civil | criminal | administrative | labor | family | economic.
        claim_value: Giá trị yêu cầu (VNĐ), 0 nếu không có giá.
        location: Địa phương (nơi bị đơn cư trú / xảy ra sự kiện).
    """
    try:
        cv = claim_value if claim_value and claim_value > 0 else None
        result = jurisdiction_resolver(dispute_type, claim_value=cv, location=location)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def generate_document_template_tool(doc_type: str, params_json: str = "{}") -> str:
    """
    Sinh văn bản pháp lý mẫu (đơn khởi kiện, đơn khiếu nại, hợp đồng mua bán, đơn ly hôn).

    Mẫu THAM KHẢO, cần luật sư rà soát trước khi dùng chính thức.

    Args:
        doc_type: don_khoi_kien_civil | don_khieu_nai_hanh_chinh | hop_dong_mua_ban | don_ly_hon.
        params_json: JSON các trường điền, ví dụ '{"nguyen_don":"Nguyễn Văn A","bi_don":"Trần Thị B","yeu_cau":"buộc trả nợ 100tr","noi_nhan":"TAND quận X"}'.
    """
    try:
        result = generate_document_template(doc_type, params_json=params_json)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def law_version_tool(law_key: str, effective_year: int = 0) -> str:
    """
    Tra phiên bản/lịch sử hiệu lực của một luật Việt Nam.

    Args:
        law_key: blds_2015 | luat_dat_dai_2024 | blld_2019 | luat_doanh_nghiep_2020 | luat_hngd_2014.
        effective_year: Năm tham chiếu (truyền 0 để bỏ qua) — xác nhận luật còn hiệu lực.
    """
    try:
        ey = effective_year if effective_year and effective_year > 0 else None
        result = get_law_version(law_key, effective_year=ey)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@track_tool_call
def legal_disclaimer_tool(question: str) -> str:
    """
    Kiểm tra câu hỏi có cần escalate cho luật sư + trả disclaimer pháp lý.

    NÊN gọi để gắn disclaimer vào cuối câu trả lời pháp luật. Phát hiện topic
    hình sự/bào chữa → escalate, KHÔNG tư vấn chi tiết mà chuyển luật sư.

    Args:
        question: Câu hỏi của user.
    """
    try:
        result = legal_disclaimer_check(question)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ===== CREATE TOOLS =====

# Legal calculation tools
contract_penalty_tool = FunctionTool.from_defaults(fn=contract_penalty_calculator)
legal_age_tool = FunctionTool.from_defaults(fn=legal_age_checker)
inheritance_tool = FunctionTool.from_defaults(fn=inheritance_calculator)
business_name_tool = FunctionTool.from_defaults(fn=business_name_validator)
statute_tool = FunctionTool.from_defaults(fn=statute_lookup)

# Retrieval-backed legal tools
article_lookup_func_tool = FunctionTool.from_defaults(fn=article_lookup_tool)
precedent_lookup_func_tool = FunctionTool.from_defaults(fn=precedent_lookup_tool)
cross_reference_func_tool = FunctionTool.from_defaults(fn=cross_reference_tool)
verify_citation_func_tool = FunctionTool.from_defaults(fn=verify_citation_tool)

# Knowledge / data-driven legal tools
severance_pay_func_tool = FunctionTool.from_defaults(fn=severance_pay_tool)
overtime_pay_func_tool = FunctionTool.from_defaults(fn=overtime_pay_tool)
pit_monthly_func_tool = FunctionTool.from_defaults(fn=pit_monthly_tool)
land_registration_fee_func_tool = FunctionTool.from_defaults(fn=land_registration_fee_tool)
vehicle_registration_fee_func_tool = FunctionTool.from_defaults(fn=vehicle_registration_fee_tool)
court_fee_func_tool = FunctionTool.from_defaults(fn=court_fee_tool)
admin_fine_lookup_func_tool = FunctionTool.from_defaults(fn=admin_fine_lookup_tool)
child_support_func_tool = FunctionTool.from_defaults(fn=child_support_tool)
procedure_wizard_func_tool = FunctionTool.from_defaults(fn=procedure_wizard_tool)
jurisdiction_resolver_func_tool = FunctionTool.from_defaults(fn=jurisdiction_resolver_tool)
generate_document_template_func_tool = FunctionTool.from_defaults(fn=generate_document_template_tool)
law_version_func_tool = FunctionTool.from_defaults(fn=law_version_tool)
legal_disclaimer_func_tool = FunctionTool.from_defaults(fn=legal_disclaimer_tool)

# Web search tools
google_search_tool = FunctionTool.from_defaults(fn=web_search_tool)
tavily_tool = FunctionTool.from_defaults(fn=tavily_search_tool)
quick_answer_tool_func = FunctionTool.from_defaults(fn=quick_answer_tool)

# Utility tools
current_time_tool = FunctionTool.from_defaults(fn=get_current_time)

# All available tools for agent
all_tools = [
    # Legal calc tools
    contract_penalty_tool,
    legal_age_tool,
    inheritance_tool,
    business_name_tool,
    statute_tool,
    # Retrieval-backed legal tools
    article_lookup_func_tool,
    precedent_lookup_func_tool,
    cross_reference_func_tool,
    verify_citation_func_tool,
    # Knowledge / data-driven legal tools
    severance_pay_func_tool,
    overtime_pay_func_tool,
    pit_monthly_func_tool,
    land_registration_fee_func_tool,
    vehicle_registration_fee_func_tool,
    court_fee_func_tool,
    admin_fine_lookup_func_tool,
    child_support_func_tool,
    procedure_wizard_func_tool,
    jurisdiction_resolver_func_tool,
    generate_document_template_func_tool,
    law_version_func_tool,
    legal_disclaimer_func_tool,
    # Search tools
    google_search_tool,
    tavily_tool,
    quick_answer_tool_func,
    # Utility tools
    current_time_tool,
]