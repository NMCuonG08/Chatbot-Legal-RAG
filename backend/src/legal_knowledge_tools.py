"""
Data-driven legal knowledge tools for the Vietnamese law chatbot agent.

These tools encode Vietnamese legal knowledge as in-memory data tables
(rates, fee schedules, procedure steps, document templates). Unlike
``legal_tools.py`` (pure formulas) and ``legal_retrieval_tools.py`` (Qdrant
queries), this module holds *curated* reference data that the agent can
return directly without network access.

All rates/figures reference the cited statute as of the noted law year.
Vietnamese law changes frequently — verify ``effective_year`` against the
real corpus or web_search before relying on a figure for a live user.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CURRENT_YEAR = datetime.now().year


# ---------------------------------------------------------------------------
# 1. Labor tools
# ---------------------------------------------------------------------------

def calculate_severance_pay(monthly_salary: float, months_worked: int) -> Dict:
    """
    Tính trợ cấp thôi việc theo Điều 48 Bộ luật Lao động 2019.

    Quy tắc (Đ48 BLLĐ 2019): mỗi năm làm việc = trợ cấp bằng 0,5 tháng tiền lương
    cho thời gian làm việc trước ngày 1/1/2009, và 1 tháng cho mỗi năm từ 1/1/2009.
    Bản简化 này áp dụng 1 tháng/năm cho toàn bộ thời gian (trường hợp phổ biến,
    làm việc sau 2009). User làm việc trước 2009 cần tư vấn riêng.

    Args:
        monthly_salary: Mức lương bình quân 6 tháng cuối (VNĐ).
        months_worked: Tổng số tháng làm việc.

    Returns:
        Dict trợ cấp + căn cứ pháp lý.
    """
    try:
        if monthly_salary <= 0 or monthly_salary > 10**11:
            return {"error": "Mức lương không hợp lệ (0 - 100 tỷ VNĐ)."}
        if months_worked < 0 or months_worked > 600:
            return {"error": "Số tháng làm việc không hợp lệ (0 - 600 tháng)."}

        years = months_worked / 12.0
        allowance = monthly_salary * years  # 1 month/year post-2009 simplification
        rounded_allowance = round(allowance, -3)  # round to nearest 1k

        return {
            "monthly_salary": f"{monthly_salary:,.0f} VNĐ",
            "months_worked": months_worked,
            "years_equivalent": round(years, 2),
            "severance_allowance": f"{rounded_allowance:,.0f} VNĐ",
            "legal_basis": "Điều 48 Bộ luật Lao động 2019",
            "note": (
                "Tính 1 tháng lương/năm (trường hợp làm việc sau 1/1/2009). "
                "Nếu thời gian có trước 2009, phần trước 2009 = 0,5 tháng/năm — cần tư vấn riêng. "
                "Lương tính theo bình quân 6 tháng liền kề trước khi chấm dứt HĐLĐ."
            ),
        }
    except Exception as e:
        logger.error(f"severance calc error: {e}")
        return {"error": str(e)}


def calculate_overtime_pay(hourly_wage: float, hours: float, day_type: str = "weekday") -> Dict:
    """
    Tính tiền làm thêm giờ theo Điều 107 Bộ luật Lao động 2019.

    Hệ số làm thêm:
      - weekday: 150% (ngày thường)
      - rest_day: 200% (ngày nghỉ weekly)
      - holiday: 300% (ngày lễ/trả lương theo Đ111)

    Args:
        hourly_wage: Lương giờ giờ thường (VNĐ).
        hours: Số giờ làm thêm.
        day_type: weekday | rest_day | holiday.

    Returns:
        Dict tiền OT + hệ số + căn cứ.
    """
    try:
        if hourly_wage <= 0 or hourly_wage > 10**7:
            return {"error": "Lương giờ không hợp lệ."}
        if hours < 0 or hours > 24:
            return {"error": "Số giờ làm thêm không hợp lệ (0-24)."}

        rates = {"weekday": 1.5, "rest_day": 2.0, "holiday": 3.0}
        if day_type not in rates:
            return {"error": f"day_type không hợp lệ. Hợp lệ: {list(rates.keys())}"}

        rate = rates[day_type]
        ot_pay = hourly_wage * rate * hours
        return {
            "hourly_wage": f"{hourly_wage:,.0f} VNĐ/giờ",
            "hours": hours,
            "day_type": day_type,
            "multiplier": f"{rate}x",
            "overtime_pay": f"{ot_pay:,.0f} VNĐ",
            "legal_basis": "Điều 107 Bộ luật Lao động 2019",
            "note": "Ngày nghỉ/lễ đã trả lương thì hệ số nhân thêm 1 đơn vị (rest_day→300%, holiday→400%).",
        }
    except Exception as e:
        logger.error(f"overtime calc error: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 2. Tax & fee tools
# ---------------------------------------------------------------------------

# PIT monthly brackets (Luật Thuế TNCN 04/2007 + TT 111/2013, còn hiệu lực tới 2024).
PIT_MONTHLY_BRACKETS = [
    (0, 5_000_000, 0.05),
    (5_000_000, 10_000_000, 0.10),
    (10_000_000, 18_000_000, 0.15),
    (18_000_000, 32_000_000, 0.20),
    (32_000_000, 52_000_000, 0.25),
    (52_000_000, 80_000_000, 0.30),
    (80_000_000, float("inf"), 0.35),
]


def calculate_pit_monthly(taxable_income: float) -> Dict:
    """
    Tính thuế TNCN lương tháng theo biểu lũy tiến (Luật Thuế TNCN + TT 111/2013).

    Args:
        taxable_income: Thu nhập tính thuế tháng = thu nhập assessed - các khoản giảm trừ
                        (bản thân 11tr/tháng + người phụ thuộc 4.4tr/đối).

    Returns:
        Dict thuế phải nộp theo bậc + căn cứ.
    """
    try:
        if taxable_income <= 0:
            return {"taxable_income": taxable_income, "tax": 0, "note": "Thu nhập tính thuế <=0: không chịu thuế."}
        if taxable_income > 10**10:
            return {"error": "Thu nhập tính thuế quá lớn (>10 tỷ) — kiểm tra đơn vị VNĐ/tháng."}

        total_tax = 0.0
        breakdown = []
        remaining = taxable_income
        for low, high, rate in PIT_MONTHLY_BRACKETS:
            if remaining <= 0:
                break
            bracket_amount = min(remaining, high - low)
            tax_in_bracket = bracket_amount * rate
            if tax_in_bracket > 0:
                high_label = f"{high:,.0f}" if high != float("inf") else "∞"
                breakdown.append({
                    "bracket": f"{low:,.0f}–{high_label}",
                    "rate": f"{int(rate*100)}%",
                    "taxable_in_bracket": f"{bracket_amount:,.0f} VNĐ",
                    "tax": f"{tax_in_bracket:,.0f} VNĐ",
                })
            total_tax += tax_in_bracket
            remaining -= bracket_amount

        return {
            "taxable_income": f"{taxable_income:,.0f} VNĐ",
            "tax_owed": f"{total_tax:,.0f} VNĐ",
            "effective_rate": f"{(total_tax / taxable_income * 100):.2f}%" if taxable_income else "0%",
            "breakdown": breakdown,
            "legal_basis": "Luật Thuế TNCN 27/2007 + TT 111/2013/TT-BTC",
            "note": "Đã giả định taxable_income = thu nhập assessed - giảm trừ. Giảm trừ bản thân 11tr/tháng.",
        }
    except Exception as e:
        logger.error(f"PIT calc error: {e}")
        return {"error": str(e)}


def calculate_land_registration_fee(property_value: float, is_first_home: bool = True) -> Dict:
    """
    Tính lệ phí trước bạ nhà đất (Nghị định 10/2022 và TT 80/2020).

    Mức thu: 0,5% giá trị tài sản (nhà ở, đất ở) lần đầu và lần thứ 2 trở đi.

    Args:
        property_value: Giá trị nhà/đất theo giá nhà nước thẩm định (VNĐ).
        is_first_home: Có phải lần đầu cấp sổ không (ảnh hưởng một số diện đặc thù).

    Returns:
        Dict lệ phí + căn cứ.
    """
    try:
        if property_value <= 0 or property_value > 10**13:
            return {"error": "Giá trị nhà đất không hợp lệ (0 - 10.000 tỷ VNĐ)."}
        rate = 0.005
        fee = property_value * rate
        return {
            "property_value": f"{property_value:,.0f} VNĐ",
            "is_first_home": is_first_home,
            "rate": f"{rate*100:.1f}%",
            "registration_fee": f"{fee:,.0f} VNĐ",
            "legal_basis": "Nghị định 10/2022/NĐ-CP; TT 80/2020/TT-BTC",
            "note": "Tính trên giá nhà nước, KHÔNG phải giá giao dịch thị trường. Một số diện nhà ở xã hội có miễn.",
        }
    except Exception as e:
        logger.error(f"land reg fee error: {e}")
        return {"error": str(e)}


def calculate_vehicle_registration_fee(vehicle_value: float, vehicle_type: str = "car", is_first_time: bool = True) -> Dict:
    """
    Tính lệ phí trước bạ xe (Nghị định 10/2022; TT 80/2020).

    Ô tô tải/xe khách lần đầu 2%, lần sau 4%. Ô tô chở người dưới 9 chỗ lần đầu
    10% (tại Hà Nội/HCM), 11-12% tại một số tỉnh. Xe máy 1-5%.

    Args:
        vehicle_value: Giá trị xe (VNĐ).
        vehicle_type: car | motorcycle | truck.
        is_first_time: Có phải lần đầu đăng ký không.

    Returns:
        Dict lệ phí + căn cứ.
    """
    try:
        if vehicle_value <= 0 or vehicle_value > 10**11:
            return {"error": "Giá trị xe không hợp lệ."}

        # Simplified schedule; real rates vary by province and seating.
        schedule = {
            "car": (0.10, 0.12),      # HN/HCM 10% first, 12% later
            "motorcycle": (0.05, 0.05),
            "truck": (0.02, 0.04),
        }
        if vehicle_type not in schedule:
            return {"error": f"vehicle_type không hợp lệ. Hợp lệ: {list(schedule.keys())}"}

        first_rate, later_rate = schedule[vehicle_type]
        rate = first_rate if is_first_time else later_rate
        fee = vehicle_value * rate
        return {
            "vehicle_value": f"{vehicle_value:,.0f} VNĐ",
            "vehicle_type": vehicle_type,
            "is_first_time": is_first_time,
            "rate": f"{int(rate*100)}%",
            "registration_fee": f"{fee:,.0f} VNĐ",
            "legal_basis": "Nghị định 10/2022/NĐ-CP; TT 80/2020/TT-BTC",
            "note": "Tỷ lệ thực tùy tỉnh/thành. Ô tô dưới 9 chỗ HN/HCM 10% lần đầu. Cần tra theo địa phương chính xác.",
        }
    except Exception as e:
        logger.error(f"vehicle reg fee error: {e}")
        return {"error": str(e)}


def calculate_court_fee(claim_value: float, case_type: str = "civil_first") -> Dict:
    """
    Tính án phí dân sự sơ thẩm theo Nghị quyết 326/2016/UBTVQH14.

    Đối với yêu cầu có giá: 5% giá trị yêu cầu, tối thiểu 300.000đ.
    Yêu cầu không có giá: 300.000đ (sơ thẩm).

    Args:
        claim_value: Giá trị yêu cầu (VNĐ). 0 nếu không có giá.
        case_type: civil_first | civil_appeal | no_value.

    Returns:
        Dict án phí + căn cứ.
    """
    try:
        if claim_value < 0 or claim_value > 10**13:
            return {"error": "Giá trị yêu cầu không hợp lệ."}
        if case_type not in ("civil_first", "civil_appeal", "no_value"):
            return {"error": "case_type hợp lệ: civil_first | civil_appeal | no_value"}

        MIN_FEE = 300_000
        if case_type == "no_value" or claim_value == 0:
            fee = MIN_FEE
            return {
                "case_type": case_type,
                "claim_value": "Không có giá / 0",
                "court_fee": f"{fee:,.0f} VNĐ",
                "legal_basis": "NQ 326/2016/UBTVQH14",
                "note": "Yêu cầu không có giá: án phí sơ thẩm 300.000đ.",
            }

        rate = 0.05
        fee = max(claim_value * rate, MIN_FEE)
        return {
            "case_type": case_type,
            "claim_value": f"{claim_value:,.0f} VNĐ",
            "rate": "5%",
            "court_fee": f"{fee:,.0f} VNĐ",
            "min_fee_applied": fee == MIN_FEE,
            "legal_basis": "Nghị quyết 326/2016/UBTVQH14 (có hiệu lực từ 1/1/2017)",
            "note": "Sơ thẩm: 5%, tối thiểu 300k. Phúc thẩm: án phí = 50% sơ thẩm nếu đúng/yêu cầu.",
        }
    except Exception as e:
        logger.error(f"court fee calc error: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 3. Administrative fines & family
# ---------------------------------------------------------------------------

ADMIN_FINES_TABLE = {
    "traffic_no_license": {
        "fine": "1.000.000 – 2.000.000 VNĐ (ô tô); 400.000 – 600.000 VNĐ (xe máy)",
        "additional": "Tạm giữ xe 7 ngày; có thể tịch thu xe.",
        "legal_basis": "Điều 21, 22 Nghị định 100/2019/NĐ-CP (sửa đổi ND 123/2021)",
    },
    "traffic_alcohol_car_low": {
        "fine": "6.000.000 – 8.000.000 VNĐ (ô tô, nồng độ >0 nhưng ≤80mg/100ml)",
        "additional": "Tước GPLX 10-12 tháng.",
        "legal_basis": "Điều 5, 8 Nghị định 123/2021/NĐ-CP",
    },
    "traffic_alcohol_motorbike": {
        "fine": "2.000.000 – 3.000.000 VNĐ (xe máy, nồng độ >40mg/100ml)",
        "additional": "Tước GPLX 22-24 tháng.",
        "legal_basis": "Điều 6, 8 Nghị định 123/2021/NĐ-CP",
    },
    "business_unregistered": {
        "fine": "1.000.000 – 2.000.000 VNĐ (cá nhân); 2.000.000 – 5.000.000 VNĐ (tổ chức)",
        "additional": "Buộc đăng ký kinh doanh + nộp đủ thuế.",
        "legal_basis": "Điều 12 Nghị định 166/2013/NĐ-CP; Luật Doanh nghiệp 2020",
    },
    "land_violation_encroachment": {
        "fine": "Tùy diện tích và phân loại đất, 1.000.000 – 10.000.000 VNĐ/01 ha phần lấn chiếm",
        "additional": "Buộc trả lại đất, khôi phục tình trạng.",
        "legal_basis": "Điều 12 Nghị định 91/2019/NĐ-CP; Luật Đất đai 2024",
    },
}


def lookup_administrative_fine(violation_type: str) -> Dict:
    """
    Tra mức phạt vi phạm hành chính phổ biến (giao thông, kinh doanh, đất đai).

    Bảng phạt tham khảo, mức thực tùy tình tiết tăng/nặng.

    Args:
        violation_type: Loại vi phạm (xem keys).

    Returns:
        Dict mức phạt + bổ sung + căn cứ, hoặc danh sách keys hợp lệ.
    """
    key = (violation_type or "").strip().lower()
    entry = ADMIN_FINES_TABLE.get(key)
    if not entry:
        return {
            "error": f"Loại vi phạm '{violation_type}' không có trong bảng.",
            "available_types": list(ADMIN_FINES_TABLE.keys()),
            "note": "Bảng chỉ chứa vài vi phạm phổ biến. Tra luật gốc + web_search cho trường hợp khác.",
        }
    return {
        "violation_type": key,
        "fine": entry["fine"],
        "additional": entry["additional"],
        "legal_basis": entry["legal_basis"],
        "note": "Mức tham khảo; tùy tình tiết tăng/nặng. Luôn xác nhận với văn bản gốc.",
    }


def calculate_child_support(payer_income: float, num_children: int = 1) -> Dict:
    """
    Ước lượng mức cấp dưỡng nuôi con sau ly hôn (Luật HNGĐ 2014).

    VN KHÔNG có công thức cố định: tòa quyết định theo tình trạng tài chính
    hai bên + nhu cầu con. Hướng dẫn: thường ≥10% thu nhập/người phụ thuộc,
    tùy độ tuổi và hoàn cảnh (Đ82 Luật HNGĐ 2014).

    Args:
        payer_income: Thu nhập hàng tháng của bên có nghĩa vụ (VNĐ).
        num_children: Số con được cấp dưỡng.

    Returns:
        Dict ước lượng + căn cứ + cảnh báo.
    """
    try:
        if payer_income < 0 or payer_income > 10**9:
            return {"error": "Thu nhập không hợp lệ."}
        if num_children < 1 or num_children > 10:
            return {"error": "Số con không hợp lệ (1-10)."}

        # Guideline: 25% for 1 child, decay per additional child, capped 30%.
        per_child_rate = max(0.10, 0.25 - 0.05 * (num_children - 1))
        per_child_rate = min(per_child_rate, 0.30)
        per_child = payer_income * per_child_rate
        total = per_child * num_children

        return {
            "payer_income": f"{payer_income:,.0f} VNĐ/tháng",
            "num_children": num_children,
            "per_child_rate_guideline": f"{per_child_rate*100:.0f}%",
            "per_child_estimated": f"{per_child:,.0f} VNĐ/tháng",
            "total_estimated": f"{total:,.0f} VNĐ/tháng",
            "legal_basis": "Điều 82 Luật Hôn nhân và Gia đình 2014",
            "warning": (
                "Mức này là ƯỚC LƯỢNG hướng dẫn, KHÔNG phải quy định pháp luật. "
                "Tòa quyết định cuối cùng dựa trên nhu cầu con + khả năng hai bên + điều kiện sống. "
                "Bên có nghĩa vụ nuôi con dưới 18 tuổi phải cấp dưỡng bắt buộc."
            ),
        }
    except Exception as e:
        logger.error(f"child support error: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 6. Law version & compliance / disclaimer
# ---------------------------------------------------------------------------

# The version table lives in a dedicated data module so it can grow without
# inflating this tools file and so ``legal_effectivity`` can resolve the
# replaces/amended chain. Re-exported here for backward compatibility with any
# caller that imported ``LAW_VERSIONS`` from this module.
from legal_corpus_versions import (  # noqa: E402 - re-export
    LAW_VERSIONS,
    available_law_keys,
    find_version_by_key,
    find_version_by_name,
)
from legal_effectivity import classify_effectivity  # noqa: E402


def get_law_version(law_key: str, effective_year: Optional[int] = None) -> Dict:
    """
    Tra phiên bản/lịch sử hiệu lực của một luật Việt Nam.

    Args:
        law_key: Khóa luật, ví dụ blds_2015, luat_dat_dai_2024, blld_2019.
        effective_year: Năm tham chiếu (tùy chọn) để xác nhận luật còn hiệu lực
            hay đã được thay. Khi cung cấp, ``status_at_year`` được tính theo
            năm đó thay vì hôm nay.

    Returns:
        Dict thông tin phiên bản + trạng thái hiệu lực (in_force |
        not_yet_effective | repealed | amended).
    """
    key = (law_key or "").strip().lower()
    entry = find_version_by_key(key)
    if not entry:
        return {"error": "Không có trong bảng phiên bản.", "available": available_law_keys()}

    full_name = entry.get("full_name")
    if effective_year:
        # Project the reference date to the end of the requested year so a
        # statute effective mid-year is treated as in force within that year.
        from datetime import date as _date
        ref = _date(effective_year, 12, 31)
        status = classify_effectivity(full_name, effective_year, as_of=ref)
    else:
        status = classify_effectivity(full_name)

    return {
        "law_key": key,
        **entry,
        "status_at_year": status,
        "note": "Tra cứu phiên bản để tránh tư vấn theo luật đã hết hiệu lực. Xác nhận trên CSDL văn bản pháp luật.",
    }


# Topics requiring a licensed lawyer — chatbot must escalate, not advise.
ESCALATION_TOPICS = [
    "bào chữa", "bào chữa viên", "khởi tố hình sự", "truy tố",
    "tội danh", "trộm cắp", "giết người", "cố ý gây thương tích",
    "mua bán ma túy", "chống người thi hành công vụ",
    "khiếu nại án", "phúc thẩm hình sự",
    "thi hành án hình sự", "tù giam", "tử hình",
]


def legal_disclaimer_check(question: str) -> Dict:
    """
    Kiểm tra câu hỏi có thuộc diện cần luật sư hành nghề không, trả disclaimer + cờ escalate.

    Dùng cho mọi câu trả lời pháp luật: agent nên gọi để lấy disclaimer gắn cuối câu trả lời,
    và escalate khi phát hiện topic hình sự/phức tạp.

    Args:
        question: Câu hỏi của user.

    Returns:
        Dict: escalate (bool), disclaimer (str), matched_topics (list).
    """
    q = (question or "").lower()
    matched = [t for t in ESCALATION_TOPICS if t in q]
    escalate = bool(matched)

    if escalate:
        disclaimer = (
            "⚠️ Đây là vấn đề thuộc diện cần LUẬT SƯ hành nghề (hình sự/bào chữa). "
            "Trợ lý AI chỉ cung cấp thông tin chung, KHÔNG thay thế luật sư. "
            "Vui lòng liên hệ luật sư hoặc Trung tâm trợ giúp pháp lý nhà nước."
        )
    else:
        disclaimer = (
            "ℹ️ Thông tin trên mang tính tham khảo pháp lý, không phải ý kiến pháp lý chính thức. "
            "Mỗi vụ việc có tình tiết riêng — nên tham vấn luật sư hoặc cơ quan có thẩm quyền."
        )

    return {
        "escalate": escalate,
        "matched_topics": matched,
        "disclaimer": disclaimer,
        "note": "Luôn gắn disclaimer vào cuối câu trả lời pháp luật để tuân thủ giới hạn trợ lý AI.",
    }