import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def coerce_float(val: Any, default: float = 0.0) -> float:
    """Safely coerce any int, float, or stringified number into a float."""
    if isinstance(val, (int, float)):
        return float(val)
    if not val:
        return default
    s = str(val).lower().replace("vnđ", "").replace("vnd", "").replace("đ", "").strip()
    if "." in s and "," in s:
        if s.find(".") < s.find(","):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s and s.count(".") > 1:
        s = s.replace(".", "")
    elif "," in s:
        s = s.replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    return default


def coerce_int(val: Any, default: int = 0) -> int:
    """Safely coerce any int, float, or stringified number into an int."""
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    f = coerce_float(val, default=float(default))
    return int(f)


class ContractPenaltyInput(BaseModel):
    contract_value: float = Field(gt=0, description="Giá trị hợp đồng (VNĐ)")
    penalty_rate: float = Field(ge=0, description="Tỷ lệ phạt (%/ngày)")
    days_late: int = Field(ge=0, description="Số ngày chậm trễ")


class LegalEntityAgeInput(BaseModel):
    birth_year: int = Field(gt=1900, description="Năm sinh (4 chữ số)")
    action_type: str = Field(default="sign_contract", description="Loại hành vi pháp lý")
    gender: Optional[str] = Field(default="", description="Giới tính (male/female)")


class InheritanceShareInput(BaseModel):
    total_value: float = Field(gt=0, description="Tổng giá trị di sản (VNĐ)")
    heirs: List[Dict[str, Any]] = Field(description="Danh sách người thừa kế")


def calculate_contract_penalty(
    contract_value: float, penalty_rate: float, days_late: int
) -> Dict:
    """
    Tính tiền phạt vi phạm hợp đồng theo quy định pháp luật Việt Nam.

    Args:
        contract_value: Giá trị hợp đồng (VNĐ)
        penalty_rate: Tỷ lệ phạt (%, thường 0.05-0.3%/ngày theo Bộ luật Dân sự)
        days_late: Số ngày chậm trễ

    Returns:
        Dict with penalty amount and details
    """
    try:
        c_val = coerce_float(contract_value)
        p_rate = coerce_float(penalty_rate)
        d_late = coerce_int(days_late)

        # Validate with Pydantic
        inp = ContractPenaltyInput(
            contract_value=c_val if c_val > 0 else 1.0,
            penalty_rate=max(0.0, p_rate),
            days_late=max(0, d_late),
        )
        contract_value, penalty_rate, days_late = inp.contract_value, inp.penalty_rate, inp.days_late
        # Theo Điều 418 Bộ luật Dân sự 2015: phạt vi phạm không quá 8% giá trị
        # phần nghĩa vụ hợp đồng bị vi phạm. Trần pháp định, KHÔNG phải "thông
        # lệ" 12% như bản cũ đã ghi sai.
        STATUTORY_PENALTY_CAP_RATE = 0.08

        penalty_amount = contract_value * (penalty_rate / 100) * days_late
        max_penalty = contract_value * STATUTORY_PENALTY_CAP_RATE

        if penalty_amount > max_penalty:
            penalty_amount = max_penalty
            note = "Đã áp dụng mức phạt tối đa 8% giá trị hợp đồng (Điều 418 BLDS 2015)"
        else:
            note = "Tính theo tỷ lệ phạt đã thỏa thuận"

        result = {
            "contract_value": f"{contract_value:,.0f} VNĐ",
            "penalty_rate": f"{penalty_rate}%/ngày",
            "days_late": days_late,
            "penalty_amount": f"{penalty_amount:,.0f} VNĐ",
            "note": note,
            "legal_basis": "Điều 418 Bộ luật Dân sự 2015 - Phạt vi phạm không quá 8% giá trị nghĩa vụ bị vi phạm",
        }

        logger.info(f"[TOOL] Contract penalty calculated: {result}")
        return result

    except Exception as e:
        logger.error(f"Error calculating contract penalty: {e}")
        return {"error": str(e)}


def check_legal_entity_age(
    birth_year: int, action_type: str = "sign_contract", gender: str = ""
) -> Dict:
    """
    Kiểm tra tuổi pháp lý để thực hiện hành vi dân sự theo Bộ luật Dân sự Việt Nam.

    Args:
        birth_year: Năm sinh
        action_type: Loại hành vi (sign_contract, marriage, work, criminal_responsibility)
        gender: Giới tính ("male"/"female"). BẮT BUỘC khi action_type="marriage"
                vì tuổi kết hôn khác nhau theo giới (nam 20, nữ 18).

    Returns:
        Dict with eligibility status and legal details
    """
    try:
        birth_year = coerce_int(birth_year, default=2000)
        current_year = datetime.now().year
        age = current_year - birth_year

        # Các mốc tuổi pháp lý theo pháp luật Việt Nam
        age_requirements = {
            "sign_contract": {
                "min_age": 18,
                "description": "Đủ 18 tuổi để ký hợp đồng (Điều 21 Bộ luật Dân sự 2015)",
                "partial_age": 15,
                "partial_note": "Từ 15-18 tuổi cần có sự đồng ý của người đại diện hợp pháp",
            },
            "work": {
                "min_age": 15,
                "description": "Đủ 15 tuổi được làm việc (Điều 143 Bộ luật Lao động 2019)",
                "note": "Dưới 15 tuổi chỉ làm công việc nghệ thuật với điều kiện đặc biệt",
            },
            "criminal_responsibility": {
                "min_age": 16,
                "description": "Đủ 16 tuổi chịu trách nhiệm hình sự (Điều 12 Bộ luật Hình sự 2015)",
                "partial_age": 14,
                "partial_note": "Từ 14-16 tuổi chỉ chịu trách nhiệm với tội đặc biệt nghiêm trọng",
            },
        }

        # Kết hôn: tuổi khác nhau theo giới (Điều 8 Luật HNGĐ 2014).
        # Nam: đủ 20; Nữ: đủ 18. Bắt buộc có gender để tránh tư vấn sai.
        if action_type == "marriage":
            gender_norm = (gender or "").strip().lower()
            if gender_norm not in ("male", "female"):
                return {
                    "error": (
                        "Để kiểm tra tuổi kết hôn cần biết giới tính vì nam phải đủ 20 tuổi, "
                        "nữ phải đủ 18 tuổi (Điều 8 Luật Hôn nhân và Gia đình 2014). "
                        "Vui lòng truyền gender='male' hoặc gender='female'."
                    )
                }
            min_age = 20 if gender_norm == "male" else 18
            description = (
                f"{'Nam đủ 20 tuổi' if gender_norm == 'male' else 'Nữ đủ 18 tuổi'} "
                f"(Điều 8 Luật Hôn nhân và Gia đình 2014)"
            )
            req = {
                "min_age": min_age,
                "description": description,
                "note": "Nam: 20 tuổi, Nữ: 18 tuổi",
            }
        else:
            req = age_requirements.get(action_type, age_requirements["sign_contract"])

        min_age = req["min_age"]
        partial_age = req.get("partial_age", 0)

        if age >= min_age:
            eligible = True
            status = "Đủ điều kiện"
        elif partial_age > 0 and age >= partial_age:
            eligible = "partial"
            status = f"Có điều kiện: {req.get('partial_note', '')}"
        else:
            eligible = False
            status = "Chưa đủ điều kiện"

        result = {
            "age": age,
            "action_type": action_type,
            "gender": (gender or "").strip().lower() or None,
            "eligible": eligible,
            "status": status,
            "legal_basis": req["description"],
            "note": req.get("note", ""),
        }

        logger.info(f"[TOOL] Legal age check: {result}")
        return result

    except Exception as e:
        logger.error(f"Error checking legal age: {e}")
        return {"error": str(e)}


def calculate_inheritance_share(total_value: float, heirs: List[Dict]) -> Dict:
    """
    Tính phần thừa kế theo pháp luật Việt Nam (thừa kế theo pháp luật).

    Áp dụng hàng thừa kế theo Điều 651 Bộ luật Dân sự 2015:
      - Hàng thứ nhất: vợ/chồng, cha đẻ, mẹ đẻ, cha nuôi, mẹ nuôi, con đẻ, con nuôi.
      - Hàng thứ hai: ông nội, bà nội, ông ngoại, bà ngoại, anh ruột, chị ruột,
        em ruột (cùng cha cùng mẹ hoặc cùng cha khác mẹ hoặc cùng mẹ khác cha).
      - Hàng thứ ba: cô ruột, dì ruột, chú ruột, bác ruột, cậu ruột.
    Nguyên tắc: chỉ người thuộc hàng thừa kế cao nhất còn sống mới được thừa kế;
    hàng thấp hơn chỉ được thừa kế khi KHÔNG còn ai ở hàng cao hơn.

    Args:
        total_value: Tổng giá trị tài sản thừa kế (VNĐ)
        heirs: Danh sách người thừa kế [{"name": str, "relation": str, "is_minor": bool}]
               relation: spouse|child|parent (hàng 1), sibling|grandparent (hàng 2),
                         uncle|aunt (hàng 3)

    Returns:
        Dict with inheritance shares for each heir
    """
    try:
        if not heirs:
            return {"error": "Không có người thừa kế"}

        # Ánh xạ relation -> hàng thừa kế (Điều 651 BLDS 2015).
        TIER_MAP = {
            # Hàng thứ nhất
            "spouse": 1,
            "child": 1,
            "parent": 1,
            # Hàng thứ hai
            "sibling": 2,
            "grandparent": 2,
            # Hàng thứ ba
            "uncle": 3,
            "aunt": 3,
        }

        tier_labels = {
            1: "hàng thừa kế thứ nhất (vợ/chồng, cha, mẹ, con)",
            2: "hàng thừa kế thứ hai (ông bà, anh chị em ruột)",
            3: "hàng thừa kế thứ ba (cô, dì, chú, bác, cậu ruột)",
        }

        # Gom người thừa kế theo hàng. Bỏ qua relation không hợp lệ.
        tier_to_heirs: Dict[int, List[Dict]] = {}
        excluded: List[Dict] = []
        for heir in heirs:
            relation = (heir.get("relation") or "").strip().lower()
            tier = TIER_MAP.get(relation)
            if tier is None:
                excluded.append(
                    {"name": heir.get("name", ""), "relation": relation,
                     "reason": "Không thuộc hàng thừa kế hợp lệ theo Điều 651 BLDS 2015"}
                )
                continue
            tier_to_heirs.setdefault(tier, []).append(heir)

        if not tier_to_heirs:
            return {
                "error": "Không có người thừa kế hợp lệ trong danh sách",
                "excluded": excluded,
                "legal_basis": "Điều 651 Bộ luật Dân sự 2015",
            }

        # Chỉ hàng cao nhất có mặt được chia; hàng thấp hơn bị loại bỏ.
        active_tier = min(tier_to_heirs.keys())
        active_heirs = tier_to_heirs[active_tier]
        skipped_lower = [
            {"name": h.get("name", ""), "relation": h.get("relation", ""),
             "tier": t, "reason": f"Bị loại vì có người ở hàng {active_tier} còn sống"}
            for t, hs in tier_to_heirs.items() if t > active_tier
            for h in hs
        ]

        num_heirs = len(active_heirs)
        share_per_heir = total_value / num_heirs

        distribution = []
        for heir in active_heirs:
            heir_share = {
                "name": heir.get("name", ""),
                "relation": heir.get("relation", ""),
                "tier": active_tier,
                "share": f"{share_per_heir:,.0f} VNĐ",
                "percentage": f"{(100 / num_heirs):.2f}%",
            }
            if heir.get("is_minor", False):
                heir_share["note"] = (
                    "Người chưa thành niên, cần người đại diện quản lý tài sản"
                )
            distribution.append(heir_share)

        result = {
            "total_value": f"{total_value:,.0f} VNĐ",
            "num_heirs": num_heirs,
            "share_per_heir": f"{share_per_heir:,.0f} VNĐ",
            "distribution": distribution,
            "active_tier": active_tier,
            "active_tier_label": tier_labels[active_tier],
            "excluded": excluded,
            "skipped_lower_tiers": skipped_lower,
            "legal_basis": "Điều 651 Bộ luật Dân sự 2015 - Thừa kế theo pháp luật theo hàng thừa kế",
            "note": (
                f"Chỉ chia cho {tier_labels[active_tier]}. "
                "Hàng thấp hơn chỉ được thừa kế khi không còn ai ở hàng cao hơn. "
                "Lưu ý Điều 652: phần của người thừa kế đã chết được chuyển cho con cháu họ."
            ),
        }

        logger.info(f"[TOOL] Inheritance calculated: {result}")
        return result

    except Exception as e:
        logger.error(f"Error calculating inheritance: {e}")
        return {"error": str(e)}


def check_business_name_rules(business_name: str) -> Dict:
    """
    Kiểm tra tên doanh nghiệp có hợp lệ theo Luật Doanh nghiệp Việt Nam không.

    Args:
        business_name: Tên doanh nghiệp cần kiểm tra

    Returns:
        Dict with validation results
    """
    try:
        issues = []
        warnings = []

        # Theo Điều 36 Luật Doanh nghiệp 2020

        # 1. Không được trùng hoặc tương tự với tên đã đăng ký
        # (Cần tra cứu cơ sở dữ liệu - bỏ qua trong demo đơn giản)

        # 2. Không chứa các từ ngữ cấm
        prohibited_words = [
            "việt nam",
            "quốc gia",
            "chính phủ",
            "nhà nước",
            "đảng",
            "bộ",
            "ban",
            "ngành",
            "ủy ban",
        ]

        name_lower = business_name.lower()
        for word in prohibited_words:
            if word in name_lower:
                issues.append(
                    f"Tên không được chứa từ '{word}' (Điều 36 Luật Doanh nghiệp 2020)"
                )

        # 3. Kiểm tra độ dài (không quá ngắn)
        if len(business_name) < 5:
            warnings.append("Tên doanh nghiệp quá ngắn, nên có ít nhất 5 ký tự")

        # 4. Kiểm tra ký tự đặc biệt
        import re

        if re.search(r'[!@#$%^&*()_+=\[\]{};\'\\:"|,.<>?/~`]', business_name):
            warnings.append(
                "Tên chứa ký tự đặc biệt, có thể gây khó khăn trong đăng ký"
            )

        # 5. Kiểm tra số ở đầu
        if business_name and business_name[0].isdigit():
            warnings.append("Tên bắt đầu bằng số, nên bắt đầu bằng chữ cái")

        is_valid = len(issues) == 0

        result = {
            "business_name": business_name,
            "is_valid": is_valid,
            "status": "Hợp lệ" if is_valid else "Không hợp lệ",
            "issues": issues,
            "warnings": warnings,
            "legal_basis": "Điều 36 Luật Doanh nghiệp 2020",
            "recommendation": "Nên tra cứu trên hệ thống đăng ký kinh doanh quốc gia để đảm bảo không trùng lặp",
        }

        logger.info(f"[TOOL] Business name check: {result}")
        return result

    except Exception as e:
        logger.error(f"Error checking business name: {e}")
        return {"error": str(e)}


def get_statute_of_limitations(case_type: str) -> Dict:
    """
    Tra cứu thời hiệu khởi kiện theo pháp luật Việt Nam.

    Args:
        case_type: Loại vụ việc (civil, labor, administrative, criminal)

    Returns:
        Dict with statute of limitations info
    """
    try:
        # Theo các bộ luật Việt Nam
        statutes = {
            "civil": {
                "general": "3 năm",
                "description": "Thời hiệu khởi kiện chung là 3 năm (Điều 155 Bộ luật Dân sự 2015)",
                "exceptions": [
                    "Tranh chấp quyền sở hữu đất đai: 10 năm",
                    "Yêu cầu công nhận cha, mẹ, con: Không có thời hiệu",
                    "Bồi thường thiệt hại ngoài hợp đồng: 3 năm",
                ],
                "start_date": "Từ ngày người có quyền yêu cầu biết hoặc phải biết quyền và lợi ích hợp pháp bị xâm phạm",
            },
            "labor": {
                "general": "1 năm",
                "description": "Thời hiệu khởi kiện tranh chấp lao động là 1 năm (Điều 193 Bộ luật Lao động 2019)",
                "exceptions": [
                    "Tranh chấp về tiền lương, trợ cấp thôi việc: 2 năm",
                    "Sa thải trái pháp luật: 1 năm",
                ],
                "start_date": "Từ ngày quyền, lợi ích bị xâm phạm",
            },
            "administrative": {
                "general": "1 năm",
                "description": "Thời hiệu khởi kiện hành chính là 1 năm (Điều 31 Luật TTHC 2015)",
                "exceptions": [
                    "Quyết định xử phạt vi phạm hành chính: 1 năm",
                    "Quyết định thu hồi đất: 1 năm",
                ],
                "start_date": "Từ ngày nhận được quyết định hành chính hoặc biết quyết định",
            },
            "criminal": {
                "general": "Tùy mức hình phạt",
                "description": "Thời hiệu truy cứu trách nhiệm hình sự (Điều 27 Bộ luật Hình sự 2015)",
                "details": [
                    "Phạm tội ít nghiêm trọng: 5 năm",
                    "Phạm tội nghiêm trọng: 10 năm",
                    "Phạm tội rất nghiêm trọng: 15 năm",
                    "Phạm tội đặc biệt nghiêm trọng: 20 năm",
                    "Tội phạm chiến tranh, tội phạm chống loài người: Không có thời hiệu",
                ],
                "start_date": "Từ ngày thực hiện tội phạm",
            },
        }

        statute = statutes.get(case_type)
        if not statute:
            return {
                "error": f"Loại vụ việc '{case_type}' không hợp lệ",
                "valid_types": list(statutes.keys()),
            }

        result = {
            "case_type": case_type,
            "general_statute": statute["general"],
            "description": statute["description"],
            "start_date": statute["start_date"],
        }

        if "exceptions" in statute:
            result["exceptions"] = statute["exceptions"]
        if "details" in statute:
            result["details"] = statute["details"]

        logger.info(f"[TOOL] Statute of limitations: {result}")
        return result

    except Exception as e:
        logger.error(f"Error getting statute of limitations: {e}")
        return {"error": str(e)}