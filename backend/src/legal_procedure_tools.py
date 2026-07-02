"""
Procedure / jurisdiction / document-template tools for the Vietnamese law chatbot.

Split out of ``legal_knowledge_tools.py`` to keep each module under the
800-line guideline. These tools return curated procedural guidance and
generate fill-in-the-blank legal documents from parameters.
"""

import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


PROCEDURES = {
    "marriage_registration": {
        "title": "Đăng ký kết hôn",
        "competent_authority": "UBND cấp xã/phường/thị trấn nơi một bên cư trú.",
        "required_docs": [
            "Tờ khai đăng ký kết hôn (mẫu TP/HT-2012)",
            "Giấy chứng nhận nhân thân (CCCD/hộ chiếu) của cả hai bên",
            "Giấy xác nhận tình trạng hôn nhân (nếu cần)",
            "Giấy khám sức khỏe trước kết hôn (theo quy định)",
        ],
        "fee": "Miễn phí",
        "processing_time": "Nhận ngay trong ngày làm thủ tục (đủ điều kiện).",
        "steps": [
            "Chuẩn bị hồ sơ đủ theo danh sách.",
            "Nộp tại UBND cấp xã nơi một bên cư trú.",
            "Cán bộ hôn nhân kiểm tra, đối chiếu giấy tờ.",
            "Nếu đủ điều kiện → cấp Giấy xác nhận kết hôn ngay.",
        ],
        "legal_basis": "Luật Hộ tịch 2014; Luật HNGĐ 2014.",
    },
    "business_registration": {
        "title": "Đăng ký doanh nghiệp",
        "competent_authority": "Phòng Đăng ký kinh doanh thuộc Sở KH&ĐT cấp tỉnh.",
        "required_docs": [
            "Giấy đề nghị đăng ký doanh nghiệp (mẫu theo loại hình)",
            "Điều lệ công ty (công ty cổ phần/TNHH 2 thành viên trở lên)",
            "Danh sách thành viên/cổ đông sáng lập",
            "Bản sao hợp lệ CCCD/hộ chiếu của thành viên/cổ đông + người đại diện",
            "Văn bản ủy quyền (nếu qua đại diện)",
        ],
        "fee": "Theo quy định từng tỉnh; cơ bản 100.000 – 3.000.000 VNĐ tùy loại hình.",
        "processing_time": "03 ngày làm việc kể từ khi nhận hồ sơ hợp lệ.",
        "steps": [
            "Chọn loại hình + tên doanh nghiệp (kiểm tra trùng trên Cổng QHTKKQG).",
            "Soạn hồ sơ + ký điện tử.",
            "Nộp online qua Cổng thông tin quốc gia về đăng ký doanh nghiệp.",
            "Nhận Giấy chứng nhận đăng ký doanh nghiệp + công bố nội dung đăng ký.",
            "Khắc con dấu + mở tài khoản ngân hàng + nộp tiền lệ phí.",
        ],
        "legal_basis": "Luật Doanh nghiệp 2020; Nghị định 01/2021/NĐ-CP.",
    },
    "lawsuit_filing_civil": {
        "title": "Khởi kiện dân sự",
        "competent_authority": "Tòa án nhân dân cấp huyện (nơi bị đơn cư trú/có trụ sở).",
        "required_docs": [
            "Đơn khởi kiện (theo mẫu BLTTDS)",
            "Tài liệu, chứng cứ chứng minh yêu cầu (hợp đồng, biên nhận, v.v.)",
            "Bản sao giấy tờ tùy thân nguyên đơn",
            "Giấy ủy quyền (nếu qua người đại diện)",
            "Biên lai nộp tạm ứng án phí",
        ],
        "fee": "Tạm ứng án phí = 50% án phí sơ thẩm (NQ 326/2016).",
        "processing_time": "Thụ lý trong 8 ngày; giải quyết sơ thẩm ~4 tháng (BLTTDS 2015).",
        "steps": [
            "Xác định thẩm quyền tòa án + nơi nộp (nơi bị đơn cư trú).",
            "Soạn đơn khởi kiện + tài liệu chứng cứ.",
            "Nộp tạm ứng án phí tại Kho bạc Nhà nước.",
            "Nộp hồ sơ tại tòa → thụ lý → hòa giải → xét xử.",
        ],
        "legal_basis": "Bộ luật Tố tụng Dân sự 2015; NQ 326/2016.",
    },
    "land_ownership_certificate": {
        "title": "Cấp Giấy chứng nhận quyền sử dụng đất (sổ đỏ/sổ hồng)",
        "competent_authority": "Văn phòng đăng ký đất đai cấp huyện/tỉnh.",
        "required_docs": [
            "Đơn xin cấp GCNQSDĐ theo mẫu",
            "Hồ sơ gốc: giấy tờ pháp lý về đất, nhà",
            "Bản đồ địa chính / trích đo địa chính",
            "Giấy tờ tùy thân",
        ],
        "fee": "Lệ phí địa chính + cấp GCN (tỉnh quy định); lệ phí trước bạ 0,5% nếu chuyển quyền.",
        "processing_time": "≤30 ngày (huyện) / ≤50 ngày (tỉnh) từ nhận hồ sơ hợp lệ.",
        "steps": [
            "Chuẩn bị hồ sơ pháp lý về đất/nhà.",
            "Nộp tại Văn phòng đăng ký đất đai hoặc UBND cấp xã.",
            "Đo đạc, xác minh hiện trạng.",
            "Cấp GCN; ký nhận.",
        ],
        "legal_basis": "Luật Đất đai 2024; Luật Địa chính 2024.",
    },
    "administrative_complaint": {
        "title": "Khiếu nại quyết định hành chính",
        "competent_authority": "Cơ quan có thẩm quyền giải quyết khiếu nại (cơ quan ra quyết định hoặc cấp trên).",
        "required_docs": [
            "Đơn khiếu nại (nội dung, lý do, yêu cầu cụ thể)",
            "Quyết định hành chính bị khiếu nại (bản sao)",
            "Giấy tờ tùy thân người khiếu nại",
        ],
        "fee": "Miễn phí.",
        "processing_time": "Giải quyết lần đầu ≤30 ngày (cấp huyện) / ≤45 ngày (cấp tỉnh).",
        "steps": [
            "Gửi đơn khiếu nại đến cơ quan có thẩm quyền giải quyết.",
            "Cơ quan thụ lý, kiểm tra xác minh.",
            "Ra quyết định giải quyết khiếu nại.",
            "Không đồng ý → khiếu nại lần 2 hoặc khởi kiện hành chính.",
        ],
        "legal_basis": "Luật Khiếu nại 2011; Luật Tố tụng Hành chính 2015.",
    },
    "divorce": {
        "title": "Ly hôn",
        "competent_authority": "Tòa án nhân dân cấp huyện nơi bị đơn cư trú/làm việc.",
        "required_docs": [
            "Đơn xin ly hôn (thỏa thuận hoặc đơn khởi kiện)",
            "Giấy chứng nhận kết hôn (bản gốc)",
            "CCCD hai bên",
            "Giấy khai sinh con (nếu có con chung)",
            "Sổ hộ khẩu + tài liệu chứng cứ tài sản chung (nếu có)",
        ],
        "fee": "Theo yêu cầu: 300.000đ (không có giá) hoặc 5% giá trị tài sản chia.",
        "processing_time": "Thỏa thuận: ~2-3 tháng. Đơn phương: ~4-6 tháng sơ thẩm.",
        "steps": [
            "Thỏa thuận được → nộp hồ sơ thuận tình ly hôn (tòa công nhận).",
            "Đơn phương → nộp đơn khởi kiện tại tòa nơi bị đơn cư trú.",
            "Tòa hòa giải bắt buộc.",
            "Hòa giải không thành → xét xử.",
        ],
        "legal_basis": "Luật HNGĐ 2014; BLTTDS 2015.",
    },
}


def procedure_wizard(procedure_type: str) -> Dict:
    """
    Hướng dẫn thủ tục pháp lý theo loại: hồ sơ, phí, cơ quan thụ lý, bước, thời hạn.

    Args:
        procedure_type: marriage_registration | business_registration | lawsuit_filing_civil
                        | land_ownership_certificate | administrative_complaint | divorce.

    Returns:
        Dict chi tiết thủ tục hoặc danh sách loại hợp lệ.
    """
    key = (procedure_type or "").strip().lower()
    proc = PROCEDURES.get(key)
    if not proc:
        return {
            "error": f"Loại thủ tục '{procedure_type}' không có.",
            "available_types": list(PROCEDURES.keys()),
        }
    return {
        "procedure_type": key,
        **proc,
        "disclaimer": "Thông tin tham khảo; quy định từng địa phương có thể khác. Xác nhận với cơ quan thụ lý.",
    }


def jurisdiction_resolver(dispute_type: str, claim_value: Optional[float] = None, location: str = "") -> Dict:
    """
    Xác định cơ quan/tòa án có thẩm quyền thụ lý một tranh chấp.

    Args:
        dispute_type: civil | criminal | administrative | labor | family | economic.
        claim_value: Giá trị yêu cầu (VNĐ), 0 nếu không có. Ảnh hưởng cấp huyện/tỉnh.
        location: Địa phương (nơi bị đơn cư trú / xảy ra sự kiện) — ảnh hưởng tòa cụ thể.

    Returns:
        Dict cấp tòa/agency thụ lý + căn cứ.
    """
    key = (dispute_type or "").strip().lower()
    jurisdiction_map = {
        "civil": {
            "authority": "TAND cấp huyện (nơi bị đơn cư trú); trên 500 triệu hoặc phức tạp → TAND cấp tỉnh.",
            "general_rule": "Nơi bị đơn cư trú/trụ sở; nếu không rõ → nơi bị đơn làm việc/có tài sản.",
            "legal_basis": "Điều 39, 35, 36, 37 BLTTDS 2015.",
        },
        "criminal": {
            "authority": "TAND cấp huyện xét xử sơ thẩm; tội nghiêm trọng trở lên → TAND cấp tỉnh có thể xét xử.",
            "general_rule": "Nơi tội phạm hoàn thành hoặc nơi bị can cư trú.",
            "legal_basis": "Điều 28, 29 BLTTHS 2015.",
        },
        "administrative": {
            "authority": "TAND cấp huyện (cá nhân khiếu kiện); cấp tỉnh nếu quyết định của cơ quan cấp tỉnh/trung ương.",
            "general_rule": "Nơi cơ quan ra quyết định có trụ sở; khiếu kiện thu hồi đất → cấp tỉnh.",
            "legal_basis": "Điều 11, 12, 14 Luật TTHC 2015.",
        },
        "labor": {
            "authority": "Hội đồng trọng tài lao động (tranh chấp quyền) hoặc TAND (tranh chấp lợi ích).",
            "general_rule": "Nơi bị đơn có trụ sở/cư trú hoặc nơi thực hiện HĐLĐ.",
            "legal_basis": "Điều 32 BLTTDS 2015; Điều 187-192 BLLĐ 2019.",
        },
        "family": {
            "authority": "TAND cấp huyện nơi bị đơn cư trú/làm việc (ly hôn, nuôi con, tài sản chung).",
            "general_rule": "Nơi bị đơn cư trú; đơn phương ly hôn nộp nơi bị đơn cư trú.",
            "legal_basis": "Điều 37, 39 BLTTDS 2015; Luật HNGĐ 2014.",
        },
        "economic": {
            "authority": "TAND cấp tỉnh (tranh chấp kinh tế thương mại) hoặc Trọng tài thương mại.",
            "general_rule": "Theo thỏa thuận trọng tài; nếu không → tòa nơi bị đơn trụ sở.",
            "legal_basis": "Luật Trọng tài thương mại 2010; BLTTDS 2015.",
        },
    }

    entry = jurisdiction_map.get(key)
    if not entry:
        return {"error": "dispute_type không hợp lệ.", "available_types": list(jurisdiction_map.keys())}

    value_note = ""
    if key == "civil" and claim_value is not None and claim_value > 500_000_000:
        value_note = "Yêu cầu trên 500 triệu → TAND cấp tỉnh (có thể)."
    elif claim_value is not None:
        value_note = f"Giá trị yêu cầu {claim_value:,.0f} VNĐ → thường cấp huyện."

    return {
        "dispute_type": key,
        "location": location or "(chưa rõ)",
        "claim_value": f"{claim_value:,.0f} VNĐ" if claim_value is not None else "không có giá",
        **entry,
        "value_note": value_note,
        "disclaimer": "Thẩm quyền còn phụ tình tiết + kháng cáo; xác nhận với tòa tiếp nhận hồ sơ.",
    }


def generate_document_template(doc_type: str, params_json: str = "{}") -> Dict:
    """
    Sinh văn bản pháp lý mẫu từ tham số.

    Mẫu chỉ mang tính tham khảo, cần luật sư rà soát trước khi dùng chính thức.

    Args:
        doc_type: don_khoi_kien_civil | don_khieu_nai_hanh_chinh | hop_dong_mua_ban | don_ly_hon.
        params_json: JSON các trường điền, ví dụ
                     '{"nguyen_don":"Nguyễn Văn A","bi_don":"Trần Thị B","yeu_cau":"buộc trả nợ 100tr","noi_nhan":"TAND quận X"}'.

    Returns:
        Dict với template đã điền + cảnh báo.
    """
    doc_type = (doc_type or "").strip().lower()
    try:
        params = json.loads(params_json) if params_json else {}
    except json.JSONDecodeError:
        return {"error": "params_json không đúng định dạng JSON."}

    def p(key: str, default: str = ".....") -> str:
        return str(params.get(key, default))

    templates = {
        "don_khoi_kien_civil": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "--------------------\n\n"
            "ĐƠN KHỞI KIỆN\n\n"
            "Kính gửi: {noi_nhan}\n\n"
            "Nguyên đơn: {nguyen_don}, sinh năm {nam_sinh}, CCCD: {cccd}\n"
            "Địa chỉ: {dia_chi_nd}\n\n"
            "Bị đơn: {bi_don}, địa chỉ: {dia_chi_bd}\n\n"
            "Yêu cầu: {yeu_cau}\n\n"
            "Nội dung: {noi_dung}\n\n"
            "Kính đề nghị Tòa xem xét, buộc bị đơn {yeu_cau}.\n\n"
            "Nơi nhận: {noi_nhan}\n"
            "Ngày..... tháng..... năm.....\n"
            "                          Nguyên đơn (ký, ghi rõ họ tên)\n"
        ),
        "don_khieu_nai_hanh_chinh": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "--------------------\n\n"
            "ĐƠN KHIẾU NẠI\n\n"
            "Kính gửi: {co_quan_giai_quyet}\n\n"
            "Người khiếu nại: {nguoi_kn}, CCCD: {cccd}, địa chỉ: {dia_chi}\n\n"
            "Quyết định bị khiếu nại: {quyet_dinh} ngày {ngay}\n"
            "Lý do khiếu nại: {ly_do}\n"
            "Yêu cầu: {yeu_cau}\n\n"
            "Đề nghị cơ quan có thẩm quyền xem xét, hủy/bổ sung quyết định trên.\n\n"
            "Ngày..... tháng..... năm.....\n"
            "                          Người khiếu nại (ký, ghi rõ họ tên)\n"
        ),
        "hop_dong_mua_ban": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "--------------------\n\n"
            "HỢP ĐỒNG MUA BÁN TÀI SẢN\n\n"
            "Bên A (bán): {ben_a}, CCCD: {cccd_a}, địa chỉ: {dia_chi_a}\n"
            "Bên B (mua): {ben_b}, CCCD: {cccd_b}, địa chỉ: {dia_chi_b}\n\n"
            "Đối tượng: {tai_san}\n"
            "Giá: {gia} VNĐ\n"
            "Thanh toán: {thanh_toan}\n"
            "Giao nhận: {giao_nhan}\n\n"
            "Điều khoản trách nhiệm, giải quyết tranh chấp theo BLDS 2015.\n\n"
            "Ngày..... tháng..... năm.....\n"
            "    Bên A (ký)              Bên B (ký)\n"
        ),
        "don_ly_hon": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập - Tự do - Hạnh phúc\n"
            "--------------------\n\n"
            "ĐƠN XIN LY HÔN (thuận tình)\n\n"
            "Kính gửi: {noi_nhan}\n\n"
            "Vợ: {vo}, sinh năm {nam_sinh_v}, CCCD: {cccd_v}\n"
            "Chồng: {chong}, sinh năm {nam_sinh_c}, CCCD: {cccd_c}\n\n"
            "Chúng tôi đăng ký kết hôn ngày {ngay_ket_hon} tại {noi_dang_ky}.\n"
            "Con chung: {con_chung}\n"
            "Tài sản chung: {tai_san}\n\n"
            "Vì không còn tình cảm, chúng tôi đồng ý ly hôn và thỏa thuận:\n"
            "- Con: {thoa_thuan_con}\n"
            "- Tài sản: {thoa_thuan_tai_san}\n\n"
            "Kính đề nghị Tòa công nhận thuận tình ly hôn.\n\n"
            "Ngày..... tháng..... năm.....\n"
            "              Vợ (ký)          Chồng (ký)\n"
        ),
    }

    template = templates.get(doc_type)
    if not template:
        return {
            "error": f"doc_type '{doc_type}' không có.",
            "available_types": list(templates.keys()),
        }

    placeholders = {
        "noi_nhan": p("noi_nhan"), "nguyen_don": p("nguyen_don"), "nam_sinh": p("nam_sinh"),
        "cccd": p("cccd"), "dia_chi_nd": p("dia_chi_nd"), "bi_don": p("bi_don"),
        "dia_chi_bd": p("dia_chi_bd"), "yeu_cau": p("yeu_cau"), "noi_dung": p("noi_dung"),
        "co_quan_giai_quyet": p("co_quan_giai_quyet"), "nguoi_kn": p("nguoi_kn"),
        "dia_chi": p("dia_chi"), "quyet_dinh": p("quyet_dinh"), "ngay": p("ngay"),
        "ly_do": p("ly_do"), "ben_a": p("ben_a"), "cccd_a": p("cccd_a"),
        "dia_chi_a": p("dia_chi_a"), "ben_b": p("ben_b"), "cccd_b": p("cccd_b"),
        "dia_chi_b": p("dia_chi_b"), "tai_san": p("tai_san"), "gia": p("gia"),
        "thanh_toan": p("thanh_toan"), "giao_nhan": p("giao_nhan"), "vo": p("vo"),
        "nam_sinh_v": p("nam_sinh_v"), "cccd_v": p("cccd_v"), "chong": p("chong"),
        "nam_sinh_c": p("nam_sinh_c"), "cccd_c": p("cccd_c"), "ngay_ket_hon": p("ngay_ket_hon"),
        "noi_dang_ky": p("noi_dang_ky"), "con_chung": p("con_chung"),
        "thoa_thuan_con": p("thoa_thuan_con"), "thoa_thuan_tai_san": p("thoa_thuan_tai_san"),
    }
    try:
        filled = template.format(**placeholders)
    except KeyError as ke:
        return {"error": f"Template thiếu placeholder: {ke}"}

    return {
        "doc_type": doc_type,
        "document": filled,
        "warning": (
            "Mẫu THAM KHẢO. Trước khi dùng chính thức cần luật sư rà soát tình tiết, "
            "thẩm quyền, và cập nhật theo luật mới. Không thay thế tư vấn pháp lý."
        ),
    }