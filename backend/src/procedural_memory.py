"""Procedural memory (CoALA procedural layer) — per-case-type workflows.

Returns a short, deterministic step scaffold the agent injects into its system
prompt when the user's ``case_type`` is known (read from ``UserProfile``).
This is NOT a tool and NOT a free-text LLM memory: it is a static playbook the
agent follows so citizens get consistent, correct procedural guidance across
case types (inheritance / land / marriage / business / traffic).

Public API:
- ``workflow_block(case_type) -> str``: the injectable scaffold, or "" if the
  case_type is unknown / None.
- ``CASE_WORKFLOWS``: the dict of case_type -> {"title", "steps": [...]}.
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Workflows — 4-6 bullet steps per case_type, Vietnamese, plain imperative.
# Keep steps generic + safe (no case-specific legal advice, only the procedural
# path: paper, authority, deadline, escalation). Sourced from common practice.
# ---------------------------------------------------------------------------

CASE_WORKFLOWS: Dict[str, Dict] = {
    "inheritance": {
        "title": "Thủ tục thừa kế",
        "steps": [
            "B1: Xác định di sản + danh sách người hưởng kế (theo pháp luật hoặc di chúc).",
            "B2: Chuẩn bị hồ sơ: giấy chứng tử, giấy tờ tài sản, quan hệ nhân thân.",
            "B3: Lập bản phân chia di sản (thỏa thuận hoặc yêu cầu Tòa án nếu tranh chấp).",
            "B4: Công chứng văn bản phân chia + làm thủ tục sang tên tài sản.",
            "B5: Nếu tranh chấp không thỏa thuận → khởi kiện tại Tòa án nhân dân cấp huyện.",
        ],
    },
    "land": {
        "title": "Thủ tục đất đai",
        "steps": [
            "B1: Kiểm tra giấy chứng nhận quyền sử dụng đất + ranh giới thực tế.",
            "B2: Chuẩn bị hồ sơ: sổ đỏ, giấy tờ chuyển nhượng, bản đồ địa chính.",
            "B3: Nộp hồ sơ tại Văn phòng đăng ký đất đai cấp huyện.",
            "B4: Bổ sung giấy tờ nếu cơ quan yêu cầu, theo dõi thời hạn giải quyết.",
            "B5: Nếu tranh chấp ranh giới/người hưởng → hòa giải xã rồi khởi kiện Tòa án.",
        ],
    },
    "marriage": {
        "title": "Thủ tục hôn nhân / ly hôn",
        "steps": [
            "B1: Xác định điều kiện (ly hôn thuận tình hay đơn phương, có con/tài sản chung).",
            "B2: Chuẩn bị hồ sơ: đăng ký kết hôn, giấy khai sinh con, chứng từ tài sản.",
            "B3: Nộp đơn tại UBND cấp xã (thuận tình) hoặc Tòa án (đơn phương / tranh chấp).",
            "B4: Tham gia hòa giải (bắt buộc nếu có con/tài sản) → Tòa án ra phán quyết.",
        ],
    },
    "business": {
        "title": "Thủ tục doanh nghiệp",
        "steps": [
            "B1: Xác định loại hình (TNHH, cổ phần, tư nhân) + ngành nghề kinh doanh.",
            "B2: Chuẩn bị hồ sơ đăng ký: điều lệ, danh sách thành viên, giấy tờ địa điểm.",
            "B3: Nộp hồ sơ tại Sở kế hoạch đầu tư / Cổng đăng ký doanh nghiệp quốc gia.",
            "B4: Nhận giấy chứng nhận + đăng ký mã số thuế + mở tài khoản vốn điều lệ.",
            "B5: Nếu tranh chấp hợp đồng thương mại → Trọng tài hoặc Tòa án kinh tế.",
        ],
    },
    "traffic": {
        "title": "Thủ tục vi phạm / tai nạn giao thông",
        "steps": [
            "B1: Giữ nguyên hiện trường + báo công an giao thông (nếu có thiệt hại).",
            "B2: Lập biên bản vi phạm / tai nạn, ghi nhận chứng cứ + nhân chứng.",
            "B3: Xác định lỗi + mức phạt theo Nghị định 100, 123 (chủ xe / người điều khiển).",
            "B4: Nộp phạt + thực hiện thủ tục bồi thường (thỏa thuận hoặc Tòa án nếu tranh chấp).",
            "B5: Nếu có thương vong → truy cứu trách nhiệm hình sự, cần luật sư bào chữa.",
        ],
    },
    "other": {
        "title": "Hướng dẫn chung",
        "steps": [
            "B1: Xác định rõ sự việc pháp lý + quyền/nghĩa vụ liên quan.",
            "B2: Chuẩn bị hồ sơ + chứng từ liên quan đến sự việc.",
            "B3: Xác định cơ quan thẩm quyền (UBND, cơ quan thuế, Tòa án, Trọng tài).",
            "B4: Nếu tranh chấp → ưu tiên hòa giải trước, khởi kiện khi cần.",
        ],
    },
}


def workflow_block(case_type: str | None) -> str:
    """Return the procedural scaffold for ``case_type``, or "" if unknown.

    The block is a clearly-labeled hint so the agent follows the right path
    without reciting it verbatim. Returns "" for None/unknown case types so the
    system prompt is untouched (no contamination for anonymous/general queries).
    """
    if not case_type:
        return ""
    wf = CASE_WORKFLOWS.get(case_type)
    if not wf:
        return ""
    steps = "\n".join(f"- {s}" for s in wf["steps"])
    return (
        f"\n\n[Quy trình thủ tục — {wf['title']}, chỉ dùng khi câu hỏi liên quan "
        f"đến thủ tục này, KHÔNG lặp lại nguyên văn nếu user không hỏi]:\n{steps}"
    )