"""Lightweight router eval: calls ``detect_route`` directly for the 16-question
legal eval set and compares against expected routes. No full graph, no RAG,
no tools — just the router LLM call, so it is cheap (16 LLM calls).

Run:
    python -m eval_router            # from backend/src
    python backend/src/eval_router.py # from repo root
"""
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv(_SRC.parent / ".env", override=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_router")

from brain import detect_route

# (id, question, expected_route). Mirrors run_question_test.QUESTION_SET so
# this script stays independent of the heavy tasks import.
QUESTION_SET = [
    ("Q01", "Tôi làm việc 3 năm với lương bình quân 15 triệu/tháng, bây giờ nghỉ việc thì nhận bao nhiêu trợ cấp thôi việc?", "agent_tools"),
    ("Q02", "Tôi làm thêm 4 giờ vào ngày nghỉ với lương giờ thường 50k, được trả bao nhiêu tiền làm thêm?", "agent_tools"),
    ("Q03", "Thu nhập tính thuế 20 triệu/tháng thì phải đóng thuế TNCN bao nhiêu?", "agent_tools"),
    ("Q04", "Tôi mua nhà 2 tỷ, lệ phí trước bạ là bao nhiêu?", "agent_tools"),
    ("Q05", "Tôi mua ô tô 1 tỷ lần đầu, lệ phí trước bạ bao nhiêu?", "agent_tools"),
    ("Q06", "Khởi kiện đòi nợ 100 triệu thì án phí dân sự bao nhiêu?", "agent_tools"),
    ("Q07", "Chạy xe máy sau khi uống rượu bị phạt bao nhiêu?", "agent_tools"),
    ("Q08", "Ly hôn thì cần giấy tờ gì, nộp ở đâu?", "agent_tools"),
    ("Q09", "Tranh chấp dân sự 600 triệu thì tòa cấp nào thụ lý?", "agent_tools"),
    ("Q10", "Sinh đơn khởi kiện đòi nợ 100 triệu giúp tôi với", "agent_tools"),
    ("Q11", "Bộ luật Lao động 2019 hiện còn hiệu lực không?", "agent_tools"),
    ("Q12", "Cấp dưỡng 1 con khi thu nhập 10 triệu/tháng là bao nhiêu?", "agent_tools"),
    ("Q13", "Tôi bị khởi tố hình sự về tội trộm cắp, cần bào chữa thế nào?", "agent_tools"),
    ("Q14", "Điều 418 Bộ luật Dân sự 2015 nói về vấn đề gì?", "legal_rag"),
    ("Q15", "Xin chào, bạn có thể giúp gì cho tôi?", "general_chat"),
    ("Q16", "Mức lương tối thiểu vùng năm 2024 mới nhất là bao nhiêu?", "web_search"),
]


def main():
    print(f"{'ID':<5} {'EXPECTED':<13} {'GOT':<13} {'RESULT':<6}  QUESTION")
    print("-" * 110)
    correct = 0
    failures = []
    for qid, question, expected in QUESTION_SET:
        try:
            got = detect_route([], question)
        except Exception as e:
            got = f"ERROR:{e}"
        ok = got == expected
        correct += int(ok)
        mark = "OK" if ok else "FAIL"
        print(f"{qid:<5} {expected:<13} {got:<13} {mark:<6}  {question[:60]}")
        if not ok:
            failures.append((qid, expected, got, question))

    print("-" * 110)
    total = len(QUESTION_SET)
    print(f"Accuracy: {correct}/{total} = {correct/total:.0%}")
    if failures:
        print("\nFAILURES:")
        for qid, exp, got, q in failures:
            print(f"  {qid}: expected={exp} got={got} | {q}")
        sys.exit(1)
    print("\nAll routes correct.")


if __name__ == "__main__":
    main()