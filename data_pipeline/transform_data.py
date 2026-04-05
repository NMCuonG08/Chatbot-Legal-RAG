import json
from pathlib import Path

def convert_to_qa_jsonl(input_file, output_file, context_join_sep=" "):
    """
    Chuyển đổi file dữ liệu thô sang định dạng QA JSONL chuẩn.
    Args:
        input_file: File JSONL thô (mỗi dòng là 1 dict)
        output_file: File output QA JSONL
        context_join_sep: Ký tự nối context nếu là list
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line)
            question = item.get("question", "").strip()
            if not question:
                continue
            context = item.get("context", "")
            context_list = item.get("context_list", None)
            if context_list and isinstance(context_list, list):
                context = context_join_sep.join(str(c).strip() for c in context_list if c)
            elif isinstance(context, list):
                context = context_join_sep.join(str(c).strip() for c in context if c)
            context = str(context).strip()
            if not context:
                continue
            qa_pair = {"question": question, "context": context}
            fout.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
            count += 1
            if count % 100 == 0:
                print(f"  Đã xử lý {count} records...")
    print(f"✅ Đã lưu {count} records QA vào {output_file}")

if __name__ == "__main__":
    # Ví dụ: chuyển từ raw_legal_documents.jsonl sang train.jsonl
    convert_to_qa_jsonl(
        input_file="data/raw_legal_documents.jsonl",
        output_file="data/train.jsonl",
        context_join_sep=" "
    )
