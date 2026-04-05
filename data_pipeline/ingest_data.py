import json
from pathlib import Path
from datasets import load_dataset

def ingest_legal_documents(output_file="data/raw_legal_documents.jsonl", split="train"):
    """
    Tải dữ liệu YuITC/Vietnamese-Legal-Documents và lưu về file JSONL thô.
    Args:
        output_file: Đường dẫn file output JSONL
        split: Tên split (train/test)
    """
    print(f"📥 Đang tải YuITC/Vietnamese-Legal-Documents ({split}) ...")
    ds = load_dataset("YuITC/Vietnamese-Legal-Documents")
    data = ds[split]
    print(f"Tổng số records: {len(data)}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')
    print(f"✅ Đã lưu dữ liệu thô vào {output_file}")

if __name__ == "__main__":
    ingest_legal_documents()
