# Data Pipeline cho Legal RAG

Pipeline này giúp tự động hóa các bước xử lý dữ liệu phục vụ huấn luyện và đánh giá mô hình Legal RAG.

## Cấu trúc thư mục

- `data/`: Chứa dữ liệu gốc và dữ liệu đã xử lý
	- `finetune_data/`, `finetune_data2/`: Dữ liệu huấn luyện, test, metadata
- `utils/`: Chứa các script xử lý dữ liệu (merge, preprocess, ...)
- `run_pipeline.py`: Script orchestration chạy toàn bộ pipeline
- `tests/`: Chứa test cho các bước pipeline


## Các bước pipeline chuẩn

1. **Ingest (Thu thập dữ liệu gốc)**
	- Script: `ingest_data.py`
	- Output: `data/raw_legal_documents.jsonl`
2. **Transform (Tiền xử lý & chuyển đổi format)**
	- Script: `transform_data.py`
	- Input: `data/raw_legal_documents.jsonl`
	- Output: `data/train.jsonl` (chuẩn QA)
3. **Gộp dữ liệu instruction format**
	- Script: `utils/merge_instruction_data.py`
	- Output: `data/finetune_llm_data.jsonl`
4. (Có thể mở rộng thêm các bước preprocess, kiểm thử, chia tập, ...)

## Hướng dẫn sử dụng

### 1. Cài đặt phụ thuộc

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # để chạy test
```


### 2. Chạy từng bước pipeline

#### a. Ingest dữ liệu gốc
```bash
python ingest_data.py
```

#### b. Transform sang định dạng QA chuẩn
```bash
python transform_data.py
```

#### c. Gộp instruction format (nếu cần)
```bash
python utils/merge_instruction_data.py
```

#### d. (Tùy chọn) Chạy toàn bộ pipeline tự động
```bash
python run_pipeline.py
```

### 3. Chạy test kiểm thử pipeline

```bash
pytest tests/
```

## Ghi chú
- Có thể mở rộng pipeline bằng cách thêm các bước vào `run_pipeline.py`.
- Nên tách biệt dữ liệu gốc, dữ liệu đã xử lý, và output nếu pipeline phức tạp hơn.
- Đã bổ sung kiểm thử đơn giản cho bước merge instruction format.
