# RAG Evaluation Suite

Module đánh giá chất lượng pipeline RAG của hệ thống chatbot pháp luật.

## Tại sao đo những metric này

Pipeline RAG có hai pha tách biệt — **retrieval** (lấy đúng tài liệu chưa?) và **generation** (trả lời tốt từ tài liệu đó chưa?). Một câu trả lời sai có thể đến từ retrieval kém hoặc generation kém, đo riêng giúp khoanh vùng lỗi.

### Retrieval metrics

Ground truth: `data/train.jsonl` cho mỗi câu hỏi đã có đúng một đoạn `context` là gốc — coi đó là tài liệu vàng.

| Metric | Ý nghĩa | Khi nào nó tốt |
|---|---|---|
| **Hit@K** | Trong top-K kết quả có tài liệu vàng không (0/1) | Càng cao càng tốt, mục tiêu thực tế: Hit@5 ≥ 0.85 |
| **Recall@K** | Tỷ lệ tài liệu vàng nằm trong top-K | Tương tự Hit@K khi mỗi câu chỉ có 1 gold doc |
| **MRR** | Trung bình của 1/rank của tài liệu vàng | Đo việc "đẩy lên top". MRR ≥ 0.7 là tốt |
| **nDCG@K** | Discounted Cumulative Gain — ưu tiên rank cao | Chuẩn IR học thuật, [0..1] |

### Generation metrics (LLM-as-judge)

Dùng chính LLM (Groq llama-3.1) làm "trọng tài" chấm điểm câu trả lời. Đây là cách tiếp cận của Ragas và TruLens, nhưng được implement trực tiếp để chạy được tiếng Việt và không phụ thuộc OpenAI.

| Metric | Ý nghĩa |
|---|---|
| **Faithfulness** | Câu trả lời có bịa không, có grounded vào tài liệu không |
| **Answer Relevance** | Câu trả lời có đúng câu hỏi không, hay lạc đề |
| **Context Precision** | Trong các đoạn được retrieve, bao nhiêu phần trăm thật sự liên quan |

Thang điểm 0-1, mỗi đánh giá có rationale ngắn để audit.

### End-to-end metrics

Chạy thực tế qua `tasks.run_chat_graph` (LangGraph) để bao gồm cả phần routing.

| Metric | Ý nghĩa |
|---|---|
| **success_rate** | Tỷ lệ trả về câu trả lời không rỗng, không lỗi |
| **latency_ms_mean / p95** | Độ trễ trung bình và p95 |

## Kiến trúc module

```
backend/src/evaluation/
├── __init__.py
├── dataset.py              # load + sample train.jsonl thành EvalSample
├── metrics_retrieval.py    # Hit@K, Recall@K, MRR, nDCG@K
├── metrics_generation.py   # judge prompts + evaluate_*
├── eval_retrieval.py       # ablation 5 cấu hình retrieval
├── eval_generation.py      # retrieval + generate + judge
├── eval_e2e.py             # chạy qua chat graph
├── run_eval.py             # CLI orchestrator
└── README.md
```

### 5 cấu hình retrieval được so sánh

| Config | Bao gồm |
|---|---|
| `vector` | Chỉ dense vector search (Qdrant) |
| `hybrid` | BM25 + vector |
| `hybrid_rerank` | Hybrid + Cohere rerank |
| `multi_query` | Multi-query rewriting + hybrid |
| `full` | Multi-query + hybrid + rerank (đường production) |

So sánh 5 config trên cùng test set để biết từng thành phần đóng góp bao nhiêu — đây là **ablation study**, cách chuẩn để evaluate RAG.

## Yêu cầu trước khi chạy

1. Backend services phải sẵn sàng:
   - Qdrant chạy ở `QDRANT_URL` (default `http://localhost:6333`)
   - Custom embedding service ở `CUSTOM_EMBEDDING_API_URL` (default `http://localhost:5000`) — hoặc bật `USE_LOCAL_EMBEDDING_FALLBACK=true` để dùng fallback hash-based
   - Vietnamese LLM API ở `VIETNAMESE_LLM_API_URL` cho phần generation
   - Groq API key ở `GROQ_API_KEY` cho judge và fallback

2. Đã import data vào Qdrant ít nhất một lần:
   ```cmd
   cd backend\src
   python import_data.py --limit 5000
   ```

3. Activate venv và cài deps:
   ```cmd
   .venv\Scripts\activate
   pip install -r backend\requirements.txt
   ```

## Cách chạy

CLI là `backend/src/evaluation/run_eval.py`. Chạy trực tiếp với `python` từ thư mục `backend/src`:

### Retrieval ablation (nhanh, không tốn LLM)

```cmd
cd backend\src
python -m evaluation.run_eval --mode retrieval --n 50 --output ..\..\eval_reports
```

Khoảng 50 câu × 5 config × ~1-2s/query ≈ 5 phút. Output: bảng so sánh 5 config với Hit@K, MRR, nDCG.

### Generation eval (tốn LLM, mỗi sample 4 lần gọi LLM)

```cmd
python -m evaluation.run_eval --mode generation --n 30 --output ..\..\eval_reports
```

30 câu × (1 generate + 3 judge calls) ≈ 120 LLM calls. Chú ý rate limit của Groq.

### Full pipeline (chậm nhất)

```cmd
python -m evaluation.run_eval --mode all --n 30 --output ..\..\eval_reports
```

### Chỉ một số config retrieval

```cmd
python -m evaluation.run_eval --mode retrieval --n 100 --configs vector hybrid full
```

### Tham số

| Flag | Mặc định | Mô tả |
|---|---|---|
| `--mode` | `retrieval` | `retrieval` / `generation` / `e2e` / `all` |
| `--n` | `50` | Số câu hỏi sample |
| `--seed` | `42` | Random seed (reproducibility) |
| `--top-k` | `10` | Lấy top-K cho retrieval |
| `--configs` | tất cả | Subset của 5 config retrieval |
| `--output` | `eval_reports` | Thư mục xuất `report.json` và `report.md` |
| `--data-file` | `data/train.jsonl` | Override file ground truth |
| `--verbose` | False | Log debug |

## Output

Sau mỗi lần chạy, thư mục `--output` sẽ có:

- `report.json` — toàn bộ raw kết quả: per-sample ranks, scores, rationales, latency
- `report.md` — báo cáo gọn dạng bảng

Ví dụ một bảng retrieval ablation:

```
| config        | n  | hit@1 | hit@5 | mrr   | lat_mean_ms |
|---------------|----|-------|-------|-------|-------------|
| vector        | 50 | 0.620 | 0.820 | 0.713 | 145.2       |
| hybrid        | 50 | 0.700 | 0.880 | 0.781 | 198.5       |
| hybrid_rerank | 50 | 0.760 | 0.900 | 0.825 | 411.8       |
| multi_query   | 50 | 0.720 | 0.900 | 0.802 | 612.3       |
| full          | 50 | 0.800 | 0.920 | 0.851 | 832.1       |
```

Đọc bảng:
- `full` cho retrieval tốt nhất nhưng chậm 5.7x so với `vector` thuần.
- `hybrid` đã đủ tốt cho hầu hết câu hỏi nếu cần latency thấp.
- `hybrid_rerank` là sweet spot — gần với `full` nhưng nhanh hơn 2x.

## Cách module chấm điểm

### Match gold context — `dataset.gold_in_retrieved`

Một retrieved chunk được coi là khớp với gold context khi:
1. Hash giống y nhau (chunk = gold), HOẶC
2. Một bên là substring của bên kia (case-insensitive).

Lý do: pipeline ingest split document thành nhiều chunk, nên gold context (đoạn dài) thường chứa retrieved chunk (đoạn ngắn) hoặc ngược lại. Cách match này ổn cho hầu hết trường hợp; nếu muốn chặt hơn có thể chuyển sang ROUGE-L hay BERTScore.

### LLM judge

Mỗi prompt có format yêu cầu LLM trả về JSON `{"score": 0..1, "reason": "..."}`. Module có parser lenient để xử lý JSON không chuẩn. Khi judge fail, score = 0 và rationale ghi `judge_error`.

## Mở rộng

### Thêm metric mới

Thêm hàm vào `metrics_retrieval.py` hoặc `metrics_generation.py`, rồi gọi từ `eval_*.py` tương ứng và update bảng trong `run_eval._build_markdown_report`.

### Thêm config retrieval mới

Trong `eval_retrieval.py`, thêm hàm `_config_<tên>` rồi đăng ký vào `CONFIGS` dict. Sẽ tự động xuất hiện trong CLI.

### Multi-relevant ground truth

`recall_at_k` đã có tham số `total_relevant`. `gold_in_retrieved` cần đổi sang trả về list of ranks. Hiện tại assume 1-relevant để khớp với schema train.jsonl.

## Hạn chế đã biết

- LLM-as-judge phụ thuộc Groq API; nếu rate-limit thì kết quả generation eval sẽ thiếu.
- Match substring có thể false-positive khi 2 đoạn ngắn đều phổ biến (ví dụ "Theo Điều 1"). Với eval pháp luật thì các chunk đủ dài nên ít khi gặp.
- Cohere rerank cần `COHERE_API_KEY`. Không có key thì `hybrid_rerank` và `full` sẽ tự fallback về top-N của input (đo hơi không công bằng — module sẽ vẫn chạy, chỉ là score sẽ giống `hybrid`).
- Full eval với 89k câu là không khả thi về cost LLM. Default n=50 đủ tin cậy thống kê cho retrieval; n=30 cho generation đã tốn ~120 LLM calls.

## Judge hardening & pairwise — hạn chế (P2)

- **Self-preference / cross-family hạn chế**: judge chỉ Groq (Llama) + Ollama (Llama/glm nếu cấu hình). Không có key OpenAI/Anthropic nên panel cùng họ Llama → ít đa dạng gia đình. Mitigate: swap augmentation (position bias), CoT-before-score (G-Eval), multi-judge panel, Cohen's kappa calibration. Khuyến nghị: khi có key cross-family, thêm judge thứ ba.
- **Swap augmentation chi phí gấp đôi**: `swap_augment_pairwise` gọi judge 2 lần/sample → cost x2. Tắt nếu chỉ survey nhanh.
- **nest_asyncio**: `run_chat_graph` sync, mỗi thread trong `ThreadPoolExecutor` có event loop riêng → không cần `nest_asyncio`. Nếu node dùng `asyncio.get_event_loop().run_until_complete` trên loop đang chạy, patch sang `asyncio.run` (flag, chỉ fix khi test fail).
- **Pairwise A/B**: `run_pairwise_eval` pin provider/model qua contextvars (`LLM_PROVIDER_CONTEXTVAR`/`LLM_MODEL_CONTEXTVAR`); không mutate env. Swap inconsistency → tie. Sign test (binomtest) cho p-value, bootstrap CI cho win rate.
