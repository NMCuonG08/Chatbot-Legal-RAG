# RUN.md — Trang duy nhất để chạy mọi thứ

> Tóm tắt: đã xây xong **hệ thống eval/harness 7 phase** (P1–P7) cho Legal chatbot. Tất cả code + test đã commit (`9dd0782`→`ef24020`).
> **Một entry duy nhất: `scripts/dev.sh`** — infra / app / test / eval / prod đều qua nó. Ops/MLOps/opsdev gọi cùng path.
> **Onboard nhanh**: đọc `CODEBASE_GUIDE.md` (knowledge-graph dạng markdown — architecture layers + component map + flows + guided tour, sinh bằng method Understand-Anything).

## ⚡ Cheat sheet (1 lệnh / việc)

```bash
scripts/dev.sh up            # boot Redis/MariaDB/Qdrant/Prometheus/Grafana
scripts/dev.sh setup          # (1 lần) pip install + DB schema + Qdrant collections
scripts/dev.sh app            # Celery + FastAPI :8002 + Streamlit :8501 (Ctrl-C stop all)
scripts/dev.sh test           # offline gate (không cần service/key)
scripts/dev.sh gate           # pytest + mypy + black (mirror CI)
scripts/dev.sh phase P1       # chạy test phase P1..P7 riêng
scripts/dev.sh golden         # build data/golden_unified.jsonl
scripts/dev.sh eval           # eval --mode all, judge Groq  (env N, PARALLEL)
scripts/dev.sh eval --ollama  # judge Ollama (không tốn Groq quota)
scripts/dev.sh drift B.json R.json   # PSI+KL drift giữa 2 run
scripts/dev.sh redteam        # promptfoo red-team (cần Node + app :8002)
scripts/dev.sh prod           # full boot: infra -> wait -> seed -> app
scripts/dev.sh ci-check       # mirror .github/workflows/ci.yml
```

`scripts/dev.sh` (không args) in ra full usage. Tất cả lệnh chạy từ repo root, shell bash.

---

## Chi tiết (khi cần override)

## 0. Cài môi trường (1 lần)

```bash
pip install -r requirements_dev.txt      # pytest, deepeval, ragas, scipy, opentelemetry, mypy, black
pip install -r backend/requirements.txt  # app deps (fastapi, langgraph, groq, qdrant, ...)
```

Check `backend/.env` có đủ (xem `backend/.env.example`):
```env
GROQ_API_KEY=gsk_...          # bắt buộc để agent + judge chạy
TAVILY_API_KEY=tvly-...        # web search (optional)
COHERE_API_KEY=...             # rerank (optional, fallback nếu thiếu)
DATABASE_URL=...               # MySQL/Postgres
REDIS_URL=redis://localhost:6379/0
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
JUDGE_PROVIDER=groq
JUDGE_MODEL=llama-3.1-8b-instant
```

---

## 1. Chạy APP (chatbot + API)

### Bước 1: boot services (Redis/MariaDB/Qdrant/Neo4j)
```bash
docker compose up -d
```
Check: `docker compose ps` — 4 service Up.

### Bước 2: seed data pháp luật vào Qdrant
```bash
cd backend/src
PYTHONPATH=. python import_data.py
cd ../..
```

### Bước 3: chạy Celery worker (xử lý chat nền) — terminal riêng
```bash
cd backend/src
celery -A tasks.celery_app worker --loglevel=info -P solo
```

### Bước 4: chạy FastAPI — terminal riêng
```bash
cd backend/src
uvicorn app:app --host 0.0.0.0 --port 8002
```
App: http://localhost:8002  •  Docs: http://localhost:8002/docs

Test chat:
```bash
curl -X POST http://localhost:8002/chat -H "Content-Type: application/json" \
  -d '{"user_message":"Điều 10 Bộ luật Dân sự quy định gì?","sync_request":true}'
```

---

## 2. Chạy TEST (eval harness)

### Offline gate — KHÔNG cần service/key (chạy mọi lúc)
```bash
PYTHONPATH=backend/src:. pytest -q
```
Kết quả mong đợi: `502 passed, 3 skipped, 10 deselected`.
- skipped = Qdrant/COHERE không có (cần service).
- deselected = marker `slow/live/redteam_live/integration` (chỉ chạy live).

### Từng phase
```bash
PYTHONPATH=backend/src:. pytest tests/test_run_metadata.py tests/test_stats.py tests/test_parallel.py tests/test_regression.py -q   # P1
PYTHONPATH=backend/src:. pytest tests/test_judge_panel.py tests/test_pairwise_eval.py -q                                          # P2
PYTHONPATH=backend/src:. pytest tests/test_sim_user.py tests/test_scenarios.py -q                                                 # P3
PYTHONPATH=backend/src:. pytest tests/test_redteam_dataset.py tests/test_redteam_metrics.py tests/test_pii_detector.py tests/test_redteam_deepeval.py -q  # P4
PYTHONPATH=backend/src:. pytest tests/test_slicing.py tests/test_metrics_extended.py tests/test_golden_unified.py -q               # P5
PYTHONPATH=backend/src:. pytest tests/test_cost_routing.py tests/test_drift.py tests/test_otel_bridge.py tests/test_canary_shadow.py -q  # P6
PYTHONPATH=backend/src:. pytest tests/test_ci_markers.py tests/test_deepeval_gate.py -q                                            # P7
```

### Tests bị deselected (deepeval/ragas/slow/live) — cần đã cài + service
```bash
PYTHONPATH=backend/src:. pytest -m "slow or live or redteam_live" -q
```

---

## 3. Chạy EVAL (cần service + Groq key)

### Bước 1: build golden set
```bash
PYTHONPATH=backend/src:. python -c "from evaluation.golden_unified import write_unified_dataset; write_unified_dataset()"
# -> data/golden_unified.jsonl
```

### Bước 2: chạy eval suite (cần app/service đang chạy)
```bash
GROQ_API_KEY=$GROQ_API_KEY JUDGE_PROVIDER=groq JUDGE_MODEL=llama-3.1-8b-instant \
PYTHONPATH=backend/src:. python backend/src/evaluation/run_eval.py \
  --mode all --data-file data/golden_unified.jsonl --parallel 4 --n 20 \
  --output eval_reports/live/manual
```

### Dùng Ollama làm judge (không tốn Groq quota)
```bash
JUDGE_PROVIDER=ollama JUDGE_MODEL=qwen2.5:0.5b \
PYTHONPATH=backend/src:. python backend/src/evaluation/run_eval.py \
  --mode all --data-file data/golden_unified.jsonl --parallel 2 --n 10 \
  --output eval_reports/live/ollama
```

### Regression diff (so baseline)
```bash
PYTHONPATH=backend/src:. python backend/src/evaluation/run_eval.py \
  --baseline eval_reports/runs/<run_id>.json
```

### Drift detection
```bash
PYTHONPATH=backend/src:. python -c "from evaluation.drift import detect_drift; import json; reps=detect_drift('eval_reports/runs/<baseline>.json','eval_reports/runs/<recent>.json'); print(json.dumps([r.__dict__ for r in reps], indent=2, default=str))"
```

### Red-team (promptfoo — cần Node)
```bash
npm install -g promptfoo
PYTHONPATH=backend/src:. python -c "from evaluation.redteam.promptfoo_config import write_promptfoo_config; from evaluation.redteam.dataset import load_redteam_dataset; write_promptfoo_config(load_redteam_dataset(), 'promptfoo_tests.json', agent_endpoint='http://localhost:8002/chat')"
npx promptfoo eval -c promptfoo_tests.json
```

---

## 4. Bật tính năng P6 (opt-in, default off)

```bash
OTEL_BRIDGE_ENABLED=true OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   # OTel trace mirror
COST_ROUTING_ENABLED=true     # route→model (legal_rag big, rest small)
SHADOW_MODE_ENABLED=true      # chạy candidate song song (doubles Groq cost)
```

---

## 5. CI (tự động)

- **Mỗi push/PR** → `.github/workflows/ci.yml` offline gate (pytest + mypy + black). Không cần service.
- **Mỗi đêm 02:07 UTC + manual** → `.github/workflows/eval-live.yml` full live eval + regression + red-team + drift. Cần secrets GitHub: `GROQ_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `JUDGE_MODEL`, `JUDGE_PROVIDER`.

---

## 6. Tôi đã làm gì (tóm tắt 1 dòng/phase)

| Phase | Việc |
|-------|------|
| P1 | pin git sha+model+prompt hash, bootstrap CI/McNemar/Wilcoxon, eval song song, regression gate |
| P2 | judge swap-augment + CoT G-Eval + multi-judge panel + Cohen kappa, pairwise A/B |
| P3 | simulated user (LLM-as-user), τ-bench scenario scoring (r_action+r_output) |
| P4 | red-team 6 category probes, PII detector homegrown, safety metrics |
| P5 | slicing by intent/difficulty/language/oos, tool-call accuracy/noise/hallucination/p99, unified golden |
| P6 | OTel bridge, drift (PSI+KL), cost routing, canary/shadow |
| P7 | pytest.ini marker gate, CI offline, live workflow, deepeval/ragas/promptfoo |

Chi tiết đầy đủ: `EVAL_SYSTEM_SUMMARY.md`. Module map: `backend/src/evaluation/README.md`.

---

## 7. Re-ingest sau Production Upgrade (Phase 1-6)

Upgrade metadata + BM25 tokenizer + KG cross-ref **không train lại model** — embedding giữ nguyên (text không đổi). Chạy 1 lần sau deploy:

```bash
# 1. Backfill Qdrant payload (clause/point/document_*/effectivity) + re-MERGE Neo4j + rebuild BM25.
#    set_payload = không re-embed. Idempotent (re-run chỉ ghi đè cùng field).
PYTHONPATH=backend/src python scripts/reingest_metadata.py --collection llm

#    Skip graph hoặc BM25 nếu chưa cần:
#    python scripts/reingest_metadata.py --no-graph --no-bm25
```

**Verify sau re-ingest**:
- Qdrant payload có field mới (scroll API): `clause_number`, `point_letter`, `document_number`, `document_year`, `document_type`, `effectivity_status`.
- Neo4j edge count theo loại: `MATCH ()-[r:CITES]->() RETURN count(r)` (thay CITES bằng AMENDS/REPEALS/REPLACED_BY).
- BM25 cache rebuild xong (log `BM25 cache rebuilt`).

**Env mới** (xem `.env.example`): `RRF_W_VECTOR`, `RRF_W_BM25`, `RRF_BLEND_ALPHA`, `RRF_TOP_K`, `RRF_TOP_N`, `RERANK_TOP_N`, `RERANKER_TYPE`, `BGE_RERANK_MODEL`, `BGE_RERANK_DEVICE`.

**Test**:
```bash
python -m pytest tests/test_legal_metadata.py tests/test_legal_effectivity.py \
  tests/test_graph_relations.py tests/test_graph_memory.py tests/test_citations.py -q
```

---

## Lỗi thường gặp

| Lỗi | Fix |
|------|-----|
| `ModuleNotFoundError: evaluation` | thêm `PYTHONPATH=backend/src:.` |
| `GROQ_API_KEY` thiếu | set trong `backend/.env` hoặc export trước lệnh |
| Qdrant connection refused | `docker compose up -d` rồi chờ `curl localhost:6333/healthz` |
| 9 test module collection error (`ModelProfile`) | pre-existing langchain-groq mismatch, đã ignore trong gate — bình thường |
| `deepeval`/`ragas` skip | `pip install -r requirements_dev.txt` rồi chạy lại |