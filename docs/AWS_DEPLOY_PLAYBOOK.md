# AWS Deploy Playbook — Tư duy Senior AI Dev

> Mục tiêu: giúp bạn **tư duy như 1 senior AI engineer** trước khi đưa chatbot pháp luật (Legal RAG LLM) lên AWS production. Không phải step-by-step copy lệnh (đã có `DEPLOY_AWS.md` + `RUN.md`), mà là **khung quyết định + cổng kiểm tra (readiness gates) + rủi ro + rollback**.
>
> Đọc kèm: `DEPLOY_AWS.md` (architecture EC2), `RUN.md` (ops entry), `CODEBASE_GUIDE.md` (architecture layer), `EVAL_SYSTEM_SUMMARY.md` (harness 7 phase).

---

## 0. Tâm thế senior — 5 câu hỏi trước khi deploy

Trước khi viết 1 dòng Terraform hay `docker compose up`, trả lời:

1. **Ai dùng, dùng khi nào, sai thì hậu quả thế nào?** Legal chatbot = tư vấn pháp luật. Sai → người dùng bị thiệt hại pháp lý. Đây là hệ thống **high-stakes**, không phải toy demo. Mọi quyết định ưu tiên **an toàn > rẻ > đẹp**.
2. **SLA thực sự cần gì?** Latency trả lời chấp nhận được bao nhiêu giây? Uptime target (99%? 99.9%?). Không có SLA → không có monitoring đúng.
3. **Hao tiền ở đâu khi scale?** Đây là app LLM gọi Groq API + BGE-M3 embedding local + Qdrant. Bottleneck không phải CPU mà là **Groq token quota + RAM load embedding model**. Scale sai = đốt tiền vô ích.
4. **Model/agent có "thối" không?** Có. Luật VN đổi (điều luật sửa/bãi bỏ), hành vi user đổi, LLM provider đổi model. → Cần drift detect + re-ingest + regression eval (repo đã có harness P1-P7 + drift).
5. **Rollback khi model mới tệ hơn thì sao?** Phải có cơ chế quay lại prompt/model/embedding cũ **trong phút**, không phải "git revert + rebuild 30 phút".

> **Quy tắc sống còn**: app này = LLM + RAG + agent graph. "MLOps" ở đây không phải train model GPU, mà là **LLMOps**: quản lý prompt/model version, eval khép kín, drift, cost routing, canary, guardrail. Repo đã xây 7 phase eval — **khai thác nó, đừng reinvent**.

---

## 1. Bản đồ hệ thống (nhìn qua lăng kính deploy)

```
┌─────────────────────────────────────────────────────────────┐
│  Client (Streamlit :8501)                                     │
│    └─► POST /chat/complete ─► Celery worker (background)      │
│                  │                                            │
│                  ▼                                            │
│  LangGraph (route→planner→specialist→verify→metacog→END)      │
│        │                                                      │
│        ├──► brain.py: Groq API (Llama 3.1/3.3) / Ollama        │
│        ├──► RAG: Qdrant (vector+BM25) + rerank (Cohere/BGE)   │
│        ├──► Neo4j (legal knowledge graph)                    │
│        ├──► Guardrails (NeMo + PII + disclaimer)             │
│        └──► Trace: MariaDB + Redis pub/sub + OTel            │
│                                                               │
│  Infra sidecar: Redis (broker/cache) · MariaDB (state) ·      │
│                 Qdrant (vector) · Neo4j (graph) · Prometheus  │
│                 + Grafana                                     │
└─────────────────────────────────────────────────────────────┘
```

**Đặc điểm quyết định kiến trúc deploy:**
- LLM = API Groq (stateless, external) → **không cần GPU EC2 cho inference**.
- Embedding BGE-M3 = local → cần RAM ổn (≥8GB). Đây là lý do `DEPLOY_AWS.md` chọn `t3.large`.
- Có background worker Celery → cần **2 process** (API + worker), không phải 1.
- Đã có eval harness + drift + cost routing + canary/shadow → **infra chỉ cần phơi bày** những feature này, không phải xây mới.

---

## 2. Khung quyết định: chọn topology deploy

### 2.1 Ma trận 3 lựa chọn

| Tiêu chí | **A. Single EC2 + Docker Compose** (đã có trong `DEPLOY_AWS.md`) | **B. ECS Fargate + managed DB** | **C. EKS + Helm (như repo Ecommerce recommender)** |
|---|---|---|---|
| Chi phí/tháng | ~$42-66 | ~$120-180 | ~$250+ (cluster control plane) |
| Độ phức tạp ops | Thấp | Trung bình | Cao (K8s + Helm + Istio) |
| Phù hợp giai đoạn | **MVP/early prod** ← bạn đang ở đây | Khi cần auto-scale + HA | Khi đội ≥3 người + multi-model serving |
| Rollback | Stop/restart container | Task definition revision | `kubectl rollout undo` |
| Mất mát khi 1 node chết | Toàn bộ (single point) | Không (Fargate multi-AZ) | Không (multi-node) |

> **Recommend cho bạn**: Bắt đầu **A** (đã có plan sẵn). Chỉ bump lên B khi: traffic thật sự tăng (xem §6 cost), hoặc uptime yêu cầu ≥99.5%, hoặc cần tách DB ra để backup/restore độc lập. **Đừng nhảy C** — K8s overkill cho 1 chatbot, team 1 người sẽ chết maintain.

### 2.2 Quy tắc "managed vs self-hosted"

Mỗi component, hỏi: **"Nếu nó chết lúc 3h sáng, tôi có thể khôi phục trong bao lâu?"**

| Component | Self-host (compose) | Managed AWS | Khi nào chuyển |
|---|---|---|---|
| Qdrant | OK ở A | Amazon OpenSearch (không phải Qdrant) — **không tương thích** | Giữ self-host. Qdrant chưa có managed AWS |
| MariaDB | OK ở A | RDS MariaDB | Khi data pháp luật ≥ 10GB hoặc cần automated backup PITR |
| Redis | OK ở A | ElastiCache | Khi cache hit quan trọng + cần multi-AZ |
| Neo4j | OK ở A (optional) | Amazon Neptune (graph, nhưng phải migrate query) | Khi graph query trở thành bottleneck |
| Embedding | Local trên EC2 | Bedrock/Groq embedding API | Khi RAM EC2 không đủ hoặc muốn tách compute |

> **Cẩn thận**: Qdrant **không có managed equivalent trực tiếp** trên AWS. OpenSearch là nearest nhưng đổi query syntax + mất semantic cache feature (`user_episodes`). Nếu chuyển = refactor `vectorize.py` + `semantic_cache.py`. → Giữ Qdrant self-host đến khi thật sự cần.

---

## 3. Readiness gates — cổng kiểm tra trước khi mở public

**Không mở domain ra internet nếu chưa qua hết 8 cổng.** Senior dev không deploy rồi mới phát hiện.

### Gate 1 — Secrets & config
- [ ] Tất cả key (Groq, Tavily, Cohere, JWT secret, MCP_API_KEY) nằm trong **environment variable / AWS Secrets Manager**, KHÔNG commit vào git.
- [ ] `.env` có trong `.gitignore` (repo có rồi — verify lại).
- [ ] JWT HS256 secret **≥32 byte random**, không phải "secret123".
- [ ] `MCP_ALLOW_NO_AUTH` = false ở prod (stdio dev bypass phải tắt).
- [ ] Grafana admin password đổi từ "admin" mặc định (`GF_SECURITY_ADMIN_PASSWORD`).

### Gate 2 — Security groups / network
- [ ] Chỉ mở public: 80 (HTTP→redirect), 443 (HTTPS).
- [ ] 22 (SSH) restrict về IP cá nhân, **không** `0.0.0.0/0`.
- [ ] 3306 (MariaDB), 6333/6334 (Qdrant), 7474/7687 (Neo4j), 8000/8002 (API) **chỉ bind localhost** / docker internal network, không public. (Lưu ý: `DEPLOY_AWS.md` đã nhấn, nhưng `docker-compose.yml` đang publish port ra host — cần đảm bảo Nginx/SG chặn).
- [ ] CORS whitelist domain thật, không `*`.

### Gate 3 — Eval gate (đây là vũ khí chính)
Repo có harness 7 phase (`EVAL_SYSTEM_SUMMARY.md`). **Trước mỗi deploy**, chạy:
- [ ] `scripts/dev.sh test` — offline gate pass (502 passed baseline).
- [ ] `scripts/dev.sh eval` trên golden set → **regression vs baseline không suy giảm** (`run_eval --baseline`).
- [ ] `scripts/dev.sh redteam` — red-team 6 category không có critical fail.
- [ ] `scripts/dev.sh drift B R` — PSI+KL trong ngưỡng chấp nhận.
- [ ] PII detector (`test_pii_detector.py`) pass — legal app **bắt buộc** không rò rỉ PII.

> **Nguyên tắc**: deploy = promote model/prompt version qua eval gate, giống CI/CD code. Không có eval xanh thì không deploy. CI nightly `eval-live.yml` đã setup — **dùng nó làm gate tự động**.

### Gate 4 — Guardrail & safety
- [ ] `guardrails_manager.verify_input` + `verify_output_rag` bật.
- [ ] `verify_answer` groundedness gate (threshold `VERIFY_ANSWER_THRESHOLD=0.7`) — không trả câu không có bằng chứng.
- [ ] `metacognitive.should_escalate` — HIGH stakes query → escalate luật sư, không tự trả.
- [ ] Legal disclaimer (`add_legal_disclaimer`) luôn kèm output.
- [ ] RBAC: admin/lawyer/user/guest phân quyền tool đúng, sensitive tool qua approval gate.

### Gate 5 — Data persistence & backup
- [ ] EBS volume gp3 attach, dữ liệu Qdrant/MariaDB/Redis mount ra host volume (không mất khi container recreate).
- [ ] **Backup script** MariaDB + Qdrant snapshot chạy cron (ít nhất daily), copy lên S3.
- [ ] Test restore backup ít nhất 1 lần — backup không test = không có backup.
- [ ] Neo4j data (nếu dùng) cũng backup.

### Gate 6 — Observability
- [ ] Prometheus scrape backend metrics (latency, error, tool call).
- [ ] Grafana dashboard có alert: p99 latency, error rate, Celery queue length, Groq quota/429.
- [ ] Trace (MariaDB `AgentStep` + OTel bridge nếu `OTEL_BRIDGE_ENABLED=true`) — debug được "tại sao trả lời này".
- [ ] Log tập trung (Loki hoặc CloudWatch agent) — không chỉ log trong container.

### Gate 7 — Cost & quota
- [ ] Đặt **Groq daily/monthly spend limit** (nếu Groq hỗ trợ) hoặc monitor token usage.
- [ ] `COST_ROUTING_ENABLED=true` ở prod → route `legal_rag` dùng big model, rest small (giảm cost).
- [ ] `SHADOW_MODE_ENABLED` **giữ false** trừ khi A/B test (double Groq cost).
- [ ] Monitor bill AWS + Groq hàng ngày tuần đầu.

### Gate 8 — Rollback & runbook
- [ ] Có script rollback prompt/model version (quay về commit git + Groq model tag cũ).
- [ ] `scripts/dev.sh prod` runbook test trên môi trường staging.
- [ ] Documented: "khi chatbot trả sai, làm gì trong 5 phút" (tắt feature flag → fallback general_chat, hoặc degrade về model nhỏ).

---

## 4. LLMOps cho app này — gì cần, gì không

Repo Ecommerce recommender dùng full MLOps (Feast/Kubeflow/Ray/MLflow/KServe). **App Legal KHÔNG cần cùng bộ đó.** Vì:

| MLOps trụ (recommender) | Cần cho Legal? | Lý do |
|---|---|---|
| Feature Store (Feast) | **Không** | Legal không có feature serving real-time theo user. RAG dùng Qdrant + BM25 thay. |
| Distributed training (Ray/Kubeflow) | **Không** | Không train model. Fine-tune Llama làm 1 lần trên Colab (`colab_llama_finetune.ipynb`), không cần cluster. |
| Model registry (MLflow) | **Có, nhẹ** | Quản lý version của: prompt template, Groq model tag, embedding model, reranker. Có thể dùng MLflow hoặc đơn giản = git tag + env var. |
| Model serving (KServe/Triton) | **Không** | LLM ở Groq API (remote). Embedding local FastAPI. Không cần GPU serving infra. |
| Drift monitoring | **CÓ (đã có)** | `evaluation/drift.py` PSI+KL. Rất quan trọng vì luật đổi. |
| CI/CD model promotion | **Có** | Eval xanh → promote. CI `eval-live.yml` đã làm. |
| Observability | **Có (đã có)** | Prometheus + Grafana + trace. |

> **Senior insight**: LLMOps ≠ MLOps. App này "model" = **prompt + RAG corpus + model tag**. Version chúng, eval chúng, drift chúng. Đó là LLMOps đủ cho production. Đừng kéo K8s/Feast vào chỉ vì "đẹp".

### 4.1 Artifact cần version-control
1. **Prompt templates** (system prompt, planner, verify) — git + hash. `run_metadata.py` đã pin prompt hash.
2. **RAG corpus** — Qdrant snapshot + BM25 cache. Mỗi lần re-ingest (`scripts/reingest_metadata.py`) = version mới.
3. **Model tags** — Groq `llama-3.1-8b-instant`, embedding `BAAI/bge-m3`, reranker. Pin qua env var, không "latest".
4. **Eval golden set** — `data/golden_unified.jsonl` version theo corpus.

---

## 5. Phased rollout — không mở 100% ngày 1

```
Phase R0 (internal)  → chỉ bạn + team test, staging env, domain private
Phase R1 (closed beta) → ~10 user tin cậy, monitor 1 tuần
Phase R2 (soft launch) → mở public nhưng rate-limit, không quảng cáo
Phase R3 (full)       → bỏ rate-limit, mở marketing
```

Mỗi phase có **exit criteria** (đạt mới qua tiếp):
- R0→R1: Eval gate xanh + 0 PII leak + guardrail hoạt động.
- R1→R2: CSAT / 👍👎 ratio ≥ 80% tốt, p99 latency < target, 0 incident cao.
- R2→R3: Drift trong ngưỡng, cost/quota ổn, backup test pass.

> **Canary/shadow đã có**: `SHADOW_MODE_ENABLED=true` chạy candidate song song (double cost) — dùng khi thử model/prompt mới mà không phơi user. Tuyệt vời, tận dụng.

---

## 6. Cost & scale — khi nào biết phải lên kế hoạch

### 6.1 Dấu hiệu cần scale
- Groq 429 (rate limit) thường xuyên → cần request quota increase hoặc cache mạnh hơn.
- p99 latency > 8s → embedding load chậm / Qdrant query chậm / Groq queue.
- Celery queue dài > 50 → worker跟不上, tăng `CELERY_WORKER_CONCURRENCY` hoặc thêm worker.
- RAM EC2 > 80% → embedding + Qdrant ăn RAM, bump instance.

### 6.2 Thứ tự optimize cost (làm theo thứ tự, đừng đảo)
1. **Semantic cache** (`semantic_cache.py`) — cache hit cao = đỡ Groq call. Đây là lever #1, rẻ nhất.
2. **Cost routing** (`COST_ROUTING_ENABLED`) — route đúng kích cỡ model.
3. **Reranker passthrough** khi Cohere quota hết (`RERANKER_TYPE` fallback) — đỡ cost rerank.
4. **Khoản cuối cùng** mới nghĩ bump instance / managed service.

### 6.3 Đơn giá tham khảo (đã có trong `DEPLOY_AWS.md`)
- EC2 `t3.large` ~$60/tháng on-demand, ~$37 savings plan.
- Nếu dùng **chỉ Groq API** (không load embedding local) → `t3.medium` ~$25/tháng đủ.
- Groq: free tier có quota, sau đó pay-per-token — monitor chặt.

---

## 7. Risk matrix — top rủi ro + mitigations

| Rủi ro | Mức | Hậu quả | Mitigation |
|---|---|---|---|
| Chatbot trả sai luật → user tin | **CRITICAL** | Tài sản/thủ tục pháp lý | `verify_answer` gate + `metacognitive` escalate + disclaimer + RBAC approval + eval gate trước deploy |
| PII rò rỉ vào Qdrant/Groq | **CRITICAL** | Phạm luật bảo vệ dữ liệu VN | PII detector (`detect_pii_vietnamese`) + `semantic_cache` privacy filter + `GUARDRAILS_PII_OUTPUT_ENABLED` |
| Groq outage / quota hết | HIGH | App chết | Fallback provider Ollama (đã có `FallbackLLM`) + monitor 429 + alert |
| Qdrant data mất (no backup) | HIGH | Mất corpus pháp luật, re-ingest hàng giờ | EBS volume + S3 snapshot cron + test restore |
| Model/prompt mới tệ hơn | HIGH | Regression âm thầm | Eval regression gate (`run_eval --baseline`) + canary + rollback script |
| Luật mới ban hành, corpus cũ | MEDIUM | Trả lời lỗi thời | Drift detect + re-ingest `scripts/reingest_metadata.py` (set_payload, không re-embed) + `legal_effectivity` classify |
| EC2 đơn điểm chết | MEDIUM | Downtime | Stage A chấp nhận; khi lên B dùng multi-AZ |
| Bill bất ngờ (Groq/bedrock) | MEDIUM | Tốn tiền | Spend limit + daily monitor + `SHADOW_MODE` off mặc định |
| Grafana/MariaDB default password | MEDIUM | Bị leo đặc quyền | Đổi pass (Gate 1) + SG chặn port |

---

## 8. Monitoring — metric phải theo dõi

Dashboard Grafana (đã có compose) tối thiểu cần panel:

**LLM/RAG metric:**
- p50/p95/p99 latency `/chat/complete`
- Groq call count + 429 rate + token usage
- Semantic cache hit ratio
- Rerank fail flag (`rerank_failed=True`) rate
- Tool call accuracy / hallucination rate (eval harness P5)

**Agent metric:**
- Graph recursion hit cap (`GRAPH_RECURSION_LIMIT=32`) — có nghĩa graph loop
- `GraphRunTimeout` rate — degrade graceful kick in
- Handoff count (`MAX_HANDOFF_STEPS=5`) hit cap
- verify_answer "not supported" rate — flag câu không có bằng chứng

**Infra metric:**
- Celery queue length + worker prefetch
- Qdrant query latency + collection size
- MariaDB connection pool
- Redis memory + eviction
- EC2 CPU/RAM/disk

**Alert (PagerDuty/Slack/Email):**
- Error rate > 5% trong 5 phút
- p99 > 10s
- Groq 429 > 10/phút
- Disk > 80%
- Eval nightly fail (CI `eval-live.yml` tạo issue trên GitHub)

---

## 9. Runbook — "khi sự cố xảy ra"

| Tình huống | Làm gì trong 5 phút |
|---|---|
| Chatbot trả sai nhiều | Set env `LLM_MODEL` về tag cũ → restart worker → giữ domain, debug sau |
| Groq 429 flood | Tạm `LLM_PROVIDER=ollama` (local, chậm hơn nhưng sống) + tăng cache TTL |
| Qdrant slow/OOM | Restart `qdrant` container (data trên volume không mất) + check collection size |
| PII rò rỉ phát hiện | Bật `GUARDRAILS_PII_OUTPUT_ENABLED=true` + redact output + audit log + thông báo user |
| Eval nightly đỏ | **Không deploy bản mới**, đọc issue GitHub, reproduce local `scripts/dev.sh eval` |
| Disk full | Xóa `eval_reports/` cũ + Docker `docker system prune` + kiểm tra Qdrant snapshot |

> **Mọi runbook đều test trước**. Không có "hy vọng nó chạy" lúc 3h sáng.

---

## 10. Pre-deploy checklist cuối (print + tick)

```
□ Secrets không commit, JWT secret mạnh, MCP auth bật
□ Security group: chỉ 80/443 public, SSH restrict IP, DB port internal
□ Eval gate xanh (test + eval + redteam + drift)
□ Guardrail + verify_answer + metacog + disclaimer hoạt động
□ RBAC 4 role + approval gate test
□ Backup MariaDB + Qdrant + test restore ≥1 lần
□ Prometheus + Grafana + alert config
□ Grafana/MariaDB password đổi
□ COST_ROUTING on, SHADOW off
□ Rollback script test
□ Rate-limit + phased rollout plan (R0→R3)
□ Spend monitor bật (AWS + Groq)
□ Runbook viết + test trên staging
```

---

## 11. Tóm tắt tư duy senior

1. **High-stakes app** → an toàn trước, rẻ sau. Eval gate + guardrail + escalation không thương lượng.
2. **LLMOps ≠ MLOps** → version prompt/corpus/model tag, eval khép kín, drift, cost routing. Repo đã có 7 phase harness — dùng.
3. **Bắt đầu đơn giản (single EC2)**, có plan bump lên managed khi metric bảo. Đừng K8s quá sớm.
4. **Cổng kiểm tra trước public** — 8 gate + checklist cuối. Không mở domain nếu chưa tick hết.
5. **Rollback trong phút**, không trong giờ. Version mọi artifact.
6. **Monitor LLM-specific** (cache hit, 429, hallucination, recursion cap), không chỉ CPU/RAM.
7. **Phased rollout** — không 100% ngày 1, có exit criteria mỗi phase.
8. **Cost lever theo thứ tự**: cache > routing > rerank fallback > bump infra.

> Bạn không cần trở thành K8s wizard. Bạn cần trở thành người **biết hệ thống sẽ chết ở đâu trước khi nó chết**, và **có nút quay lại khi nó tệ**. Đó là senior.

---

*Tài liệu bổ sung cho `DEPLOY_AWS.md` (how) và `RUN.md` (ops). File này = why + when + what-if.*