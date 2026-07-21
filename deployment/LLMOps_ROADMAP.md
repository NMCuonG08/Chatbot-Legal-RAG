# LLMOps Roadmap — Legal Chatbot

> Từ trạng thái hiện tại (đã có 70% LLMOps) lên production-grade. 4 tuần.
> Mỗi task: WHAT + files + verify. Không reinvent — repo đã có harness 7 phase.

## Trạng thái hiện tại (đã có — KHÔNG làm lại)

| Trụ | File | Trạng thái |
|---|---|---|
| Eval harness 7 phase | `backend/src/evaluation/*` | ✅ |
| Drift (PSI+KL) | `evaluation/drift.py` | ✅ |
| Regression gate | `run_eval --baseline` | ✅ |
| LLM-as-judge + pairwise | `judge_panel.py` | ✅ |
| Guardrail + PII | `guardrails_manager.py` | ✅ |
| Verify answer (faithfulness) | `verify_answer.py` | ✅ |
| RBAC + approval | `rbac.py`, `approval.py` | ✅ |
| Trace + OTel | `trace.py`, `otel_bridge.py` | ✅ |
| Cost routing | `cost_routing.py` | ✅ |
| Canary/shadow | `SHADOW_MODE_ENABLED` | ✅ |
| Prompt hash pin | `run_metadata.py` | ✅ |
| CI offline + nightly eval | `.github/workflows/ci.yml`, `eval-live.yml` | ✅ |

---

## Tuần 1 — Tier 1.1 + 1.2: model registry + prompt versioning

### 1.1 Model registry nhẹ
- **What**: version embedding/reranker/LLM tag/judge/prompt trong 1 file.
- **Files**:
  - `cp deployment/model_registry/models.yaml.example backend/models.yaml`
  - `cp deployment/model_registry/model_registry.py backend/src/model_registry.py`
  - Wire `backend/src/app.py`: `from model_registry import log_versions, get_versions; log_versions()`
- **Verify**: boot app → log `[model_registry] active versions: {...}`. `get_versions()` return dict.
- Xem: `deployment/model_registry/README.md`

### 1.2 Prompt tách ra file
- **What**: prompt inline → file + version. Rollback prompt = env var.
- **Files**:
  - `cp deployment/prompts/prompt_loader.py backend/src/prompt_loader.py`
  - `mkdir backend/prompts && cp deployment/prompts/agent_system.v1.txt backend/prompts/`
  - Refactor `agent.py:231 _get_agent_system_prompt` → `load_prompt("agent_system", current_date=..., current_year=...)`
  - Refactor `tasks.py:628,769,792,1035,1612` → `load_prompt("<name>")`
- **Verify**: `scripts/dev.sh test` pass. Chat 1 query → trace có `prompt_version`. Set `PROMPT_AGENT_SYSTEM=v1` → reload dùng v1.
- Xem: `deployment/prompts/README.md`

---

## Tuần 2 — Tier 1.3 + 1.4: CI/CD GHCR + Terraform

### 1.3 Container registry + CD
- **What**: CI build+push image GHCR, EC2 pull (không build local).
- **Files**:
  - `cp deployment/github-actions/deploy.yml .github/workflows/deploy.yml`
  - GitHub Secrets: `GHCR_PAT`, `EC2_HOST`, `EC2_SSH_KEY`, `AWS_*`
  - Cập nhật prod compose: image thay vì `build:` (xem `deployment/docker-compose.prod.yml`)
- **Verify**: `git tag v1.0.0 && git push --tags` → Actions build+push+SSH deploy. EC2 `docker ps` thấy image mới.
- Rollback: `docker pull ghcr.io/<owner>/legal-backend:<old-sha>` + `docker compose up -d --no-build`.

### 1.4 Terraform IaC
- **What**: EC2+EBS+EIP+SG+S3 bằng code, không click-ops.
- **Files**: `deployment/terraform/` (main.tf, variables.tf, tfvars.example, user_data.sh)
- **Verify**: `terraform apply` → EC2 up, SSH được, app dir clone sẵn, docker cài.
- Xem: `deployment/terraform/README.md`

---

## Tuần 3 — Tier 2.5 + 2.6: golden version + canary traffic

### 2.5 Golden set version theo corpus
- **What**: `golden_unified_<corpus_sha>.jsonl` + baseline eval pinned theo corpus.
- **Files**: sửa `evaluation/golden_unified.py` thêm suffix corpus_sha; `models.yaml rag.corpus_sha`.
- **Verify**: re-ingest → corpus_sha đổi → eval nightly so baseline corpus hiện tại.

### 2.6 Canary traffic split (thật, không chỉ shadow)
- **What**: 5% traffic → model/prompt mới, 95% → stable. Auto promote/rollback theo CSAT+latency.
- **Files**: mở rộng `cost_routing.py` thành traffic split; thêm `evaluation/canary_decision.py`.
- **Verify**: 2 version chạy song song, dashboard thấy split %, alert khi candidate tệ.

---

## Tuần 4 — Tier 2.7 + 2.8: feedback loop + alert receiver

### 2.7 Feedback → eval
- **What**: 👎 rate cao trên 1 loại query → auto thêm vào redteam probes, đánh lại eval nightly.
- **Files**: `evaluation/feedback_to_eval.py` đọc `rlhf_store.py` 👎 → append `probes.jsonl`.
- **Verify**: 👎 nhiều → nightly eval có probe mới.

### 2.8 Alert receiver thật
- **What**: Grafana alert rule → SNS → email/Slack.
- **Files**: `monitoring/prometheus/rules/alerts.yml` đã có rule; wire Alertmanager `targets` → SNS.
- **Verify**: trigger alert thủ công → email nhận.

---

## Tier 3 — Nice to have (sau tuần 4)

| # | What | Khi nào |
|---|---|---|
| 9 | MLflow full (registry trung tâm + UI) | ≥3 model tag song song + team ≥2 |
| 10 | Auto re-ingest khi drift (EventBridge trigger) | drift > ngưỡng → re-ingest → eval → **human approve** → promote |
| 11 | Langfuse (prompt observability UI) | muốn UI đẹp quản lý prompt + trace |

> ⚠️ **Auto re-train/auto-promote KHÔNG human review** = cấm với app pháp luật. Eval pass → human approve → promote.

---

## Traps tránh

- **Đừng kéo K8s/Helm/Kubeflow** — app LLM gọi Groq API, không serve model local. Overkill.
- **Đừng self-host MLflow** nếu `models.yaml` + git tag đủ — thêm service = thêm gánh.
- **Đừng auto-promote không human review** — app pháp luật, sai = hậu quả pháp lý.
- **Prompt inline = nợ kỹ thuật** — ưu tiên tách (Tuần 1.2) trước.

## Success metric (sau 4 tuần)

- [ ] Mỗi chat response mang version set (registry + prompt) trong trace
- [ ] `git tag vX.Y.Z` → prod tự deploy trong 5 phút, rollback trong 1 phút
- [ ] `terraform apply` recreate infra 1 lệnh
- [ ] Canary split 5/95 chạy, auto decision theo metric
- [ ] 👎 feedback nuôi eval nightly
- [ ] Alert email nhận khi p99>10s / 429 / eval fail

→ Đây = **LLMOps production-grade**, vượt bản gốc `Vietnamese-Legal-Chatbot-RAG-System` nhiều.