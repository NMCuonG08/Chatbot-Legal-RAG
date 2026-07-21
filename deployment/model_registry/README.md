# Model Registry (LLMOps Tier 1)

Version nhẹ cho: embedding, reranker, LLM tag, judge, prompts, RAG corpus. Không cần MLflow cho app 1 người — YAML + git tag đủ.

## Cài đặt

```bash
# 1. Copy registry file
cp deployment/model_registry/models.yaml.example backend/models.yaml
nano backend/models.yaml   # fill corpus_sha, bump prompt versions khi đổi

# 2. Copy loader vào source
cp deployment/model_registry/model_registry.py backend/src/model_registry.py

# 3. Wire vào app (backend/src/app.py, sau imports):
#    from model_registry import log_versions, get_versions
#    log_versions()
#    APP_VERSIONS = get_versions()   # dùng khi emit trace / metadata

# 4. pyyaml (thường đã có): pip install pyyaml
```

## Workflow promotion (đưa model/prompt mới lên prod)

```
1. Đổi code/prompt/corpus local
2. Chạy eval gate: scripts/dev.sh test && scripts/dev.sh eval
3. Regression vs baseline KHÔNG suy giảm
4. Bump version trong backend/models.yaml (vd agent_system v2 → v3)
5. Commit + git tag v1.2.0 → CI push image + deploy
6. Verify prod: trace có version mới, 1 query test
7. Nếu tệ: revert models.yaml + redeploy tag cũ (rollback ~1 phút)
```

## Tích hợp trace (biết câu nào do version nào trả)

Trong `tasks.py:llm_handle_message` (đã có run_metadata), thêm:
```python
from model_registry import get_versions
meta = {**existing_meta, **get_versions()}   # emit_run_start / build metadata
```
→ Mỗi `AgentStep`/`GraphRun` mang version set → debug "câu này sai, version nào" chính xác.

## Khi nào nâng lên MLflow full?

Khi có ≥3 model tag song song + cần UI so sánh metric lịch sử + team ≥2. Lúc đó self-host MLflow (compose thêm 1 service), log `get_versions()` + eval metrics. Hiện overkill.