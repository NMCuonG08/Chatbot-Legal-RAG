# Prompt Versioning (LLMOps Tier 1.2)

Tách prompt inline (hardcoded trong `agent.py`/`tasks.py`) ra file riêng + version. Rollback prompt = đổi env var, không redeploy code.

## Cài đặt

```bash
# 1. Copy loader
cp deployment/prompts/prompt_loader.py backend/src/prompt_loader.py

# 2. Tạo prompts dir + copy ví dụ
mkdir -p backend/prompts
cp deployment/prompts/agent_system.v1.txt backend/prompts/

# 3. pyyaml (thường đã có): pip install pyyaml
```

## Wire vào code (refactor nhẹ)

**`backend/src/agent.py`** — hàm `_get_agent_system_prompt` (line 231):
```python
# BEFORE (inline):
base = f"""Bạn là trợ lý AI...{current_date}...{current_year}..."""

# AFTER:
from prompt_loader import load_prompt, prompt_version
base = load_prompt("agent_system", current_date=current_date, current_year=current_year)
_log.info("prompt agent_system=%s", prompt_version("agent_system"))
```

**`backend/src/tasks.py`** — các `system_prompt = """..."""` tại line 628, 769, 792, 1035, 1612:
tạo file tương ứng `generate.v1.txt`, `web_search.v1.txt`, `general_chat.v1.txt`, `planner.v1.txt`, `summarize.v1.txt` rồi `load_prompt("<name>")`.

## Quy ước đặt tên file

```
backend/prompts/
  agent_system.v1.txt        <- nội dung prompt (str.format placeholders OK)
  agent_system.v1.meta.yaml  <- optional: changelog, author, eval_run_id, diff vs v0
  agent_system.v2.txt        <- bản mới
```

## Chọn version (thứ tự ưu tiên)

1. `PROMPT_AGENT_SYSTEM=v2` env → override (test A/B nhanh)
2. `backend/models.yaml` → `prompts.agent_system: "v2"` (registry trung tâm)
3. default `v1`

## Workflow đổi prompt

```
1. Tạo backend/prompts/agent_system.v2.txt (copy v1, sửa)
2. (optional) v2.meta.yaml: changelog "rephrase grounding section"
3. Chạy eval: scripts/dev.sh eval --baseline <old_run>
4. Nếu metrics không suy giảm: bump models.yaml prompts.agent_system = "v2"
5. Commit + tag → CI deploy
6. Nếu prod tệ: set PROMPT_AGENT_SYSTEM=v1 env + restart worker (rollback ~30s)
```

## Template variables

Dùng `{var}` + `str.format(**kwargs)`. KHÔNG dùng f-string trong file (loader tự format).
Lưu ý: nếu prompt có literal `{` `}` (như JSON example), escape `{{` `}}`.

## Traps

- **Không quên verify_citation / legal_disclaimer** khi tạo v2 — safety gate, đừng lỡ xóa.
- **Diff prompt phải đi kèm eval** — đổi 1 chữ cũng có thể break grounding.
- **Prompt hash pin**: `run_metadata.py` đã pin prompt hash. Khi tách ra file, hash file = hash prompt → regression vẫn work.