# Plan: Sửa lỗi privacy semantic_cache (CRITICAL)

> Tạo: 2026-07-02. Làm mai. Lỗi CRITICAL — rò rỉ dữ liệu cá nhân giữa user.

## Bối cảnh lỗi

`backend/src/semantic_cache.py`: cache Qdrant collection `semantic_cache` keyed **chỉ theo embedding câu hỏi** (`score_threshold=0.95`), không có `user_id`/scope.

- `get_cached_response(question)` (line 52): search không filter user → User A cache answer private → User B hỏi tương tự → nhận answer của User A. **Cross-user leak.**
- `set_cached_response(question, response, sources)` (line 94): payload `{query, response, sources, cached_at}` — không scope.
- Call sites `tasks.py:1124` + `tasks.py:1197` chỉ truyền `question`, bỏ `user_id` (dù `user_id` có sẵn scope, `tasks.py:1111`).

Mức: **CRITICAL** (security rule: fix trước commit).

## Mục tiêu

1. Cache entry của user X **không bao giờ** trả cho user Y.
2. Cho phép cache chung (`scope="common"`) opt-in cho câu general (chỉ chào/hướng dẫn, không private) để giữ hit rate.
3. Backward compat với legacy point (không có scope field).
4. Test isolation: User A write → User B read = miss.

## Thay đổi

### 1. `backend/src/semantic_cache.py`

- Payload thêm field `scope`: giá trị `"user:<user_id>"` (mặc định) hoặc `"common"`.
- `get_cached_response(question: str, user_id: Optional[str] = None) -> Optional[Dict]`:
  - Build Qdrant `Filter` với `should=[scope="common", scope=f"user:{user_id}"]` (min_should=1).
  - Nếu `user_id` None → chỉ `scope="common"` (cache chung cho route general).
  - Legacy point không có `scope` → đọc được (coi common) nhưng write mới luôn có scope.
- `set_cached_response(question, response, sources, user_id: Optional[str] = None, scope: Optional[str] = None)`:
  - `scope` None → tự suy: nếu `user_id` có → `f"user:{user_id}"`; else `"common"`.
  - Caller có thể ép `scope="common"` (route general_chat).
- Giữ `clear_semantic_cache()` nguyên.

### 2. `backend/src/tasks.py`

- Line 1124: `get_cached_response(question)` → `get_cached_response(question, user_id)`.
- Line 1197: `set_cached_response(question, response_text, sources)` → `set_cached_response(question, response_text, sources, user_id)`.
  - Nếu route = `general_chat` → truyền `scope="common"` (opt-in cache chung cho lời chào). Cần biết route ở scope set — kiểm tra `state.get("route")` hoặc biến route trong hàm gọi (xem `bot_rag_answer_message` / chat handler).
  - Route khác (legal_rag, agent_tools, web_search) → mặc định per-user (`scope="user:<user_id>"`).

### 3. Test mới `tests/test_semantic_cache.py` (hoặc extend)

- `test_user_isolation_no_leak`: set_cached_response(q, ans, src, user_id="A"); get_cached_response(q, user_id="B") → None.
- `test_same_user_hits`: set A → get A → hit.
- `test_common_scope_hits_all`: set_cached_response(q, ans, src, user_id=None, scope="common"); get_cached_response(q, user_id="B") → hit.
- `test_ttl_expiry`: legacy/old entry → None.
- Mock Qdrant client (monkeypatch `semantic_cache.get_client`) để không cần Qdrant thật, kiểm filter object đúng `should` conditions.

## Thứ tự làm

1. Viết test RED trước (TDD theo rule testing).
2. Sửa `semantic_cache.py` (scope + filter).
3. Sửa `tasks.py` 2 call site.
4. Chạy test → GREEN.
5. Chạy full suite → không phá (chú ý SSE 2 fail pre-existing DisabledBackend).
6. Code review (security-reviewer agent) theo rule.
7. Commit: `fix(security): scope semantic cache per-user to prevent cross-user data leak`.

## Rủi ro / lưu ý

- **Hit rate giảm**: trước cache toàn user chung → hit cao. Sau fix per-user → hit thấp hơn (mỗi user cache riêng). Trade-off bắt buộc cho privacy. Bù bằng `scope="common"` cho general_chat.
- **Legacy point**: đọc common, an toàn dần. Muốn sạch hẳn → chạy `clear_semantic_cache()` 1 lần sau deploy.
- **Qdrant Filter `should`**: cần `min_should=1` (Filter(must=[], should=[...], min_should=1)). Verify syntax Qdrant client version hiện tại.
- **user_id None**: một số path legacy không có user_id → cache common-only, không ghi per-user rác.

## Verify acceptance (sau fix)

- Chạy lại `run_question_test.py` (16 câu) → route vẫn 100%, không 413 (lưu ý: 413 Groq TPM là bug khác, riêng — xem plan agent prompt bloat). Cache privacy fix không liên quan 413.
- Test isolation pass.

## Tài liệu liên quan

- Bug 413 Groq TPM (khác, ưu tiên thấp hơn): 26 tool schema > 6000 TPM on-demand. Fix: đổi `LLM_MODEL` env hoặc trim docstrings. Để plan riêng.