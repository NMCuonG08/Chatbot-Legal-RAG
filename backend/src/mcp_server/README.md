# Legal Tools — MCP Server

MCP (Model Context Protocol) server bộc lộ ~17 công cụ pháp luật Việt Nam làm **tools** cho bất kỳ MCP client nào (Claude Desktop, Claude Code, Cursor, MCP inspector).

Bọc các implementation thô (Dict-returning) trong `legal_tools.py`, `legal_knowledge_tools.py`, `legal_retrieval_tools.py`, `legal_procedure_tools.py`, `legal_graph_tools.py`. **Không** bọc lại LlamaIndex `FunctionTool` (tránh double-encode).

## Tools

| Tool | Mô tả | Nguồn |
|---|---|---|
| `contract_penalty_calculator` | Phạt vi phạm hợp đồng (Điều 418 BLDS) | `legal_tools` |
| `legal_age_checker` | Độ tuổi pháp lý | `legal_tools` |
| `inheritance_calculator` | Phân chia thừa kế | `legal_tools` |
| `business_name_validator` | Tên doanh nghiệp | `legal_tools` |
| `statute_lookup` | Thời hiệu khởi kiện | `legal_tools` |
| `severance_pay` | Trợ cấp thôi việc | `legal_knowledge_tools` |
| `overtime_pay` | Tiền làm thêm giờ | `legal_knowledge_tools` |
| `pit_monthly` | Thuế TNCN tháng | `legal_knowledge_tools` |
| `court_fee` | Án phí sơ thẩm | `legal_knowledge_tools` |
| `child_support` | Cấp dưỡng con | `legal_knowledge_tools` |
| `law_version` | Phiên bản/hiệu lực văn bản | `legal_knowledge_tools` |
| `article_lookup` | Tra điều luật (Qdrant) | `legal_retrieval_tools` |
| `precedent_lookup` | Án lệ (Qdrant) | `legal_retrieval_tools` |
| `cross_reference` | Dẫn chiếu (Qdrant) | `legal_retrieval_tools` |
| `verify_citation` | Kiểm trích dẫn (Qdrant) | `legal_retrieval_tools` |
| `procedure_wizard` | Các bước thủ tục | `legal_procedure_tools` |
| `jurisdiction_resolver` | Thẩm quyền giải quyết | `legal_procedure_tools` |
| `recall_legal_graph` | Đồ thị tri thức (Neo4j) | `legal_graph_tools` |

Retrieval/graph tools **best-effort**: nếu Qdrant/Neo4j chưa chạy, tool trả JSON `{"error": ...}` thay vì crash server. Calc tools không cần deps ngoài.

## Cài đặt

```bash
cd backend
pip install -r requirements.txt   # thêm mcp>=1.2.0
```

## Chạy

### stdio (local — Claude Desktop / Code)

```bash
cd backend/src
python -m mcp_server --transport stdio
```

### HTTP / streamable-http (remote/production)

```bash
cd backend/src
python -m mcp_server --transport http --host 0.0.0.0 --port 8100
```

Endpoint: `http://0.0.0.0:8100/mcp`.

### Inspector (dev UI)

```bash
mcp dev src/mcp_server/server.py:mcp
```

## Claude Desktop config

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "legal-tools-vn": {
      "command": "python",
      "args": ["-m", "mcp_server", "--transport", "stdio"],
      "cwd": "/path/to/backend/src",
      "env": {}
    }
  }
}
```

Lưu ý: chạy từ thư mục `backend/src/` (hoặc set `PYTHONPATH` tới `backend/src`) để các import `legal_*` resolve.

## Auth (HTTP transport)

stdio **không** cần auth (local process). HTTP transport bắt buộc **1 bearer key chung** cho mọi client — set 1 env là xong, dễ deploy.

### Sinh key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# ví dụ: p7xQ9...k3A  -> paste vào MCP_API_KEY
```

### Env

```bash
MCP_TRANSPORT=http
MCP_HTTP_PORT=8100
MCP_API_KEY=p7xQ9...k3A          # bắt buộc khi HTTP
MCP_ALLOW_NO_AUTH=0               # dev only = 1 (bỏ qua auth, KHÔNG dùng prod)
```

Khi `MCP_API_KEY` trống và `MCP_ALLOW_NO_AUTH != 1`, server **refuse start** (chống leak tool public). Client gửi header:

```
Authorization: Bearer <MCP_API_KEY>
```

Sai/thiếu key → `401 {"error":"unauthorized","detail":"..."}`. So sánh constant-time (`secrets.compare_digest`).

### Chạy HTTP

```bash
cd backend/src
export MCP_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
python -m mcp_server --transport http --host 0.0.0.0 --port 8100
# endpoint: http://<host>:8100/mcp
```

### Claude Desktop / Cloud Desktop — remote HTTP (streamable-http)

MCP server chạy remote (VPS/docker), client chỉ cần URL + header bearer. Config `claude_desktop_config.json` (hoặc Connectors panel trên claude.ai):

```json
{
  "mcpServers": {
    "legal-tools-vn": {
      "type": "streamable-http",
      "url": "https://your-deploy.example.com/mcp",
      "headers": {
        "Authorization": "Bearer p7xQ9...k3A"
      }
    }
  }
}
```

**Cloud desktop (claude.ai)**: Settings → Connectors → Add custom connector → dán URL `https://your-deploy.example.com/mcp` + header `Authorization: Bearer <key>`. Key chung cho cả workspace.

**Lưu ý prod:**
- Đặt server sau HTTPS reverse proxy (nginx/caddy) — bearer header chỉ an toàn qua TLS.
- Rotate key bằng cách đổi `MCP_API_KEY` + restart; toàn bộ client cũ bị kick ngay.
- Không log key. `BearerAuthMiddleware` không bao giờ echo key trong response/error.
- 1 key = all-or-nothing. Cần per-user (rate limit, audit theo user) → dùng `auth.py` JWT layer thay vì key này.

## Test

```bash
cd backend
pytest tests/test_mcp_server.py -v
```

Dùng MCP client in-process gọi vài tool calc, assert output JSON đúng. Retrieval/graph tool skip khi Qdrant/Neo4j không có.