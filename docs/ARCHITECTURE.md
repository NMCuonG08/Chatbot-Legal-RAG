# Architecture — Vietnamese Legal Assistant

This document is the canonical architecture reference: request lifecycle, the multi-source ingestion pipeline, the RAG + agentic engine, the storage tiers and lineage model, the security model, and the observability/evaluation story. For a quick-start and feature overview, see the root [`README.md`](../README.md).

---

## 1. High-level data flow

```
                         ┌──────────────────────────────────────────────┐
                         │                 Ingestion                    │
                         │   JSONL / MD / HTML / PDF → connectors →     │
                         │   raw → parse → chunk → embed → Qdrant+MySQL │
                         │   (idempotent via pipeline_documents state)   │
                         └──────────────────────────────────────────────┘
                                            │ writes shared collection
                                            ▼
┌──────────┐   query   ┌──────────┐   route   ┌──────────────────────────────────────┐
│ Streamlit│─────────▶│ FastAPI  │─────────▶│  Celery Worker (LangGraph workflow)  │
│  UI      │◀─────────│  app.py  │ task id  │  guardrails → router → engine        │
└──────────┘  result  └────┬─────┘          └──────────────┬───────────────────────┘
                           │                               │
                           │  Redis broker                 ├─ legal_rag  → hybrid search + rerank + LLM
                           │  SQL history                  ├─ agent_tools→ ReAct agent + legal/web tools
                           │                               ├─ web_search → Tavily/Google
                           │                               └─ general_chat → LLM
                           ▼
                     ┌───────────┐
                     │  Output   │ groundedness check → disclaimer → history → response
                     │ guardrails│
                     └───────────┘
```

Two planes share **one** Qdrant collection (default `llm`):
*   **Ingestion plane** writes chunks + embeddings.
*   **Query plane** reads them for RAG. There is no second serving store — the pipeline writes into exactly the collection the RAG engine reads from.

---

## 2. Multi-source ingestion pipeline (`backend/src/pipeline/`)

### 2.1 Design principle: one loop, many connectors
Adding a data source does **not** clone the fetch→parse→chunk→embed logic. A source is implemented as a `BaseConnector` subclass (set `source_id` + `source_type`, implement `fetch()`) and appended to the connector list passed to `run_pipeline`. The orchestrator never branches on source identity — only the parser branches on `source_type`.

### 2.2 Stages and normalized schema

Every stage past ingestion operates exclusively on frozen, immutable dataclasses (`pipeline/schema.py`). Each carries lineage so any chunk traces back to its raw origin.

| Tier | Dataclass | Key lineage fields | Producer |
|------|----------|--------------------|----------|
| 1 raw | `RawDocument` | `doc_id` (content hash), `source_id`, `source_type`, `origin_url`, `fetched_at` | connector |
| 2 parsed | `ParsedDocument` | `raw_doc_id`, `parser_used`, `parser_version` | `parsers.parse` |
| 3 serving | `ChunkedDocument` | `parsed_doc_id`, `chunk_config`, `embed_model` | `chunker.chunk` (+ embedder fills `embed_model`) |

`doc_id` is a content hash, so the same document fetched twice yields the same id — the basis for idempotent ingestion.

### 2.3 The orchestrator loop (`orchestrator.run_pipeline`)

```
for connector in connectors:
    raw_docs = connector.fetch()           # broken source → log + continue (no halt)
    for doc in raw_docs:
        if state.is_done(doc.doc_id, EMBEDDED):     # idempotency
            skipped++; continue
        try:
            storage.persist_raw(doc)        # write-once (no-op if exists)
            state.mark_status(FETCHED)
            parsed = parse(doc)             # branch on source_type, not source
            storage.persist_parsed(parsed)
            state.mark_status(PARSED)
            chunked = chunk(parsed, use_semantic)
            storage.persist_serving(chunked)
            state.mark_status(CHUNKED)
            embedded = embed_chunks(chunked, collection)  # dedup + orphan GC
            storage.persist_serving(embedded)             # re-persist with embed_model
            state.mark_status(EMBEDDED)
        except Exception:
            state.mark_status(FAILED, error=...)          # per-doc isolation
```

Guarantees:
*   **Idempotency:** `state.is_done` skips already-embedded docs; re-runs are cheap.
*   **Per-doc isolation:** a failing doc is marked `failed` and the loop continues — a bad PDF never kills the JSONL batch.
*   **Connector isolation:** a connector whose `fetch()` raises is logged and skipped; the next connector still runs.

### 2.4 Connectors

| Connector | `source_type` | Behavior |
|-----------|--------------|----------|
| `JsonlQaConnector` | `json` | Reads `{question, context}` (legacy `answer` also accepted) JSONL, one `RawDocument` per line. Skips blank/malformed lines and lines missing Q or context. `doc_id = source_id + sha256(content)[:16]`. |
| `MarkdownConnector` | `markdown` | Recursive `*.md` under a dir; markdown is already plain text so parsing is passthrough. |
| `HtmlConnector` | `html` | Local `*.html` under a dir + optional URL list (remote fetch via `requests`, best-effort — a dead URL never kills the batch). Raw HTML stored; tag-stripping happens in the parser. |
| `PdfConnector` | `pdf` | `*.pdf` under a dir; stores raw bytes **base64-encoded** so the raw tier is the original document and the pipeline can re-parse without re-reading the file. Text extraction (pypdf / pdfplumber) happens in the parser. |

### 2.5 Parsers (`parsers.parse`)
Dispatch on `source_type` only (never on source): `json` → reconstruct `question + context`; `markdown` → passthrough; `html` → tag-strip (bs4 if available, regex fallback); `pdf` → base64-decode then pypdf / pdfplumber (clear `RuntimeError` if no PDF backend — raw is still preserved). Empty parsed text raises (a hard error, not a silent empty doc).

### 2.6 Three-tier storage lake (`storage.py`)
`data/pipeline_lake/{raw,processed,serving}` (override via `PIPELINE_LAKE_DIR`):
*   `raw/<source_id>/<doc_id>.txt` — write-once; re-persist is a no-op (immutable source of truth).
*   `processed/<doc_id>.json` — `ParsedDocument` JSON with `raw_doc_id` lineage.
*   `serving/<doc_id>.json` — `ChunkedDocument` JSON with `parsed_doc_id` lineage.
*   `_safe_name` strips path-nasty characters from `doc_id` so `../../etc/passwd` cannot escape the raw dir.

### 2.7 State store (`state.py`)
A single `pipeline_documents` table shared by every source:

| Column | Purpose |
|--------|---------|
| `doc_id` (PK) | content hash |
| `source_id` | which connector produced it |
| `status` | `fetched` / `parsed` / `chunked` / `embedded` / `failed` |
| `error` | failure detail (only on `failed`) |
| `updated_at` | last status change |

`is_done(doc_id, target=EMBEDDED)` is the idempotency guard. `mark_status` is an upsert. `list_failed` returns recent failures for retry/debug.

### 2.8 Embedder (`embedder.embed_chunks`)
Embeds each chunk and upserts into the **shared** Qdrant collection + `document_chunks` MySQL metadata. Incremental via chunk hashes:
*   chunk id is deterministic: `uuid5(NAMESPACE_DNS, f"doc_{doc_id}_chunk_{idx}")`.
*   If `(cid, chunk_hash)` matches the existing row → skip (unchanged chunk).
*   Old chunk ids no longer present → orphan GC: delete from Qdrant + MySQL.
*   Returns a new immutable `ChunkedDocument` with `embed_model` filled (via `dataclasses.replace`).

### 2.9 Entry points
*   **CLI:** `python -m pipeline.run --source-type {jsonl|markdown|html|pdf} --path ... [--collection llm] [--limit N] [--no-semantic]`
*   **REST:** `POST /pipeline/ingest` with `PipelineIngestRequest{source_type, path, collection_name, limit, use_semantic}`. Path is resolved safely under `IMPORT_DATA_DIR`; collection name is pattern-validated and passed through `get_legal_collection_name`.

---

## 3. Agentic & RAG engine

### 3.1 LangGraph workflow (`tasks.py`)
Celery task orchestrates: input guardrails → intent router → **planner** → selected specialist → **supervisor handoff** → output guardrails → history → response.

The graph wraps the existing CRAG RAG loop without removing it:

```
route → planner → {rag | tool | web | chat} → supervisor → {next specialist | metacognitive → END}
```

*   **Planner node** (`planner.py`): an LLM emits an ordered `<plan>` of steps, each assigning a specialist (`rag` | `tool` | `web` | `chat`) + a Vietnamese goal. Simple queries yield a 1-step plan; multi-step queries (e.g. "tính trợ cấp thôi việc rồi dẫn điều luật áp dụng") yield 2+. `parse_plan` tolerates missing wrappers, alias names, single/double quotes; `validate_plan` caps at `MAX_PLAN_STEPS=5` and dedupes consecutive identical steps. On any LLM failure, `fallback_plan(route)` derives a 1-step plan from the router classification so behavior never regresses.
*   **Supervisor node** (`supervisor.py`): after each specialist produces an answer, an LLM emits a `<handoff next="...|END" rationale="..."/>` decision. `supervisor_decide` prefers the LLM; on any failure/unparseable output it falls back to `heuristic_handoff` (the same Vietnamese keyword markers the graph used before — `không tìm thấy` → web, `cần tra cứu` → rag/tool). A `MAX_HANDOFF_STEPS=5` loop guard forces END so a runaway handoff chain cannot loop forever.

### 3.2 Intent router (`brain.detect_route`)
LLM classifies into one of `legal_rag`, `agent_tools`, `web_search`, `general_chat`. Robust fallback: if the returned route is invalid, extract a valid substring; otherwise apply a keyword heuristic (calc/validation keywords → `agent_tools`, else default `legal_rag`). Priority when ambiguous: `agent_tools` (calc) > `legal_rag` (lookup) > `web_search` (recent) > `general_chat`.

### 3.3 Query rewrite (`brain.detect_user_intent`)
Follow-up questions are rewritten into standalone questions using conversation history — replacing pronouns with concrete nouns from context. Skipped when there is no history or the message is clearly not a follow-up (avoids needless LLM calls).

### 3.4 Advanced RAG flow
Multi-query generation → hybrid search (Qdrant dense + BM25 sparse via LlamaIndex `QueryFusionRetriever`) → rerank (local BGE reranker, Cohere cloud fallback) → generator LLM. Two-tier incremental re-indexing (MD5) and orphan garbage collection keep the collection lean.

### 3.5 ReAct agent (`agent.py`)
LlamaIndex `ReActAgent` with 8 tools, initialized **lazily** on first call (`_get_ai_agent`) so importing the module (e.g. for eval) never needs LLM env vars or network. Compat shim handles both legacy `from_tools` and newer workflow-based constructors, and both sync/async execution paths.

Legal tools (each input-range-validated, returns JSON):
*   `contract_penalty_calculator` — penalty under commercial law, 12% ceiling cap.
*   `legal_age_checker` — age eligibility (marriage gender-aware: male 20 / female 18).
*   `inheritance_calculator` — first-line heir split under Civil Code.
*   `business_name_validator` — naming rules under Enterprise Law.
*   `statute_lookup` — statute of limitations (civil/labor/administrative/criminal).

Web tools: `web_search_tool`, `tavily_search_tool`, `quick_answer_tool`.

### 3.6 Multi-provider LLM routing (`brain.py`)
`LLM_PROVIDER` selects `groq` | `ollama` | `openai`. Vietnamese LLM API and Ollama-main modes fall back to Groq on error. The Vietnamese LLM endpoint has **no hardcoded public IP default** — it must be set explicitly (data-leak prevention). Internal errors return a generic user-facing string; details are logged server-side only.

---

## 4. Security model

The platform layers four controls on top of the legacy admin API-key guard.

### 4.1 Legacy admin surface (`security.py`)
*   **Admin API-key dependency (`require_api_key`):** admin/ingestion endpoints require `X-API-Key` matching `ADMIN_API_KEY`. When `ADMIN_API_KEY` is unset, requests are **refused** unless `ALLOW_UNSAFE_ADMIN=1` (dev convenience). This prevents an accidentally-deployed open admin surface.
*   **Path-traversal guard (`resolve_safe_data_path`):** ingestion paths are resolved under `IMPORT_DATA_DIR`; traversal segments cannot escape.
*   **Collection-name validation:** FastAPI field patterns (`^[a-z0-9_-]{1,64}$`) constrain identifiers; `get_legal_collection_name` normalizes.
*   **CORS (`get_cors_origins`):** configurable allow-list.
*   **No secret leakage:** no hardcoded API keys/IPs; env-var driven; generic error messages to users.

### 4.2 Authentication (`auth.py`, `models.User`)
*   **Password hashing:** passlib bcrypt (`hash_password` / `verify_password`).
*   **JWT:** `python-jose` HS256; `create_access_token(subject, claims)` / `decode_token`. Secret from `JWT_SECRET` (validated at startup), `JWT_ALG`, `JWT_EXP_MIN` (default 60). `extract_bearer` requires the `Bearer ` scheme.
*   **Unsafe fallback:** when `JWT_SECRET` is unset and `ALLOW_UNSAFE_AUTH=1`, a single dev secret is cached at module load so encode/decode stay consistent. Without either, auth-dependent endpoints refuse.
*   **Endpoints:** `POST /auth/register` (rejects admin/lawyer self-registration), `POST /auth/login` (same message on both branches to avoid user enumeration), `GET /auth/me`. Startup `seed_admin()` creates a default admin from `SEED_ADMIN_USERNAME` / `SEED_ADMIN_PASSWORD` (best-effort).

### 4.3 RBAC + tool policy (`rbac.py`)
*   **Roles:** `admin` | `lawyer` | `user` | `guest` (`Role`), each mapping to an allowed tool-name set (`ROLE_PERMISSIONS`); `admin` = `{"*"}`. `Principal` is a frozen dataclass (`is_admin`, `is_approval_exempt`).
*   **Tool filtering:** `filter_tools_by_policy(tools, principal)` drops tools the role may not call before the ReAct agent ever sees them. `is_tool_allowed(name, role)` / `needs_approval(name, principal)` answer per-tool queries.
*   **Sensitive tools** (require approval): `generate_document_template_tool`, `web_search_tool`, `tavily_search_tool`. Approval-exempt roles: `admin`, `lawyer`.
*   **FastAPI deps:** `get_current_user` (strict, 401 on missing/bad token), `get_current_user_optional` (anonymous allowed), `require_role(*roles)` / `require_admin` (403 on wrong role).

### 4.4 Approval workflow (`approval.py`, `models.ToolApproval`)
*   **Gate:** before the ReAct agent loop, `evaluate_tool_gate(principal, anticipated_tool_names, run_id)` intersects the query's anticipated tools with `SENSITIVE_TOOLS`. If a sensitive tool is anticipated and the principal is not exempt, a `ToolApproval(status=pending)` row is created and the run returns a Vietnamese "chờ phê duyệt" message + `approval_id` instead of executing.
*   **Decide:** `POST /approvals/{id}/decide` (admin only, idempotent) sets `approved` | `rejected`, writes an audit entry, and on approval records `approval:allowed:{run_id}` (Redis, with in-process fallback) so a retry of the same run skips the gate.
*   **Scope/limitation:** the graph is **not** fully suspended LangGraph-checkpoint-style; the client re-invokes on approval. This keeps the change non-invasive while still enforcing human-in-the-loop on sensitive tool use.

### 4.5 Sandbox (`sandbox.py`)
*   **Defense-in-depth:** deterministic pure-compute calc tools run in a throwaway child process with a scrubbed environment (env allowlist; `GROQ_API_KEY` etc. stripped) + UTF-8 stdio + hard timeout, so a runaway calc tool cannot wedge the agent/worker. `SAFE_TO_SANDBOX` is an explicit allowlist; network / retrieval / graph / memory / sensitive tools are **not** sandboxed (they need external resources or are gated by approval).
*   **Limitation:** Windows has no seccomp — isolation is process-level + timeout + env scrub, not a syscall filter. Adequate for demo; not production hardening.

### 4.6 Audit (`audit.py`, `models.AuditLog`)
*   `log_audit(user_id, action, resource, ip, payload)` is best-effort: any DB failure is caught + logged server-side so an audit-store outage never breaks a request. `GET /audit` (admin) lists entries with optional user/action filters.

---

## 4b. MCP server (`backend/src/mcp_server/`)

The legal tools are also exposed as an **MCP** (Model Context Protocol) server, so any MCP-compatible client (Claude Desktop, Claude Code, the MCP inspector) can call them directly without going through the HTTP chat API.

*   **Server (`server.py`):** FastMCP (`mcp` SDK) registers ~18 `@mcp.tool()` thin wrappers that call the **raw Dict-returning implementations** in `legal_tools.py` / `legal_knowledge_tools.py` / `legal_retrieval_tools.py` / `legal_procedure_tools.py` / `legal_graph_tools.py` and `json.dumps` the result. They intentionally do **not** wrap the LlamaIndex `FunctionTool` wrappers (avoids double-encoding). Each tool's Vietnamese docstring becomes its MCP description.
*   **Transports:** `stdio` (local, for Claude Desktop / Claude Code) and `streamable-http` (remote/production). CLI: `python -m src.mcp_server --transport stdio|http [--host 0.0.0.0] [--port 8100]` (`__main__.py`), also drivable via `MCP_TRANSPORT` / `MCP_HOST` / `MCP_HTTP_PORT` env.
*   **Inspect:** `mcp dev src.mcp_server.server:mcp` opens the MCP inspector with the full tool list. See `backend/src/mcp_server/README.md` for Claude Desktop config snippet.

---

## 5. Observability & evaluation

### 5.1 In-process metrics (not distributed tracing)
There is **no** LangSmith / Langfuse / OpenTelemetry wiring. Two `contextvars` provide in-process instrumentation feeding the eval suite:

| Contextvar | Producer | Consumed by |
|-----------|----------|------------|
| `agent_tool_calls` | `track_tool_call` decorator around every agent tool | agentic metric: tool-call success rate |
| `usage_accumulator` | `record_usage` in every LLM provider branch | operational metric: token count + API cost |

These are local per-request accumulators — nothing is exported to an external tracer, so there is nothing to "turn off". If distributed tracing is later desired, add a LangSmith/Langfuse callback; it is off by default.

### 5.2 Evaluation suite (`backend/src/evaluation/`)
Four pillars:
*   **Operational:** token count, API cost, latency (TTFT/TTLT).
*   **Quality:** LLM-as-judge faithfulness + answer relevance.
*   **Agentic:** tool-call success rate (from `agent_tool_calls`), router accuracy.
*   **Failure modes:** automatic classification — Retrieval / Routing / Hallucination / Execution.

---

## 6. Storage layers (summary)

| Store | Tech | Holds |
|-------|------|-------|
| Vector DB | Qdrant | chunk vectors + payload (`question`, `content`, `source`, `doc_id`); semantic cache; episodic memory |
| Relational DB | PostgreSQL / MySQL | conversation history, `document_chunks` metadata (chunk hashes), `pipeline_documents` status |
| Broker / lock | Redis | Celery broker, distributed indexing locks |
| File lake | `data/pipeline_lake/` | raw / processed / serving tiers (lineage JSON) |

---

## 7. Failure handling & reliability

*   **Connector fetch failure:** logged, source skipped, run continues.
*   **Per-doc parse/chunk/embed failure:** doc marked `failed` with error, loop continues.
*   **Idempotent re-runs:** already-embedded docs skipped; raw tier write-once.
*   **Embedding service fallback:** legacy importer falls back to Cohere cloud if the local embedding service on :5000 is down.
*   **LLM fallback:** Vietnamese LLM / Ollama-main → Groq on error.
*   **Atomic SQL transactions** for metadata edits with automatic rollback; Redis distributed locks prevent concurrent-indexing races.

---

## 8. Where things live

| Concern | File(s) |
|---------|---------|
| FastAPI endpoints + request models | `backend/src/app.py` |
| Celery + LangGraph workflow | `backend/src/tasks.py` |
| LLM routing + intent/router | `backend/src/brain.py` |
| ReAct agent + tools | `backend/src/agent.py` |
| Legal calculation logic | `backend/src/legal_tools.py` |
| Security (admin API-key, path guard, CORS) | `backend/src/security.py` |
| Authentication (JWT, bcrypt) | `backend/src/auth.py` |
| RBAC + tool policy | `backend/src/rbac.py` |
| Approval workflow | `backend/src/approval.py` |
| Sandbox (subprocess isolation) | `backend/src/sandbox.py` |
| Audit logging | `backend/src/audit.py` |
| Seed admin | `backend/src/seed_admin.py` |
| Planner node | `backend/src/planner.py` |
| Supervisor node | `backend/src/supervisor.py` |
| MCP server | `backend/src/mcp_server/` (`server.py`, `__main__.py`) |
| Guardrails | `backend/src/guardrails_manager.py` |
| Hybrid search + BM25 | `backend/src/search.py` |
| Semantic cache | `backend/src/semantic_cache.py` |
| Storage layer | `backend/src/{vectorize,models,database,cache}.py` |
| Legacy importer | `backend/src/import_data.py` |
| Pipeline | `backend/src/pipeline/` (orchestrator, run, schema, state, storage, parsers, chunker, embedder, connectors/) |
| Evaluation | `backend/src/evaluation/` |
| Tests | `tests/` (conftest auto-adds `backend/src` to `sys.path`; sqlite in-memory + JWT fixtures for Phase 2) |