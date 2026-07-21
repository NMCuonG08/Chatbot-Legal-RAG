# CODEBASE_GUIDE — Legal Chatbot Knowledge Graph

> Sinh bằng method **Understand-Anything** (Egonex-AI): scan → file-analyze → architecture-layers → domain-flows → guided-tour. File tái dùng: lần sau onboard chỉ đọc file này. Output ngôn ngữ **vi**, giữ thuật ngữ kỹ thuật English khi không có chuẩn dịch (middleware, hook, checkpoint, RAG, ReAct).
>
> Repo: `E:\MachineLearning\Legal`. Git branch `main`. Cập nhật gần nhất: commit `562aa71` (graph loop-control).

---

## 1. Tổng quan

Chatbot pháp luật Việt Nam, agent LangGraph multi-specialist (planner → supervisor → specialists), RAG hybrid+rerank, guardrails NeMo, verify-answer groundedness gate, metacognitive escalation, RLHF, approval gate RBAC, trace MySQL+Redis OTel. Đáng tin vì có **harness eval 7 phase** khép kín (pin version → eval song song → score → significance → regression → drift → canary).

- **Entry user**: Streamlit `frontend/chat_interface.py` → `POST /chat/complete` → Celery `llm_handle_message` → LangGraph `run_chat_graph`.
- **Entry eval**: `backend/src/evaluation/run_eval.py` CLI / `scripts/dev.sh eval`.
- **Entry ingest**: `python -m pipeline.run` (modern) hoặc `python -m import_data` (legacy).
- **Entry MCP**: `python -m mcp_server` (stdio | http).

---

## 2. Ngôn ngữ / Framework / Infra

| Khía cạnh | Công nghệ |
|----------|-----------|
| Ngôn ngữ | Python 3.11 (backend), Node ≥18 (promptfoo viewer MCP optional) |
| Agent graph | LangGraph `StateGraph` + `MemorySaver`/`RedisSaver`, `Command(goto=)` handoff |
| Agent loop | LlamaIndex `ReActAgent` (wrapped `FallbackLLM`) |
| LLM provider | Groq (Llama 3.1/3.3) + Ollama — **không OpenAI/Anthropic key** |
| Web | FastAPI (uvicorn) + Celery worker (`-P solo`) |
| UI | Streamlit |
| Vector | Qdrant v1.11.3 (dense + semantic cache `user_episodes`) |
| Graph | Neo4j 5.20 (`(:Statute)-[:HAS_ARTICLE]->(:Article)`) |
| Relational | MariaDB 11 (SQLAlchemy) |
| Cache/queue | Redis 7 (cache + trace pub/sub + Celery broker) |
| Eval | pytest, scipy, numpy, deepeval, ragas, promptfoo (npx), opentelemetry |
| Infra ops | docker-compose (redis/mariadb/qdrant/neo4j/backend/worker/frontend/prometheus/grafana) |
| CI | `.github/workflows/ci.yml` (offline gate) + `eval-live.yml` (nightly 02:07 UTC) |

---

## 3. Architecture Layers

Layer trong knowledge graph (node group). Mỗi layer = 1 trách nhiệm, phụ thuộc layer dưới, không import ngược.

```
┌─────────────────────────────── entrypoint ──────────────────────────────┐
│ frontend/chat_interface.py  app.py (FastAPI)  mcp_server/  pipeline/run   │
├────────────────────────── orchestration-graph ───────────────────────────┤
│ tasks.py (_build_chat_graph, nodes, run_chat_graph, llm_handle_message)   │
│ supervisor.py  planner.py  graph_loop_control.py                          │
├─────────────────────────────── agent-loop ──────────────────────────────┤
│ agent.py (ReAct + FallbackLLM + memory + tool select)                     │
│ agent_tool_wrappers.py  agent_tool_tracking.py                            │
├────────────────────────────── routing-llm ───────────────────────────────┤
│ brain.py (Groq/Ollama provider, route/intent detect, embeddings, judge) │
├─────────────────────────────── rag-retrieval ────────────────────────────┤
│ search.py  rerank.py  query_rewriter.py  vectorize.py  custom_embedding  │
├────────────────────────────── agent-tools ──────────────────────────────┤
│ legal_tools.py  legal_knowledge_tools.py  legal_retrieval_tools.py       │
│ legal_procedure_tools.py  legal_metadata.py  legal_graph_tools.py        │
│ tavily_tool.py  legal_graph_ingest.py                                     │
├────────────────────────── guardrail / verify / safety ──────────────────┤
│ guardrails_manager.py  verify_answer.py  metacognitive.py                │
├────────────────────── security-rbac / approval ─────────────────────────┤
│ auth.py  rbac.py  approval.py  audit.py  security.py  sandbox.py  seed   │
├────────────────────────────── memory / trace ───────────────────────────┤
│ cache.py  semantic_cache.py  rlhf_store.py  summarizer.py                │
│ trace.py  evaluation/otel_bridge.py                                       │
├────────────────────────────── eval-harness (P1–P7) ─────────────────────┤
│ evaluation/* (run_metadata, stats, parallel, regression, judge_panel,    │
│   pairwise_eval, sim_user, scenarios, redteam/, slicing, metrics_ext,    │
│   golden_unified, drift, cost_routing, run_eval, eval_*, metrics_*)       │
├────────────────────────────── data / infra ─────────────────────────────┤
│ models.py  database.py  graph_db.py  splitter.py  utils.py  config.py    │
│ pipeline/ (orchestrator, run, connectors, parsers, chunker, embedder)    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Map (node + edge)

### 4.1 Orchestration graph — `tasks.py`

| Node | fn (file:line) | Edge tiếp theo |
|------|---------------|----------------|
| START | — | → `route` (1105) |
| `route` | `route_node` tasks.py:761 | → `planner` (1107) |
| `planner` | `planner_node` tasks.py:782 | cond `choose_route` 1090 → `{legal_rag: retrieve, agent_tools, web_search, general_chat}` |
| `retrieve` | `retrieve_node` 844 | → `grade_documents` (1120) |
| `grade_documents` | `grade_documents_node` 866 | cond `decide_after_grade` 889 → `{generate, rewrite_query, web_search}` |
| `rewrite_query` | `rewrite_query_node` 892 | → `retrieve` (CRAG reflection loop, cap `REFLECTION_MAX=2`) |
| `generate` | `generate_node` 901 | → `verify_answer` (1135); runtime `Command(goto=web_search)` khi `_should_handoff_to_web` |
| `agent_tools` | `agent_tools_node` 983 | → `metacognitive` (1144); runtime `Command(goto= retrieve\|web_search\|general_chat)` qua `_supervisor_next`; pre-flight approval gate có thể short-circuit |
| `web_search` | `web_search_node` 1022 | → `metacognitive` (1145); runtime handoff ngược qua `_supervisor_next` |
| `general_chat` | `general_chat_node` 1038 | → `metacognitive` (1146) |
| `verify_answer` | `verify_answers_node` 1048 | cond `verify_router` 1066 → `{rewrite_query (recovery, 1140), metacognitive (1141)}` |
| `metacognitive` | `metacognitive_node` 1080 | → END (1147); prepend `ESCALATION_PREFIX` khi escalate |

**Loop control (senior)**:
- `GRAPH_RECURSION_LIMIT=32` (env) — cap recursion trên `graph.invoke`.
- `GRAPH_RUN_TIMEOUT_S=120` (env) — wall-clock deadline qua `_invoke_with_deadline` (`graph_loop_control.py`); `GraphRunTimeout` non-retryable → degrade graceful.
- Supervisor cap `MAX_HANDOFF_STEPS=5` (supervisor.py:32).
- Per-edge guard `agent_to_rag_done`/`generate_to_web_done`/`web_to_agent_done` (tasks.py:297-299) — mỗi handoff fires ≤1 lần/run.

### 4.2 Agent loop — `agent.py`

| Symbol | Vai trò |
|--------|---------|
| `FallbackLLM` (48) | LlamaIndex CustomLLM retry 3x + fallback provider |
| `_build_llm` (138) | Wire Groq/Ollama theo contextvar |
| `get_agent_memory` (293) | Per-(user,conv) `ChatMemoryBuffer` LRU 32 |
| `_summarize_chat_history` (395) | Rolling summary FACTS+TOPICS, `_RESUMMARY_INTERVAL=10`, `_MSG_CHAR_CAP=4000` |
| `filter_tools_for_query` (505) | Keyword-select tools per query |
| `_apply_role_policy` (629) | RBAC filter trước build agent |
| `_build_react_agent` (653) | LlamaIndex `ReActAgent` (max_iterations=10 legacy path) |
| `ai_agent_handle` (689) | `@shared_task` entry ReAct |

### 4.3 Routing/LLM — `brain.py`

`LLMProvider`(101) `GroqProvider`(115) `OllamaProvider`(141) · `get_main_provider`(217) đọc `LLM_PROVIDER_CONTEXTVAR`/`LLM_MODEL_CONTEXTVAR` · `build_judge_fn`(230) · `groq_chat_complete`(309) · `vietnamese_llm_chat_complete`(260) · `get_embedding`(331) · `detect_route`(487) · `detect_user_intent`(366).

### 4.4 Guardrail / verify / safety

| File | Vai trò | Key |
|------|---------|-----|
| `guardrails_manager.py` | NeMo + Tier-1 keyword block + PII homegrown + disclaimer | `verify_input`(170) `verify_output_rag`(218) `detect_pii_vietnamese`(34) `add_legal_disclaimer`(262) |
| `verify_answer.py` | Groundedness judge (Ragas-style) gate trước END | `judge_answer`(28) |
| `metacognitive.py` | Stakes classifier + escalate lawyer | `classify_stakes`(59) `should_escalate`(77) `build_escalation`(98) `HIGH_STAKES_KEYWORDS`(36) |

### 4.5 Security / RBAC / approval

| File | Vai trò | Key |
|------|---------|-----|
| `auth.py` | bcrypt + JWT HS256 (exp 60m) | `hash_password` `create_access_token` `decode_token` `extract_bearer` |
| `rbac.py` | 4 role admin/lawyer/user/guest + tool policy | `ROLE_PERMISSIONS`(89) `filter_tools_by_policy`(116) `needs_approval`(130) `get_current_user`(163) `require_admin`(193) |
| `approval.py` | Human-in-loop `ToolApproval` pending→approved/rejected | `request_approval`(70) `evaluate_tool_gate`(171) `await_approval_response`(206) |
| `audit.py` | Tamper-evident `AuditLog` (best-effort) | `log_audit` `list_audit_entries` |
| `security.py` | Path-traversal guard + `X-API-Key` admin + CORS + collection-name regex | `resolve_safe_data_path` `require_api_key` |
| `sandbox.py` | Subprocess sandbox pure-compute tools — **opt-in wired** via `@sandboxable` (default off) | `SAFE_TO_SANDBOX`(33) `run_in_sandbox`(73) |

### 4.6 Memory / trace

| File | Vai trò | Key |
|------|---------|-----|
| `cache.py` | Redis conversation-id | `get_conversation_id`(48) |
| `semantic_cache.py` | Qdrant semantic cache, per-user scope + TTL + privacy filter | `get_cached_response`(140) `set_cached_response`(205) `_scope_for`(61) |
| `rlhf_store.py` | 👍/👎 MySQL audit + Qdrant good-answers pool, rerank boost | `save_feedback`(68) `find_similar_good`(167) |
| `summarizer.py` | LangChain-Groq summarize | `summarize_text`(18) |
| `trace.py` | `AgentStep` MySQL + Redis pub/sub SSE + OTel bridge | `emit_step`(77) `emit_run_start`(89) `emit_run_end`(100) `_bridge`(65) |
| `evaluation/otel_bridge.py` | OTel span mirror (fire-and-forget, default off) | `setup_otel` `bridge_emit_step` |

### 4.7 Data / infra

`models.py` — SQLAlchemy `Base` + tables: `ChatConversation`, `Document`, `DocumentChunk`, `UserEpisode`, `AgentFeedback`, `GraphRun`, `AgentStep`, `User`, `AuditLog`, `ToolApproval`. CRUD helpers + `ensure_database_schema()`.

`database.py` (Settings, engine, Celery app) · `vectorize.py` (Qdrant CRUD + search) · `custom_embedding.py` (homegrown embed, hash fallback khi Cohere off) · `search.py` (hybrid BM25+vector) · `rerank.py` (Cohere) · `graph_db.py` (Neo4j) · `splitter.py` (LlamaIndex semantic/token) · `query_rewriter.py` (expand + multi-query + context rewrite) · `config.py` (toàn bộ env-tunable constants, xem bảng §8).

### 4.8 Agent tools — `legal_*.py`

`legal_tools.py` (contract penalty, age, inheritance, business name, statute limit) · `legal_knowledge_tools.py` (severance, overtime, PIT, land/vehicle fee, court fee, admin fine, child support, law version, disclaimer) · `legal_retrieval_tools.py` (article/precedent/cross-ref lookup + citation verify qua Qdrant) · `legal_procedure_tools.py` (procedure wizard, jurisdiction, doc template generator) · `legal_metadata.py` (extract law_name/article_number) · `legal_graph_tools.py` (Neo4j multi-hop recall) · `tavily_tool.py` (web search) · `legal_graph_ingest.py` (Neo4j MERGE).

### 4.9 MCP server — `mcp_server/`

`server.py` FastMCP "legal-tools-vn", ~15 tool VN, lazy import degrade-to-JSON-error. `auth.py` `BearerAuthMiddleware` constant-time compare `MCP_API_KEY`, dev bypass `MCP_ALLOW_NO_AUTH=1`. `__main__.py` stdio (no auth) | http (bearer, port 8100).

### 4.10 Eval harness — `backend/src/evaluation/` (P1–P7)

| Phase | Module | Mục đích |
|-------|--------|----------|
| P1 | `run_metadata.py` `stats.py` `parallel.py` `regression.py` | pin git sha+model+prompt hash, bootstrap CI/McNemar/Wilcoxon/pass@k, ThreadPool parallel, regression gate |
| P2 | `judge_panel.py` `pairwise_eval.py` | swap augment + CoT G-Eval + multi-judge panel + Cohen kappa, pairwise A/B |
| P3 | `sim_user.py` `scenarios.py` | LLM-as-user persona, τ-bench r_action+r_output composite ≥0.7 |
| P4 | `redteam/{dataset,metrics,promptfoo_config}.py` + `probes.jsonl` | 6 category probes, PII homegrown, safety metrics |
| P5 | `slicing.py` `metrics_extended.py` `golden_unified.py` | slice by intent/difficulty/language/oos, tool-call accuracy/noise/ctx-util/hallucination/p99, unify 3 golden set |
| P6 | `otel_bridge.py` `drift.py` `cost_routing.py` | OTel mirror, PSI+KL drift, route→model, canary/shadow |
| core | `dataset.py` `metrics_retrieval.py` `metrics_generation.py` `eval_retrieval.py` `eval_generation.py` `eval_e2e.py` `run_eval.py` `failure_analysis.py` `evaluate_model.py` | Hit@K/MRR/nDCG, LLM-as-judge faithfulness/relevance/precision, 5-config ablation, e2e qua chat graph, CLI orchestrator |

**Judge constraint**: Groq+Ollama only (no OpenAI/Anthropic). Self-preference mitigate = swap + CoT + Ollama 2nd judge + calibration kappa.

---

## 5. Business Domains / Flows

### Flow A — Chat (user → answer)
`POST /chat/complete` → Celery `llm_handle_message` (tasks.py:1557):
1. Resolve `conversation_id` (cache.py) → load history (models).
2. Semantic cache hit? → return (`semantic_cache.get_cached_response`).
3. Input guardrail `guardrails_manager.verify_input` (Tier-1 + NeMo).
4. `emit_run_start` (trace) → `run_chat_graph` wrapped `with_retry` + `_invoke_with_deadline`.
5. Graph: route → planner → specialist (retrieve→grade→{generate|rewrite loop|web} / agent_tools / web / chat) → verify_answer (RAG path) → metacognitive → END.
6. `update_chat_conversation` persist · `set_cached_response` · `save_episodic_memory_task.delay` · `emit_run_end`.
7. (opt-in) `shadow=True` → `_run_shadow` chạy candidate song song, persist cả 2.

### Flow B — Data ingestion
Modern: `pipeline.run` → `run_pipeline` (orchestrator) → connector.fetch → persist_raw → parse → persist_parsed → chunk → persist_serving → embed_chunks (Qdrant + MySQL `document_chunks`, idempotent qua chunk_hash) → state embedded. Per-doc fail → mark failed, loop continues.
Legacy: `import_data` → split + embed + Qdrant + MySQL + `extract_legal_metadata` + Neo4j `add_to_graph` + BM25 index.

### Flow C — Eval (closed loop)
`scripts/dev.sh golden` → `golden_unified.write_unified_dataset` (data/golden_unified.jsonl). `scripts/dev.sh eval` → `run_eval --mode all --parallel N` (judge Groq/Ollama) → `eval_reports/live/<provider>/`. `scripts/dev.sh drift B R` → PSI+KL. `scripts/dev.sh regression B` → diff vs baseline. CI nightly `eval-live.yml` boot services → seed → eval → regression → red-team → drift → artifact + issue on fail.

### Flow D — Auth / approval
`seed_admin` startup → admin bcrypt. Login → JWT HS256 60m. `get_current_user` dep → `Principal`. `filter_tools_by_policy` trước build agent. Sensitive tool non-exempt role → `request_approval` (pending) → admin `POST /approvals/{id}/decide` → re-invoke agent với tool allowed. `audit.log_audit` best-effort mọi event.

### Flow E — MCP
`python -m mcp_server --transport http` → `BearerAuthMiddleware(MCP_API_KEY)` → FastMCP tools (calc/curated/Qdrant/Neo4j). stdio = no auth (local). Degrade per-call JSON error khi store thiếu.

---

## 6. Guided Tour (thứ tự đọc để onboard)

Đọc theo thứ tự này để hiểu "mỗi mảnh ghép vừa nhau thế nào":

1. `config.py` — toàn bộ hằng số env-tunable. Đọc đầu để biết ngưỡng (recursion, timeout, retry, judge, P6 flags).
2. `models.py` — schema persistence. Biết bảng nào lưu gì trước khi đọc code ghi chúng.
3. `brain.py` — provider + routing. Nền tảng LLM mọi node gọi.
4. `tasks.py` — `_build_chat_graph` (758-1149) + bảng node flow §4.1. Trái tim repo.
5. `agent.py` — ReAct loop, memory, tool select. Agent chuyên gia.
6. `supervisor.py` + `planner.py` — handoff đa chuyên gia.
7. `guardrails_manager.py` → `verify_answer.py` → `metacognitive.py` — 3 lớp gate cuối.
8. `rbac.py` → `approval.py` → `audit.py` — security layer.
9. `trace.py` → `evaluation/otel_bridge.py` — observability + cầu nối sang harness.
10. `evaluation/run_eval.py` + `EVAL_SYSTEM_SUMMARY.md` — harness 7 phase.
11. `pipeline/orchestrator.py` → `import_data.py` — ingestion 2 style.
12. `mcp_server/server.py` — surface tool thứ 2 cho Claude Desktop/Code.
13. `frontend/chat_interface.py` — UI + SSE.
14. `scripts/dev.sh` — 1 entry ops. `RUN.md` — cheat sheet.

---

## 7. Ops — chạy (pointer)

**1 lệnh / việc** (chi tiết `RUN.md`):
```bash
scripts/dev.sh up        # boot infra
scripts/dev.sh setup     # 1 lần: install + schema + Qdrant
scripts/dev.sh app       # Celery + FastAPI :8002 + Streamlit :8501
scripts/dev.sh test      # offline gate (no service/key)
scripts/dev.sh eval      # eval --mode all, judge Groq
scripts/dev.sh prod      # full boot runbook
```

**P6 opt-in (default off)**: `OTEL_BRIDGE_ENABLED` `COST_ROUTING_ENABLED` `SHADOW_MODE_ENABLED`.

---

## 8. Config constants (`config.py`)

| Hằng | Default | Ý nghĩa |
|------|---------|---------|
| `GRAPH_RECURSION_LIMIT` | 32 | cap recursion graph.invoke |
| `GRAPH_RUN_TIMEOUT_S` | 120 | wall-clock deadline 1 invoke |
| `REFLECTION_MAX` | 2 | CRAG rewrite loop cap |
| `DOC_GRADE_THRESHOLD` | 0.35 | doc graded relevant không cần judge |
| `VERIFY_MAX_RETRIES` | 2 | verify→rewrite recovery cap |
| `VERIFY_ANSWER_THRESHOLD` | 0.7 | verdict "supported" |
| `ESCALATION_CONFIDENCE_THRESHOLD` | 0.6 | MEDIUM stakes escalate ngưỡng |
| `RLHF_RERANK_BOOST` | 0.05 | additive boost chunk 👍-backed |
| `JUDGE_PROVIDER`/`JUDGE_MODEL` | groq / llama-3.1-8b-instant | judge pin (riêng agent model) |
| `GUARDRAILS_PII_OUTPUT_ENABLED` | false | PII redact eval-time only |
| `OTEL_BRIDGE_ENABLED` | false | OTel span mirror |
| `SANDBOX_ENABLED` | false | opt-in subprocess isolation cho 16 pure-compute tool (`@sandboxable`); off = passthrough in-process |
| `COST_ROUTING_ENABLED` | false | route→model (legal_rag big, rest small) |
| `SHADOW_MODE_ENABLED` | false | candidate song song (doubles Groq cost) |

---

## 9. Known gaps / pre-existing issues

- **9 module test collection error**: langchain-groq 1.1.3 / langchain-core `ModelProfile` import mismatch (ngoài eval harness) — ignore trong offline gate (`pytest.ini` comment). Tracked riêng.
- **2 module drift**: `test_react_toolcalls`, `test_per_conversation_memory` — `agent._get_ai_agent` xóa trong refactor.
- **sandbox.py wired opt-in**: `@sandboxable` decorator trên 16 pure-compute wrappers + `config.SANDBOX_ENABLED` (default off). Khi bật, tool chạy trong subprocess scrubbed-env + timeout; khi tắt, passthrough in-process (zero overhead). Test: `test_sandboxable.py`.
- **Stop condition structured JSON**: supervisor/planner giờ yêu cầu JSON terminal (`{"next":...}` / `{"steps":[...]}`) + schema-validate qua `llm_json.extract_json`; legacy `<handoff>`/`<step>` tag là fallback. Test: `test_structured_stop.py`.
- **Trace per-tool-call + latency**: `@track_tool_call` emit `tool_call` event với `latency_ms` qua `agent_run_id`/`agent_thread_id` contextvar. `run_id` per-message (per-turn), `thread_id` per-session — `load_session_trace(thread_id)` replay cả hội thoại thành 1 trace.
- **Self-preference judge**: Llama judge Llama. Mitigate swap+CoT+Ollama+kappa.

---

## 10. File index nhanh

| Muốn tìm | Mở |
|---------|-----|
| Graph nodes/edges | `tasks.py:758-1149` |
| Run entry | `tasks.py:1157 run_chat_graph` · `tasks.py:1557 llm_handle_message` |
| LLM/provider | `brain.py` |
| Agent ReAct | `agent.py:653 _build_react_agent` |
| Handoff logic | `supervisor.py:133` · `planner.py:97` |
| Guardrail | `guardrails_manager.py` |
| Verify gate | `verify_answer.py:28` |
| Escalate | `metacognitive.py:98` |
| RBAC | `rbac.py:89 ROLE_PERMISSIONS` |
| Approval | `approval.py:171 evaluate_tool_gate` |
| Trace | `trace.py:77 emit_step` |
| Eval CLI | `evaluation/run_eval.py` |
| Eval summary | `EVAL_SYSTEM_SUMMARY.md` |
| Runbook | `RUN.md` |
| Ops entry | `scripts/dev.sh` |

---

## 11. Production Upgrade (Phase 1-6)

Bổ sung 6 trụ production-grade. Không train lại model — embedding giữ nguyên (text không đổi), chỉ re-upsert Qdrant payload + rebuild BM25 + re-MERGE Neo4j.

**Metadata chuẩn hóa (Phase 1)** — `legal_metadata.extract_legal_metadata` giờ trả đủ `law_name, article_number, clause_number, point_letter, document_number, document_year, document_type`. Hiệu lực per-document: `legal_effectivity.classify_effectivity` → `in_force|not_yet_effective|repealed|amended`, dùng bảng version `legal_corpus_versions` (thay hardcoded 5 luật). SQL: `DocumentChunk` +8 cột (nullable, indexed), migration idempotent qua `ensure_database_schema`. Payload enrich tại `import_data.py:245,378` + `scripts/reingest_metadata.py` (set_payload, không re-embed).

**BM25 Vietnamese tokenizer (Phase 2)** — `search.py` wrap pyvi `ViTokenizer.tokenize` cho BM25 text + query. Fallback raw-text khi pyvi thiếu. Rebuild BM25 cache sau deploy.

**Reranker hardening + score blend (Phase 3)** — `rerank.py`: Cohere/BGE fail → `_passthrough` log WARNING + flag `rerank_failed=True` (không silent). `search.blend_hybrid_rerank`: `final = α*norm(hybrid) + (1-α)*norm(relevance)`, α=`RRF_BLEND_ALPHA` (default 0.6). RRF weights `RRF_W_VECTOR/RRF_W_BM25` env. Top_k/top_n sync (`RRF_TOP_K=4`, `RRF_TOP_N=5`).

**KG cross-reference edges (Phase 4)** — `legal_graph_relations.extract_relations` (rule+regex) mine CITES/AMENDS/REPEALS/REPLACED_BY từ chunk text. `legal_graph_ingest.add_relations_to_graph` MERGE typed edges (Cypher 1 template per edge type — không parameterize TYPE). `legal_graph_tools.recall_legal_graph_relations` traverse outbound/inbound cross-refs. RAG `retrieve_node` (`tasks.py:890`) query graph cho multi-hop query (`MULTI_HOP_KEYWORDS`), merge hits vào documents trước rerank.

**Multi-agent verify mở rộng (Phase 5)** — `agent_tool_tracking.agent_sources` contextvar + `record_agent_source`: retrieval tools (article_lookup/cross_reference/verify_citation/precedent_lookup) ghi source chunks. `ai_agent_handle` trả `(text, tool_calls, sources)`, `agent_tools_node` set `state["sources"]`. `verify_answer.judge_answer`: agent/web route giờ có sources → judge faithfulness thật; chỉ general_chat short-circuit (log rõ, không silent).

**Cleanup + docs (Phase 6)** — docstring `legal_retrieval_tools.py` cập nhật (payload giờ có metadata cấu trúc). Test mới: `test_legal_effectivity`, `test_graph_relations`; `test_legal_metadata` mở rộng.

| Muốn tìm | Mở |
|---------|-----|
| Metadata parser | `legal_metadata.py` |
| Effectivity | `legal_effectivity.py` · `legal_corpus_versions.py` |
| SQL metadata cols | `models.py:DocumentChunk` · `ensure_database_schema` |
| BM25 tokenizer | `search.py:_tokenize_vi` |
| Rerank hardening | `rerank.py:_passthrough` |
| Score blend | `search.py:blend_hybrid_rerank` |
| KG relations extractor | `legal_graph_relations.py` |
| KG edge MERGE | `legal_graph_ingest.py:add_relations_to_graph` |
| KG cross-ref traversal | `legal_graph_tools.py:recall_legal_graph_relations` |
| Agent sources | `agent_tool_tracking.py:agent_sources` · `agent_tool_wrappers.py:record_agent_source` |
| Re-ingest | `scripts/reingest_metadata.py` |

---

*File này là knowledge-graph dạng markdown (sinh bằng method Understand-Anything). Để có dashboard tương tác + JSON graph, cài plugin: `/plugin marketplace add Egonex-AI/Understand-Anything` → `/understand` (sinh `.ua/knowledge-graph.json`). File này đủ onboard mà không cần plugin.*