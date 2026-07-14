# ⚖️ Vietnamese Legal Assistant (RAG & Agentic Chatbot)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-orange.svg)](https://www.llamaindex.ai/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-brightgreen.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)

An intelligent virtual assistant for looking up Vietnamese legal documents, calculating legal costs/penalties, and verifying civil legal conditions. Built on an **Advanced RAG** architecture combined with a **truly agentic workflow** (LangGraph router → ReAct agent), protected by multi-layered safety guardrails and a hardened admin surface.

The agent is **truly agentic**, not just a routed RAG: it **self-verifies** its final answer for citation groundedness (PEV gate), **knows its own limits** via metacognitive escalation to a lawyer-referral prefix on high-stakes / low-confidence cases, **recalls structured legal knowledge** through a Neo4j statute→article graph for multi-hop reasoning, and **learns from feedback** via a 👍/👎 RLHF store that injects good prior answers as few-shot examples and up-weights their source chunks at rerank time.

[🌟 Star](https://github.com/NMCuonG08/Chatbot-Legal-RAG/stargazers) • [🍴 Fork](https://github.com/NMCuonG08/Chatbot-Legal-RAG/fork) • [📚 Docs](docs/ARCHITECTURE.md) • [💬 Discord](https://discord.gg/legal-chatbot)

</div>

---

## 📋 Table of Contents

- [I. Overview](#i-overview)
- [II. System Architecture](#ii-system-architecture)
- [III. Project Structure](#iii-project-structure)
- [IV. Key Features & Capabilities](#iv-key-features--capabilities)
- [V. Technology Stack](#v-technology-stack)
- [VI. Quick Start Guide](#vi-quick-start-guide)
- [VII. Ingestion & Data Pipelines](#vii-ingestion--data-pipelines)
- [VIII. System Verification & Testing](#viii-system-verification--testing)
- [IX. API Documentation](#ix-api-documentation)
- [X. Disclaimer & Terms](#x-disclaimer--terms)

---

## I. Overview

Retrieval-augmented generation (RAG) systems combine generative AI with information retrieval to provide contextualized legal consultation services. This project deploys a comprehensive Vietnamese legal chatbot system using modern microservices architecture and advanced AI technologies. It features a self-corrective RAG workflow, Celery background workers, and long-term memory.

---

## II. System Architecture

Below is the system architecture detailing the ingestion pipeline and the request lifecycle (the Core Agentic & RAG Engine).

### 1. Multi-Source Ingestion Pipeline
```mermaid
%%{init: { 'themeVariables': { 'fontFamily': 'system-ui, -apple-system, sans-serif', 'fontSize': '12px' } } }%%
graph TD
    Dir["JSONL / MD / HTML / PDF"] --> Conn[Connectors]
    Conn -->|fetch| Raw[("`Raw tier
    (immutable)`")]
    Raw --> Parse["`Parsers
    (by source_type)`"]
    Parse --> Proc[("Processed tier")]
    Proc --> Chunk["`Chunker
    (semantic/token)`"]
    Chunk --> Serv[("Serving tier")]
    Chunk --> Emb[Embedder]
    Emb --> Q1[("Qdrant collection")]
    Emb --> Q2[("MySQL chunk metadata")]
    State[("`pipeline_documents
    status table`")] -.->|idempotency| Conn
```

### 2. Request Lifecycle (Core Agentic & RAG Engine)
```mermaid
%%{init: { 'themeVariables': { 'fontFamily': 'system-ui, -apple-system, sans-serif', 'fontSize': '12px' } } }%%
graph TD
    User([User]) -->|Submits query| UI[Streamlit Frontend]
    UI -->|POST /chat/complete| API[FastAPI Backend]
    API -->|Celery task| Broker[("Redis broker")]
    Worker[Celery Worker] <-->|Fetch & process| Broker

    Worker -->|Save checkpoints| Ckpt[("`RedisSaver
    checkpoint`")]
    Worker -->|Emit steps & runs| Trace[("`MySQL graph_runs
    + agent_steps`")]
    Trace -->|publish| Pub[("`Redis pub/sub
    graph_trace_events`")]
    UI -.->|SSE stream| Pub

    Worker -->|1. Input guardrails| Guard[NeMo Guardrails]
    Guard -->|2. Route| Router{LangGraph Router}

    Router -->|legal_rag| CRAG
    Router -->|agent_tools| Agent["`ReAct Agent
    (per-conv memory)`"]
    Router -->|web_search| Web[Web Search]
    Router -->|general_chat| Gen[General Chat]

    subgraph CRAG [Self-Corrective RAG loop]
        Retr["`Retrieve
        (multi-query + hybrid)`"] --> Grade["`Grade docs
        (LLM-as-judge + rerank)`"]
        Grade -->|relevant subset| GenRag["`Generate
        (groundedness guard)`"]
        Grade -->|irrelevant under cap| Rew[Rewrite query]
        Rew --> Retr
        Grade -->|cap reached| WebF[Web fallback]
    end

    Agent -->|Handoff: legal| Retr
    GenRag -->|Handoff: not found| Web
    Web -->|Handoff: tool| Agent

    GenRag -->|verify groundedness| Verify{"`verify_answer
    (PEV gate)`"}
    Verify -->|supported| Meta
    Verify -->|unsupported, retry < cap| Rew
    Verify -->|cap reached / partial| Meta

    Agent & Web & Gen -->|no sources, skip verify| Meta["`metacognitive
    (stakes × confidence)`"]
    Meta -->|high stakes OR low confidence| Esc["`⚠️ lawyer-referral
    prefix`"]
    Meta -->|low stakes & high confidence| Out
    Esc --> Out

    GenRag & Agent & Web & Gen -->|Output guardrails| Out[Final response]
    Out -->|save + trace run_end| Broker
    Out -->|conversation history| SQL[("MySQL")]
    Out -->|👍/👎 feedback| RLHF["`RLHF store
    (Qdrant good pool + MySQL audit)`"]
    RLHF -.->|few-shot + rerank boost| GenRag
```

**Key Graph Properties:**
*   **Self-corrective RAG (CRAG):** `retrieve → grade_documents → {generate | rewrite_query → retrieve (loop) | web_search}`, guarded by `REFLECTION_MAX=2`. Documents are graded by rerank `relevance_score` (threshold `DOC_GRADE_THRESHOLD=0.35`) with an LLM-as-judge batch fallback for borderline docs.
*   **Multi-agent handoff:** `Command(goto=...)` edges — `agent_tools → retrieve` (agent needs legal docs), `generate → web_search` (canned "not found"), `web_search → agent_tools` (needs tool use). Three once-per-run guard flags prevent cycles.
*   **Checkpointing:** `RedisSaver` (requires Redis Stack / RedisJSON) with `MemorySaver` auto-fallback; isolation by `thread_id = conversation_id`, so multi-turn follow-ups resume state.
*   **Trace (self-hosted):** every node emits `node_end`/`handoff` events → MySQL `agent_steps` + Redis pub/sub `graph_trace_events`. One `GraphRun` row per turn (`run_id`, route, final response, reflection_count, tool_calls). No LangSmith/Langfuse, no cloud egress — Vietnamese legal data stays local.
*   **SSE streaming:** `GET /chat/stream/{task_id}` subscribes to the pub/sub channel, filters by `run_id`, and closes on `run_end`. The Streamlit UI renders a live `Agent trace` expander.
*   **ReAct tool-call surfacing:** the `agent_tool_calls` contextvar is reset per turn, populated by `@track_tool_call`, and lifted through the graph → Celery result → async poll as an optional `tool_calls` array.
*   **Per-conversation memory:** ReAct agent memory is keyed by `(user_id, conversation_id)` in an LRU cache (cap 32) — fixing a prior global-memory cross-user leak.
*   **PEV verify_answer gate (self-verify):** after `generate` (RAG route, where `sources` exist), the final answer is judged for citation groundedness via a Ragas-style `evaluate_faithfulness` claim decomposition. Verdict `supported` → metacognitive; `unsupported`/`partial` → loop back to `rewrite_query → retrieve` (capped at `VERIFY_MAX_RETRIES=2`) then degrade to metacognitive. Score thresholds: `VERIFY_ANSWER_THRESHOLD=0.7` (supported), `VERIFY_PARTIAL_THRESHOLD=0.35`. Non-RAG routes (empty `sources`) skip the gate. The graph result exposes `verify_score`, `verify_verdict`, `retry_verify` for eval wiring.
*   **Metacognitive escalation (self-knowledge-of-limits):** a tiered stakes classifier (`high` criminal / `medium` civil disputes / `low` default) combined with the verify_answer confidence produces a graph-level gate — not a post-hoc disclaimer. High-stakes always escalate; medium-stakes escalate only when `verify_score < ESCALATION_CONFIDENCE_THRESHOLD=0.6`. Escalation prepends a "consult a practicing lawyer" prefix to the response (append-only, auditable in trace). The duplicate criminal-keyword list in `guardrails_manager` was consolidated into `metacognitive.HIGH_STAKES_KEYWORDS` (single source of truth).
*   **Neo4j legal knowledge graph (structured memory):** `(:Statute {name, year})-[:HAS_ARTICLE]->(:Article {number, title})` nodes are written idempotently (MERGE) during ingestion, reusing the regex `extract_legal_metadata` extractor. The `recall_legal_graph_tool` FunctionTool traverses the graph for multi-hop queries (keyword-gated in `filter_tools_for_query` on terms like "dẫn chiếu", "án lệ", "còn hiệu lực", "điều nào"). Graph writes are best-effort additive — Neo4j down swallows + logs, never blocks vector ingest.
*   **RLHF continual learning (learns-from-feedback):** 👍/👎 feedback → MySQL `agent_feedback` audit (always, both ratings) + Qdrant `rlhf_good_answers` pool (good-only). User-scoped via `_scope_for` + sentinel rejection (`anonymous`/`demo-session`/`""` → HTTP 400) — no cross-user leak. On later similar questions, the top good answer is injected as a system few-shot example in `generate_node` (score ≥ 0.85, same scope), and chunks whose `doc_id` backed a 👍 answer get an additive `RLHF_RERANK_BOOST=0.05` at rerank. Deterministic uuid5 point id makes re-submitting idempotent; dedup score ≥ 0.92 skips flooding.

---

## III. Project Structure

```
Chatbot-Legal-RAG/
│
├── 🖥️ backend/                    # Backend API service (FastAPI)
│   ├── src/
│   │   ├── app.py                # FastAPI endpoints, lifespan, SSE /chat/stream, request models
│   │   ├── agent.py              # ReAct agent + legal/web tools, per-conversation memory LRU, tool_calls contextvar
│   │   ├── brain.py              # LLM routing (Groq/Ollama/OpenAI), intent detection, routing
│   │   ├── custom_embedding.py   # Custom Vietnamese embedding model integrations
│   │   ├── legal_tools.py        # Vietnamese civil/commercial law calculation logic
│   │   ├── models.py             # Pydantic data models & GraphRun/AgentStep/AgentFeedback DB models
│   │   ├── search.py             # Hybrid search index + BM25 retriever
│   │   ├── semantic_cache.py     # Qdrant-based vector caching (user-scoped, sentinel-aware)
│   │   ├── config.py             # CRAG + PEV + metacognitive + RLHF tuning constants
│   │   ├── database.py           # Database connection & session orchestration
│   │   ├── cache.py              # Redis cache integration
│   │   ├── tasks.py              # Celery tasks + LangGraph StateGraph (CRAG loop, PEV verify, metacognitive, RLHF, handoff, trace, checkpoint)
│   │   ├── trace.py              # Self-hosted trace: MySQL agent_steps/graph_runs + Redis pub/sub emit_*
│   │   ├── verify_answer.py      # Phase 1 — PEV gate: judge final answer groundedness (claim decomp + verdict)
│   │   ├── metacognitive.py      # Phase 2 — tiered stakes classifier + lawyer-escalation decision
│   │   ├── guardrails_manager.py # NeMo Guardrails wrapper (HIGH_STAKES_KEYWORDS consolidated from metacognitive)
│   │   ├── graph_db.py           # Phase 3 — Neo4j driver singleton + set_graph_client test seam
│   │   ├── legal_graph_ingest.py # Phase 3 — idempotent MERGE Statute→Article during ingest
│   │   ├── legal_graph_tools.py  # Phase 3 — recall_legal_graph multi-hop FunctionTool
│   │   ├── rlhf_store.py         # Phase 4 — 👍/👎 Qdrant good pool + MySQL audit, user-scoped
│   │   ├── agent_tool_wrappers.py# FunctionTool registry (incl. recall_legal_graph_func_tool)
│   │   ├── import_data.py        # Legacy JSONL importer (incremental) + graph-ingest hook
│   │   └── pipeline/             # Multi-source Ingestion Pipeline
│   │       ├── orchestrator.py   # one core loop, idempotent, per-doc isolation
│   │       ├── run.py            # CLI entrypoint
│   │       ├── schema.py         # RawDocument / ParsedDocument / ChunkedDocument (frozen)
│   │       ├── state.py          # pipeline_documents status table (idempotency)
│   │       ├── storage.py        # raw/processed/serving three-tier lake
│   │       ├── parsers.py        # parse by source_type (json/md/html/pdf)
│   │       ├── chunker.py        # semantic/token chunking (wraps splitter)
│   │       ├── embedder.py       # embed + upsert Qdrant + MySQL chunk metadata (dedup/orphan GC)
│   │       └── connectors/       # jsonl_qa, markdown, html, pdf + base
│   ├── Dockerfile                # Container configuration
│   ├── entrypoint.sh             # Container startup script
│   └── requirements.txt          # Python dependencies
│
├── 🌐 frontend/                   # Web interface (Streamlit)
│   ├── chat_interface.py         # Main chat application with live trace expander
│   ├── config.toml               # Streamlit styling & config
│   ├── Dockerfile                # Container configuration
│   └── requirements.txt          # Python dependencies
│
├── 🔄 data_pipeline/              # Data cleaning & preprocessing
│   ├── utils/
│   │   ├── download_embed_data.ipynb  # Download legal corpus
│   │   ├── merge_instruction_data.py  # Merge instruction datasets
│   │   └── process_finetune_data.ipynb# Process training data
│   └── requirements.txt          # Python dependencies
│
├── 🤖 llm_finetuning_serving/     # LLM fine-tuning and serving
│   ├── data_processing/          # Data processing for LLaMA model
│   ├── docker/                   # Docker configurations for LLM serving
│   ├── evaluation/               # Model evaluation scripts
│   ├── finetune/                 # LLaMA fine-tuning scripts
│   ├── serving/                  # Model serving script (vLLM/Ollama)
│   ├── do_spaces_manager.py      # DigitalOcean Spaces manager
│   ├── prepare_data.sh           # Data preparation script
│   └── requirements.txt          # Python dependencies
│
├── 🗄️ database/                  # Database setup
│   ├── init.sql                  # Initial schema setup for MySQL/PostgreSQL
│   └── docker-compose.yml        # Docker compose configuration
│
├── 🚀 embed_serving/             # Embedding serving and deployment
│   ├── docker-compose.serving.yml# Production deployment configurations
│   ├── Dockerfile.cpu-serving    # CPU serving container configuration
│   ├── requirements_serving.txt  # Serving dependencies
│   ├── scripts/
│   │   ├── download_model_from_spaces.py # Download model from DO Spaces
│   │   └── serve_model.py        # Local model serving script
│   └── GPU_CPU_DEPLOYMENT_GUIDE.md# Deployment guide
│
├── 🧪 tests/                      # Pytest suite
│   ├── conftest.py               # Pytest configurations & sys.path setup
│   ├── test_api_simple.py        # Simple API validation tests
│   ├── test_backend_utils.py     # Backend utility tests
│   ├── test_basic.py             # Basic flow tests
│   ├── test_checkpoint_phase_b.py# Checkpointing tests
│   ├── test_crag_phase_a.py      # CRAG validation tests
│   ├── test_handoff_command.py   # Multi-agent handoff command tests
│   ├── test_per_conversation_memory.py # Conversation memory leak tests
│   ├── test_react_toolcalls.py   # ReAct tool-call surfacing tests
│   ├── test_sse_stream.py        # SSE live-trace streaming tests
│   ├── test_trace_tables.py      # MySQL trace tables validation tests
│   ├── test_verify_answer.py     # Phase 1 — PEV gate logic + verify recovery loop (10 tests)
│   ├── test_metacognitive.py     # Phase 2 — stakes classifier + escalation E2E (13 tests)
│   ├── test_graph_memory.py      # Phase 3 — Neo4j ingest/recall with fake driver (12 tests)
│   └── test_rlhf_store.py        # Phase 4 — 👍/👎 store + few-shot + rerank boost (12 tests)
│
├── 📝 docs/                       # Documentation
│   ├── ARCHITECTURE.md           # System architecture guide
│   ├── TESTING.md                # Testing documentation
│   └── architecture_template.drawio # Draw.io architecture file
│
├── Makefile                      # Build automation shortcuts
├── mypy.ini                      # MyPy configurations
├── .pre-commit-config.yaml       # Pre-commit hooks config
├── pyproject.toml                # Project configurations
├── requirements_dev.txt          # Development dependencies
└── setup.cfg                     # Setup configurations
```

---

## IV. Key Features & Capabilities

### 1. Multi-Source Ingestion Pipeline (`backend/src/pipeline/`)
One orchestrator loop, many connectors. Adding a data source = adding one connector — no per-source clones of the fetch→parse→chunk→embed logic.
*   **Connectors:** `JsonlQaConnector` (legacy `{question, context}` JSONL), `MarkdownConnector` (recursive `*.md`), `HtmlConnector` (local `*.html` + optional URL list), `PdfConnector` (`*.pdf`, base64-stored raw bytes).
*   **Three-tier immutable storage lake** (`data/pipeline_lake/{raw,processed,serving}`): raw is write-once (re-run any experiment from raw without re-fetching); processed/serving carry lineage JSON.
*   **Idempotent re-runs:** the `pipeline_documents` state table tracks each doc's lifecycle (`fetched → parsed → chunked → embedded | failed`). Already-embedded docs are skipped; re-runs are cheap and safe.
*   **Per-doc failure isolation:** one bad PDF never halts the JSONL batch — each doc is marked independently.
*   **Incremental embeddings:** MD5 chunk hashes detect unchanged chunks (skip) and orphaned chunks (delete from Qdrant + MySQL). Writes into the **same** Qdrant collection the RAG engine reads from — no second serving store.
*   **CLI + REST:** `python -m pipeline.run --source-type ...` or `POST /pipeline/ingest`.

### 2. LangGraph Intent Router & Query Expansion
*   **Intent Classification:** routes user queries to the optimal pipeline (`legal_rag`, `agent_tools`, `web_search`, `general_chat`) with a keyword-heuristic fallback when the LLM route is invalid.
*   **Contextual Query Rewriter:** rewrites short follow-ups into standalone questions using conversation history (e.g. *"What if it is 15 days late?"* → *"What is the contract penalty for a 15-day delay?"*).
*   **Synonym Query Expansion:** expands queries with Vietnamese legal synonyms to maximize retrieval recall.

### 3. Advanced RAG & Hybrid Search
*   **Hybrid Search:** dense vector retrieval in Qdrant combined with sparse keyword search (BM25) via LlamaIndex `QueryFusionRetriever`.
*   **Reranking:** re-orders retrieved chunks to inject the most relevant legal context into the generator (local BGE reranker, with Cohere cloud fallback).
*   **Two-Tier Incremental Re-Indexing:** MD5 hashes skip unchanged documents, avoiding redundant embeddings and lowering API costs.
*   **Garbage Collection:** automatically deletes orphaned vector chunks from Qdrant and metadata rows from MySQL.

### 4. Agentic Legal Calculators (ReAct Agent)
Powered by LlamaIndex `ReActAgent` (built **lazily** on first use so importing the module never requires LLM env vars/network). Memory is **per-conversation** — keyed by `(user_id, conversation_id)` in an LRU cache (cap 32), fixing a prior global-memory cross-user leak. Tool calls are captured via the `agent_tool_calls` contextvar (`@track_tool_call`) and surfaced through the graph → Celery result → async poll as an optional `tool_calls` array. The agent triggers programmatic tools, each guarded by input-range validation:
*   **Contract Penalty Calculator:** penalty fees under commercial law, applying the 12% legal ceiling cap of contract value.
*   **Inheritance Share Calculator:** splits inheritance among the first line of heirs under the Vietnamese Civil Code.
*   **Legal Age Verifier:** checks age eligibility for signing contracts, marriage, work, and criminal liability (gender-aware: male 20 / female 18 for marriage).
*   **Business Naming Validator:** flags business names violating legal naming guidelines.
*   **Statute of Limitations Lookup:** time limits for civil, labor, administrative, and criminal cases.
*   **Web tools:** Google-style search, Tavily AI search, Tavily Q&A quick-answer.

### 4b. Multi-Agent Planner & Supervisor Handoff
*   **Planner (`planner.py`):** before a specialist runs, an LLM emits an ordered `<plan>` of steps, each assigning a specialist (`rag` | `tool` | `web` | `chat`) + a Vietnamese goal. Simple queries → 1 step; multi-step queries (e.g. *"tính trợ cấp thôi việc rồi dẫn điều luật áp dụng"*) → 2+. `parse_plan` tolerates missing wrappers / aliases / single+double quotes; `validate_plan` caps at `MAX_PLAN_STEPS=5` and dedupes consecutive identical steps. Falls back to a router-derived 1-step plan on any LLM failure.
*   **Supervisor (`supervisor.py`):** after each specialist answers, an LLM emits a `<handoff next="...|END" rationale="..."/>` decision. Prefers the LLM; on failure/unparseable output falls back to `heuristic_handoff` (the same Vietnamese keyword markers used before — `không tìm thấy` → web, `cần tra cứu` → rag/tool). `MAX_HANDOFF_STEPS=5` loop guard forces END so a handoff chain cannot loop forever.
*   **Non-invasive:** wraps the existing CRAG RAG loop (`route → planner → specialist → supervisor → {next | metacognitive → END}`) without removing any prior node/edge.

### 5. Episodic Memory
*   **Long-Term Memory:** extracts key facts from sessions and stores them as vectors in Qdrant.
*   **Contextual Retrieval:** dual-retrieval (laws + conversation context history) for personalized answers.

### 6. Multi-Layered Safety Guardrails
*   **Input Protection:** detects jailbreaks, prompt injections, and politically sensitive queries (NVIDIA NeMo Guardrails).
*   **Output Groundedness:** verifies generated answers against source documents to prevent hallucinations and appends legal disclaimers.

### 6b. Self-Verify — PEV verify_answer Gate
*   **Final-answer groundedness:** after generation on the RAG route, `judge_answer` decomposes the answer into claims and scores each against source contexts via Ragas-style `evaluate_faithfulness`.
*   **Verdict routing:** `supported` (≥0.7) → metacognitive; `partial` (≥0.35) / `unsupported` (<0.35) → recover via `rewrite_query → retrieve`, capped at `VERIFY_MAX_RETRIES=2` before degrading to metacognitive (never infinite-loops).
*   **Non-RAG skip:** `agent_tools` / `web_search` / `general_chat` routes have empty `sources` and skip the citation check.
*   **Eval surface:** the graph result exposes `verify_score`, `verify_verdict`, `retry_verify` for the eval harness.

### 6c. Self-Knowledge-of-Limits — Metacognitive Escalation
*   **Tiered stakes:** `classify_stakes` returns `high` (criminal / `ESCALATION_TOPICS`), `medium` (civil disputes: hợp đồng, ly hôn, bồi thường, thiệt hại, khởi kiện, tòa án…), or `low`.
*   **Confidence signal:** reuses the PEV `verify_score` as the groundedness confidence.
*   **Escalation rule:** high-stakes always escalate; medium-stakes escalate only when `verify_score < ESCALATION_CONFIDENCE_THRESHOLD=0.6`; low-stakes never auto-escalate.
*   **Append-only prefix:** escalation prepends a "consult a practicing lawyer" warning to the response — auditable in the trace, never a silent mutation.
*   **Dedup:** the prior duplicate criminal-keyword list in `guardrails_manager` now references `metacognitive.HIGH_STAKES_KEYWORDS` (single source of truth).

### 6d. Structured Memory — Neo4j Legal Knowledge Graph
*   **Schema:** `(:Statute {name, year})-[:HAS_ARTICLE]->(:Article {number, title, text})`, written idempotently via MERGE during ingestion.
*   **Extraction:** reuses the regex `extract_legal_metadata` extractor (law_name + article_number) — no extra LLM call for the core path.
*   **Multi-hop tool:** `recall_legal_graph_tool` parses law + article from the query and traverses the graph; gated by multi-hop keywords ("dẫn chiếu", "án lệ", "bác bỏ", "còn hiệu lực", "điều nào") in `filter_tools_for_query`.
*   **Additive / non-blocking:** Neo4j down → swallow + log; vector ingest path is unaffected. Driver is a lazy singleton with a `set_graph_client` test seam (mirrors the Qdrant pattern).

### 6e. Continual Learning — RLHF Feedback Store
*   **👍/👎 capture:** `POST /feedback` persists every rating to MySQL `agent_feedback` (audit trail) and good answers only to a Qdrant `rlhf_good_answers` pool.
*   **User-scoped privacy:** stored via `_scope_for` (`"user:{id}"`); sentinel/empty ids rejected with HTTP 400 — no cross-user leak of endorsed answers.
*   **Idempotent + dedup:** deterministic uuid5 point id `f"{user_id}|good|{question}"`; near-identical good answers (score ≥ 0.92) are skipped to avoid flooding the few-shot pool.
*   **Few-shot injection:** when the current question is semantically near a prior 👍 (score ≥ 0.85, same scope), the top good answer is appended as a system few-shot example in `generate_node` — style/grounding steer only, never copied verbatim.
*   **Rerank up-weight:** chunks whose `doc_id` backed a 👍 answer get an additive `RLHF_RERANK_BOOST=0.05` and the ranked list is re-sorted — good-answer sources win ties without overriding clearly-relevant chunks.
*   **Frontend:** 👍/👎 buttons under each assistant reply post to `/feedback` with `user_id = session_id`.

### 7. Hardened Admin Surface
*   **API-key auth:** admin endpoints (`collection/create`, `document/create`, `data/import`, `pipeline/ingest`, `collections/.../clean`) require `X-API-Key` matching `ADMIN_API_KEY`. When unset, endpoints are **refused** unless `ALLOW_UNSAFE_ADMIN=1` (dev only).
*   **Path-traversal guard:** ingestion paths are resolved safely under the data dir (`IMPORT_DATA_DIR`) — `../../etc/passwd`-style doc_ids/paths cannot escape.
*   **No data leakage:** the Vietnamese LLM endpoint has no hardcoded public IP default — it must be configured explicitly; internal errors return a generic user-facing message (details logged server-side only).
*   **Collection name validation:** FastAPI field patterns constrain collection/source identifiers.

### 7a. Authentication & RBAC (`auth.py`, `rbac.py`)
*   **JWT + bcrypt:** passwords hashed with passlib bcrypt; `python-jose` HS256 JWTs (`JWT_SECRET`, `JWT_ALG`, `JWT_EXP_MIN=60`). Endpoints `POST /auth/register` (rejects admin/lawyer self-registration), `POST /auth/login` (same message on both branches to avoid user enumeration), `GET /auth/me`. Startup `seed_admin()` creates a default admin from `SEED_ADMIN_*` env.
*   **Roles:** `admin` | `lawyer` | `user` | `guest`, each mapping to an allowed tool-name set (`ROLE_PERMISSIONS`); `admin` = all tools. `Principal` is a frozen dataclass (`is_admin`, `is_approval_exempt`).
*   **Tool policy:** `filter_tools_by_policy` drops tools the caller's role may not call **before** the ReAct agent sees them. FastAPI deps `get_current_user` (strict, 401), `get_current_user_optional` (anonymous), `require_role` / `require_admin` (403).
*   **Anonymous demo fallback:** `ALLOW_ANONYMOUS=1` keeps the old client-supplied `user_id` path working for the Streamlit demo; production should set `JWT_SECRET` and require auth.

### 7b. Approval Workflow (`approval.py`)
*   **Sensitive tools** (`generate_document_template_tool`, `web_search_tool`, `tavily_search_tool`) require human approval for non-exempt roles (`user`, `guest`); `admin` / `lawyer` are exempt.
*   **Gate:** before the ReAct loop, `evaluate_tool_gate` intersects the query's anticipated tools with the sensitive set. If blocked, a `ToolApproval(pending)` row is created and the run returns a Vietnamese *"chờ phê duyệt"* message + `approval_id` instead of executing.
*   **Decide:** `POST /approvals/{id}/decide` (admin, idempotent) → `approved` | `rejected`, audit-logged. On approval, `approval:allowed:{run_id}` (Redis + in-process fallback) lets a retry skip the gate.
*   **Scope:** the graph is re-invoked on approval rather than fully suspended (keeps the change non-invasive); still enforces human-in-the-loop on sensitive tool use.

### 7c. Subprocess Sandbox (`sandbox.py`)
*   **Defense-in-depth:** deterministic pure-compute calc tools run in a throwaway child process — scrubbed env (secret vars stripped, UTF-8 stdio) + hard timeout — so a runaway calc tool cannot wedge the agent/worker. `SAFE_TO_SANDBOX` is an explicit allowlist; network / retrieval / graph / memory / sensitive tools are **not** sandboxed.
*   **Limitation:** Windows has no seccomp — isolation is process-level + timeout + env scrub, not a syscall filter. Demo-grade, not production hardening.

### 7d. Audit Logging (`audit.py`)
*   `log_audit(user_id, action, resource, ip, payload)` is best-effort: any DB failure is caught + logged server-side so an audit-store outage never breaks a request. `GET /audit` (admin) lists entries with optional user/action filters. Login/register/chat/admin-decision events are recorded.

### 8. Multi-Provider LLM Routing
Generation routes by `LLM_PROVIDER` (`groq` | `ollama` | `openai`) with graceful fallback (Vietnamese LLM API → Groq; Ollama main → Groq). Token usage is accumulated in-process via a `usage_accumulator` contextvar for cost metrics — no external tracing service required.

### 9. Comprehensive RAG Evaluation Suite
A 4-pillar framework tracking operational metrics (token count, API cost, latency TTFT/TTLT), quality metrics (LLM-as-judge faithfulness/relevance), agentic metrics (tool-call success via the `agent_tool_calls` contextvar, router accuracy), and failure-mode analysis (Retrieval / Routing / Hallucination / Execution).

### 10. MCP Server (`backend/src/mcp_server/`)
The legal tools are also exposed as an **MCP** (Model Context Protocol) server, so any MCP-compatible client (Claude Desktop, Claude Code, the MCP inspector) can call them directly without the HTTP chat API.
*   **Tools (~18):** `contract_penalty_calculator`, `legal_age_checker`, `inheritance_calculator`, `business_name_validator`, `statute_lookup`, `severance_pay`, `overtime_pay`, `pit_monthly`, `court_fee`, `child_support`, `law_version`, `article_lookup`, `precedent_lookup`, `cross_reference`, `verify_citation`, `procedure_wizard`, `jurisdiction_resolver`, `recall_legal_graph`. Each `@mcp.tool()` is a thin wrapper over the **raw Dict-returning implementation** (not the LlamaIndex `FunctionTool` — avoids double-encoding); the Vietnamese docstring becomes the MCP description.
*   **Transports:** `stdio` (local — Claude Desktop / Claude Code) and `streamable-http` (remote/production).
*   **Run:** `python -m src.mcp_server --transport stdio|http [--host 0.0.0.0] [--port 8100]` (or `MCP_TRANSPORT` / `MCP_HOST` / `MCP_HTTP_PORT` env).
*   **Inspect:** `mcp dev src.mcp_server.server:mcp` opens the MCP inspector with the full tool list. See `backend/src/mcp_server/README.md` for the Claude Desktop config snippet.

> **Observability note (tracing):** Tracing is **self-hosted** by design: every graph run is persisted as a `GraphRun` + `AgentStep` rows in MySQL and published to a Redis pub/sub channel (`graph_trace_events`) for live SSE streaming. Tool-call tracking and token-usage accumulation are in-process contextvars feeding the eval suite. **No LangSmith / Langfuse / OpenTelemetry, no cloud egress** — Vietnamese legal data stays local. If external tracing is ever desired, add a LangSmith/Langfuse callback; it is off by default and the self-hosted trace keeps working independently.

---

## V. Technology Stack

*   **Frontend UI:** Streamlit (Python)
*   **Backend API Framework:** FastAPI
*   **Background Tasks & Queue:** Celery + Redis
*   **Vector Database:** Qdrant DB
*   **Graph Database:** Neo4j 5.20 (legal knowledge graph: Statute→Article multi-hop recall)
*   **Relational Database:** PostgreSQL / MySQL (for logs, chat history, RLHF audit, and status checks)
*   **AI Agent & Retrieval Orchestration:** LlamaIndex, LangGraph, NVIDIA NeMo Guardrails
*   **Vietnamese Text Embedding:** BGE-M3 (locally hosted/served, or Cohere cloud backup)
*   **LLM Providers:** Llama-3.1-8B-Instruct (via Groq/Ollama), OpenAI, self-hosted Legal LLM
*   **CI/CD & Containers:** Docker, Docker Compose, GitHub Actions

---

## VI. Quick Start Guide

### 1. Prerequisites
- Docker and Docker Compose v2.0+
- Python 3.10+ (for local development)
- 8GB+ RAM (16GB recommended)

### 2. Setup Configuration
```bash
# Clone the repository
git clone https://github.com/NMCuonG08/Chatbot-Legal-RAG.git
cd Chatbot-Legal-RAG

# Copy environment template
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your API keys:
```env
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly-your_key_here
COHERE_API_KEY=your_key_here
ADMIN_API_KEY=your_admin_secret_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/legal_chatbot
REDIS_URL=redis://localhost:6379/0

# Phase 3 — Neo4j legal knowledge graph (optional; graph recall disabled if unset)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
```

Start Neo4j alongside the stack:

```bash
docker compose up -d neo4j    # ports 7474 (browser) / 7687 (bolt)
```

Graph ingest + recall are best-effort — if Neo4j is down, the vector RAG path keeps working unchanged.

### 3. Run the Services

**Celery Worker:**
```bash
cd backend/src
celery -A tasks.celery_app worker --loglevel=info -P solo
```

**FastAPI Backend:**
```bash
cd backend/src
uvicorn app:app --host 0.0.0.0 --port 8002
```

**Streamlit Frontend:**
```bash
cd frontend
streamlit run chat_interface.py --server.port 8501
```

---

## VII. Ingestion & Data Pipelines

Detailed workflow configuration for the multi-source ingestion pipeline:

```bash
cd backend/src

# Ingest JSONL document format (e.g. train.jsonl)
python -m pipeline.run --source-type jsonl --path <path_to_jsonl_file> --collection llm

# Ingest Markdown directory
python -m pipeline.run --source-type markdown --path <path_to_markdown_directory>

# Ingest raw HTML pages directory
python -m pipeline.run --source-type html --path <path_to_html_directory>

# Ingest PDF documents (using token chunking instead of semantic)
python -m pipeline.run --source-type pdf --path <path_to_pdf_directory> --no-semantic
```

### 1. Data Storage & Lineage
Files processed by the ingestion connectors are managed securely under a local three-tier storage lake structure (`data/pipeline_lake/{raw,processed,serving}`). The raw ingestion inputs and intermediate parsed outputs are preserved locally to guarantee idempotency and prevent duplicate API costs.

### 2. Data Statistics
*   **Original Legal Dataset (`train.jsonl`):** 89,261 raw Q&A legal document entries (File size: 162.13 MB).
*   **Unique Processed QA Pairs (`train_qa_format.jsonl`):** 19,536 high-quality fine-tuning records (File size: 42.27 MB).
*   **Stored Chunks (Local MySQL/PostgreSQL):** 62,854 document chunk metadata points mapped across 19,536 unique documents.
*   **Coverage:** Hand-curated Vietnamese civil code guidelines, business naming regulations, contract penalty benchmarks, and statutes of limitations.

---

## VIII. Model Training & Configuration

### 1. Training Architecture & Hardware Profiles
The LLM fine-tuning setup utilizes **Unsloth** for accelerated memory-efficient training of `Llama-3.1-8B-Instruct`. Configurations are optimized for three target hardware tiers (defined in `llm_finetuning_serving/finetune/config.py`):

*   **H200 Profile (141GB VRAM):** Full 16-bit precision training, `lora_r=128`, `lora_alpha=256`, effective batch size of 128 (16 batch size × 8 accumulation steps), learning rate `3e-4`.
*   **H100 Profile (80GB VRAM):** Full 16-bit precision training, `lora_r=64`, `lora_alpha=128`, effective batch size of 64 (8 batch size × 8 accumulation steps), learning rate `2e-4`.
*   **A4000 Profile (16GB VRAM):** 4-bit quantized training using `paged_adamw_8bit`, `lora_r=16`, `lora_alpha=32`, effective batch size of 16 (2 batch size × 8 accumulation steps), learning rate `2e-4`.

### 2. Training Execution
Finetuning runs through `llm_finetuning_serving/finetune/train_llama.py` with custom sequence lengths up to 8192 for processing long context legal documents. Real-time loss metrics, evaluation scores, and learning rate decay are monitored via standard training step logs.

### 3. Embedding Verification
Verification checks are run locally before deployment to ensure computed vector embeddings map correctly to the active Qdrant collections.


---

## IX. Production Deployment

📖 **[GPU CPU Deployment Guide](embed_serving/GPU_CPU_DEPLOYMENT_GUIDE.md)**  
☁️ **[AWS Single-Instance Deployment Plan](DEPLOY_AWS.md)**

Deploy to staging or production using Docker Compose in production mode:
```bash
cd embed_serving
docker-compose -f docker-compose.serving.yml up -d
```

### Security checklist:
- [x] API rate limiting enabled
- [x] Database credentials encrypted
- [x] HTTPS/SSL certificates setup
- [x] Firewall rules configured
- [x] Local tracing (no third-party cloud data leak)

---

## X. Testing & Quality Assurance

Run the test suite with `pytest` from the root directory:
```bash
python -m pytest tests/ -v
```

Agentic-feature units (no live Neo4j / Qdrant / MySQL required — all use fakes/mocks):

```bash
python -m pytest tests/test_verify_answer.py tests/test_metacognitive.py \
                  tests/test_graph_memory.py tests/test_rlhf_store.py -v
```

Current suite: **273 passing** (incl. Phase 1 verify_answer 10, Phase 2 metacognitive 13, Phase 3 graph memory 12, Phase 4 RLHF 12).

See [`docs/TESTING.md`](docs/TESTING.md) for full test suite coverage (including security layer, checkpointing, trace tables, CRAG flows, SSE streaming, memory leakage, PEV verify, metacognitive escalation, Neo4j graph memory, and RLHF feedback tests).

---

## XI. API Documentation

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/` | – | Root check |
| GET | `/health` | – | Health check |
| GET | `/collections` | – | List Qdrant collections |
| GET | `/documents` | – | List documents |
| GET | `/collections/{name}/points` | – | List points in a collection |
| GET | `/history/{user_id}` | – | Conversation history |
| DELETE | `/history/{user_id}` | API key | Clear history |
| POST | `/chat/complete` | – | Submit a chat query (async Celery task) |
| GET | `/chat/complete/{task_id}` | – | Poll task result |
| GET | `/chat/stream/{task_id}` | – | SSE live trace stream |
| POST | `/feedback` | – | RLHF 👍/👎 feedback (user-scoped; sentinels rejected with 400) |
| POST | `/collection/create` | API key | Create a Qdrant collection |
| DELETE | `/collections/{name}/clean` | API key | Delete vectors in a collection |
| POST | `/document/create` | API key | Create a document |
| POST | `/data/import` | API key | Legacy JSONL import |
| POST | `/pipeline/ingest` | API key | Multi-source pipeline ingestion |

*   **Frontend UI:** http://localhost:8501
*   **Backend API Docs:** http://localhost:8002/docs
*   **Qdrant Dashboard:** http://localhost:6333/dashboard

---

## XII. Disclaimer & Terms

> [!WARNING]
> This system is designed for **research, educational, and reference support purposes**.
> AI-generated results cannot replace consultation from qualified legal professionals. Always verify information with legal professionals before making decisions. The system may have errors and does not guarantee 100% accuracy.

---

<div align="center">

**⭐ If this project is helpful, please star the repository to support the development team! ⭐**

Made with ❤️ for Vietnamese Legal Community

[🌟 Star](https://github.com/NMCuonG08/Chatbot-Legal-RAG/stargazers) • [🍴 Fork](https://github.com/NMCuonG08/Chatbot-Legal-RAG/fork) • [📚 Docs](docs/ARCHITECTURE.md) • [💬 Discord](https://discord.gg/legal-chatbot)

</div>