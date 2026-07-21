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

It also carries a **CoALA / MemGPT 5-layer memory** (Working → Short → Episodic → Semantic → Procedural) that survives cross-worker restarts (Redis-backed rolling summary), extracts only user facts — never bot law text — via a delta-only episodic path, and steers the conversation with a structured `UserProfile` + per-case-type procedural workflow. Tool selection is **semantic** (query↔tool-description embedding similarity) rather than brittle keyword matching, and every arithmetic tool is guarded by **Pydantic input schemas** so a malformed arg returns a clean error instead of a crash or silent garbage.

[🌟 Star](https://github.com/NMCuonG08/Chatbot-Legal-RAG/stargazers) • [🍴 Fork](https://github.com/NMCuonG08/Chatbot-Legal-RAG/fork) • [📚 Docs](docs/ARCHITECTURE.md) • [💬 Discord](https://discord.gg/legal-chatbot)

</div>

---

## 📋 Table of Contents

- [I. Overview](#i-overview)
- [II. System Architecture](#ii-system-architecture)
  - [Memory Architecture — CoALA / MemGPT 5-Layer](#3-memory-architecture--coala--memgpt-5-layer)
  - [Tool-Calling Architecture](#4-tool-calling-architecture)
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

### 3. Memory Architecture — CoALA / MemGPT 5-Layer

The agent memory was rebuilt from an ad-hoc OrderedDict/episodic-blob design into a proper **5-layer CoALA (Cognitive Architectures for Language Agents) / MemGPT** model. Four prior architectural flaws are closed:

```mermaid
%%{init: { 'themeVariables': { 'fontFamily': 'system-ui, -apple-system, sans-serif', 'fontSize': '12px' } } }%%
graph LR
    subgraph Working[Working Memory]
        GS["`ChatGraphState slots
        active_entities / current_intent
        tool_budget / tool_calls_made
        scratchpad`"]
    end
    subgraph Short[Short-Term]
        RS["`memory_short_term.py
        Redis rolling summary
        mem:short:{u}:{c}:summary|count
        TTL 7d, LRU32 read-through`"]
    end
    subgraph Episodic[Episodic]
        EP["`UserEpisode (MySQL) + Qdrant
        delta-only extract per turn
        strip_legal_citations
        dedup score_threshold=0.88`"]
    end
    subgraph Semantic[Semantic]
        UP["`UserProfile (MySQL KV)
        name / birth_year / gender
        location / case_type / case_summary
        idempotent non-null merge`"]
    end
    subgraph Procedural[Procedural]
        PW["`procedural_memory.py
        CASE_WORKFLOWS per case_type
        inheritance/land/marriage/
        business/traffic/other`"]
    end
    GS -->|summarize every N turns| RS
    RS -->|recall context| GS
    EP -->|extract structured| UP
    UP -->|case_type -> workflow_block| GS
    PW -->|inject scaffold| GS
```

| Layer | Store | Module | Purpose | Flaw closed |
|-------|-------|--------|---------|-------------|
| **Working** | LangGraph state | `tasks.py` `ChatGraphState` | Per-turn scratchpad: `active_entities`, `current_intent`, `tool_budget`/`tool_calls_made`, `scratchpad` | — |
| **Short-Term** | Redis | `memory_short_term.py` | Cross-worker rolling summary (`mem:short:{user}:{conv}:{summary|count}`, TTL 7d, LRU32 read-through, graceful fallback) | **FLAW 1** (multi-worker drift — RAM-local OrderedDict lost on worker switch) |
| **Episodic** | MySQL `user_episodes` + Qdrant | `tasks.py` `save_episodic_memory_task` | Delta-only extraction (O(1)/turn, never ingests assistant text), `strip_legal_citations` regex, dedup `score_threshold=0.88`, uuid5 idempotent points | **FLAW 2** (O(N²) full-history re-summarize + bot law text polluting user facts) |
| **Semantic** | MySQL `user_profiles` | `models.py` `UserProfile` + `merge_user_profile` | Structured KV (name, birth_year, gender, location, case_type, case_summary); idempotent non-null merge | **FLAW 4** (free-text blob, no structured profile) |
| **Procedural** | Static dict | `procedural_memory.py` | `CASE_WORKFLOWS` per case_type → `workflow_block()` injected into agent system prompt | **FLAW 4** (no per-case procedural scaffold) |

*   **ReAct recall is agent-decided (MemGPT pattern):** long-term memory is no longer auto-injected into every RAG prompt. The `recall_user_memory_tool` (ReAct) is gated at threshold 0.65 so the agent calls it only when the query signals recall — closing **FLAW 3** (stale episodic facts bleeding into unrelated questions, e.g. an inheritance fact contaminating a vehicle-fee query). Auto-inject is disabled by default (`RAG_AUTO_INJECT_DISABLED=1`) and available for 1-release rollback.
*   **Online monitoring:** Prometheus counters (`memory_metrics.py`) — `memory_short_term_hits/misses`, `episodic_extractions{result}`, `episodic_pollution`, `react_recall{hit}`, `profile_merge{field}` — with a `_NoopCounter` fallback when prometheus_client is absent. Non-invasive, mountable at `/metrics`.
*   **Feature flags (default = NEW behavior, old available 1 release):** `MEMORY_REDIS_ENABLED`, `EPISODIC_DELTA_ENABLED`, `RAG_AUTO_INJECT_DISABLED`(default 1), `STRUCTURED_PROFILE_ENABLED`, `PROCEDURAL_WORKFLOW_ENABLED`.
*   **Migration (zero-downtime):** `create_all` auto-creates `user_profiles`; `backend/scripts/backfill_user_profiles.py` is a one-shot, idempotent backfill (concat a user's `UserEpisode` rows → extract prompt → `merge_user_profile`; dry-run by default, `--apply` to write). Redis short-term starts empty — first turn re-summarizes from DB.

### 4. Tool-Calling Architecture

*   **Semantic tool router (`tool_router.py`):** tool selection embeds the query against each tool's description and keeps the top-k above a similarity threshold (`TOOL_SIM_THRESHOLD=0.30`, `TOOL_TOP_K=8`), robust to paraphrase — *"công ty cho tôi nghỉ"* matches `severance_pay_tool` even though no keyword matches. The tool-description index is built once per process (descriptions are static). Falls back to the keyword path (`filter_tools_for_query`) when the embedding service is down or the flag is off (`SEMANTIC_TOOL_ROUTER_ENABLED`), so behavior degrades gracefully with zero regression risk.
*   **Pydantic input validation (`agent_tool_wrappers.py`):** every arithmetic tool (severance, PIT, land/vehicle fee, court fee, legal age, inheritance) declares a Pydantic `fn_schema` with `Field` constraints + Vietnamese descriptions, passed to `FunctionTool.from_defaults(fn=..., fn_schema=...)`. A `@_validated(Model)` decorator coerces + validates inputs in-process before the fn body, returning a clean Vietnamese JSON error on `ValidationError` — a malformed arg (`"5 năm"` for an `int`) no longer crashes (legal_age `str < int` TypeError) or silently computes garbage (severance `"5 VNĐ"`).
*   **Dynamic tool filtering + RBAC:** the chosen tools are further narrowed by role (`rbac.filter_tools_by_policy`) so a guest never sees sensitive/search tools and a user never sees admin-only escalation paths. Sensitive tools (`generate_document_template`, `web_search`, `tavily`) require human approval for non-exempt roles.
*   **Loop guards:** `MAX_HANDOFF_STEPS=5` (supervisor), `GRAPH_RECURSION_LIMIT=32`, per-run done-flags (`agent_to_rag_done`/`generate_to_web_done`) that detect when the inner ReAct agent already called web/retrieval tools and suppress duplicate supervisor handoffs, plus the `agent_empty_streak` no-retry guard that kills a cumulative 75s waste loop when a tool keeps returning empty with identical args.
*   **30+ custom legal tools** across calculation (contract penalty, severance, overtime, PIT, land/vehicle registration fee, court fee, child support, inheritance, legal age, statute of limitations, business-name, law-version), retrieval (article/precedent lookup, cross-reference, citation verify, unified parallel search), knowledge-graph multi-hop recall, procedure wizard, jurisdiction resolver, document-template generation, Tavily web search, current time, and long-term memory recall.

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
│   │   ├── config.py             # CRAG + PEV + metacognitive + RLHF + task-retry tuning constants
│   │   ├── retry_utils.py         # with_retry / awith_retry + build_transient_exceptions (lifecycle retry)
│   │   ├── database.py           # Database connection & session orchestration
│   │   ├── cache.py              # Redis cache integration
│   │   ├── tasks.py              # Celery tasks + LangGraph StateGraph (CRAG loop, PEV verify, metacognitive, RLHF, handoff, trace, checkpoint, working-memory slots)
│   │   ├── trace.py              # Self-hosted trace: MySQL agent_steps/graph_runs + Redis pub/sub emit_*
│   │   ├── verify_answer.py      # Phase 1 — PEV gate: judge final answer groundedness (claim decomp + verdict)
│   │   ├── metacognitive.py      # Phase 2 — tiered stakes classifier + lawyer-escalation decision
│   │   ├── guardrails_manager.py # NeMo Guardrails wrapper (HIGH_STAKES_KEYWORDS consolidated from metacognitive)
│   │   ├── graph_db.py           # Phase 3 — Neo4j driver singleton + set_graph_client test seam
│   │   ├── legal_graph_ingest.py # Phase 3 — idempotent MERGE Statute→Article during ingest
│   │   ├── legal_graph_tools.py  # Phase 3 — recall_legal_graph multi-hop FunctionTool
│   │   ├── rlhf_store.py         # Phase 4 — 👍/👎 Qdrant good pool + MySQL audit, user-scoped
│   │   ├── agent_tool_wrappers.py# FunctionTool registry (30+ tools, Pydantic fn_schema on calc tools, recall_legal_graph_func_tool)
│   │   ├── tool_router.py        # Semantic tool router — query↔tool-description embedding top-k cosine (keyword fallback)
│   │   ├── memory_short_term.py  # Short-term memory — Redis cross-worker rolling summary (CoALA layer 2)
│   │   ├── procedural_memory.py  # Procedural memory — CASE_WORKFLOWS per case_type → workflow_block (CoALA layer 5)
│   │   ├── memory_metrics.py     # Prometheus counters for memory ops (no-op safe fallback)
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
│   ├── scripts/
│   │   └── backfill_user_profiles.py  # One-shot idempotent backfill: UserEpisode → UserProfile (dry-run default)
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
│   ├── test_rlhf_store.py        # Phase 4 — 👍/👎 store + few-shot + rerank boost (12 tests)
│   ├── test_short_term_memory_redis.py # CoALA layer 2 — Redis rolling summary + cross-worker consistency (8)
│   ├── test_episodic_extraction.py     # CoALA layer 3 — delta-only extract, pollution guard (8)
│   ├── test_rag_context_no_contamination.py # FLAW 3 — RAG prompt clean of stale episodic facts (5)
│   ├── test_structured_profile.py      # CoALA layer 4 — UserProfile merge + procedural workflow (14)
│   ├── test_working_memory_state.py    # CoALA layer 1 — ChatGraphState slots + tool_budget (5)
│   ├── test_memory_eval.py             # Memory golden regression gate — FLAW1-4 + P4/P5/P6 (8)
│   ├── test_tool_selection.py           # Keyword tool-filter fallback (retrieval-first) regression (3)
│   ├── test_tool_router.py             # Semantic tool router — cosine/always-include/fallback (8 + 1 integration)
│   └── test_calc_tool_validation.py     # Pydantic fn_schema validation on 6 calc tools (10)
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
Powered by LlamaIndex `ReActAgent` (built **lazily** on first use so importing the module never requires LLM env vars/network). Memory is **per-conversation** — short-term rolling summary now lives in **Redis** (cross-worker source of truth) with an LRU cache (cap 32) read-through, fixing the prior RAM-local multi-worker drift. Tool calls are captured via the `agent_tool_calls` contextvar (`@track_tool_call`) and surfaced through the graph → Celery result → async poll as an optional `tool_calls` array. The agent triggers programmatic tools, each guarded by **Pydantic `fn_schema` input validation** (coerce + range-check + clean Vietnamese JSON error on bad args — no crash, no silent garbage) and selected per turn by the **semantic tool router**:
*   **Contract Penalty Calculator:** penalty fees under commercial law, applying the 12% legal ceiling cap of contract value.
*   **Inheritance Share Calculator:** splits inheritance among the first line of heirs under the Vietnamese Civil Code.
*   **Legal Age Verifier:** checks age eligibility for signing contracts, marriage, work, and criminal liability (gender-aware: male 20 / female 18 for marriage).
*   **Business Naming Validator:** flags business names violating legal naming guidelines.
*   **Statute of Limitations Lookup:** time limits for civil, labor, administrative, and criminal cases.
*   **Knowledge-driven calculators:** severance pay (Điều 48 BL LĐ 2019), overtime pay (Điều 107), PIT monthly (luỹ tiến), land & vehicle registration fee (trước bạ), court fee (NQ 326), child support.
*   **Web tools:** Google-style search, Tavily AI search, Tavily Q&A quick-answer.

### 4b. Multi-Agent Planner & Supervisor Handoff
*   **Planner (`planner.py`):** before a specialist runs, an LLM emits an ordered `<plan>` of steps, each assigning a specialist (`rag` | `tool` | `web` | `chat`) + a Vietnamese goal. Simple queries → 1 step; multi-step queries (e.g. *"tính trợ cấp thôi việc rồi dẫn điều luật áp dụng"*) → 2+. `parse_plan` tolerates missing wrappers / aliases / single+double quotes; `validate_plan` caps at `MAX_PLAN_STEPS=5` and dedupes consecutive identical steps. Falls back to a router-derived 1-step plan on any LLM failure.
*   **Supervisor (`supervisor.py`):** after each specialist answers, an LLM emits a `<handoff next="...|END" rationale="..."/>` decision. Prefers the LLM; on failure/unparseable output falls back to `heuristic_handoff` (the same Vietnamese keyword markers used before — `không tìm thấy` → web, `cần tra cứu` → rag/tool). `MAX_HANDOFF_STEPS=5` loop guard forces END so a handoff chain cannot loop forever.
*   **Non-invasive:** wraps the existing CRAG RAG loop (`route → planner → specialist → supervisor → {next | metacognitive → END}`) without removing any prior node/edge.

### 5. 5-Layer CoALA Memory (Working / Short / Episodic / Semantic / Procedural)
*   **Short-Term (`memory_short_term.py`):** Redis-backed rolling summary + count, key `mem:short:{user}:{conv}:{summary|count}`, TTL 7d, LRU32 read-through cache. The cross-worker source of truth — two uvicorn/Celery workers see the same summary (fixes FLAW 1). Graceful fallback to in-process dict when Redis is down.
*   **Episodic (`save_episodic_memory_task`):** delta-only extraction (O(1)/turn, never re-summarizes full history, never ingests assistant text) via `prompts/episodic_extract.v1.txt` + `strip_legal_citations` regex. Stores `UserEpisode` (MySQL audit) + Qdrant vector (semantic recall), dedup `score_threshold=0.88`, uuid5 idempotent points (fixes FLAW 2).
*   **Semantic (`models.UserProfile`):** structured KV (name, birth_year, gender, location, case_type, case_summary) merged idempotently (non-null fields only) by `merge_user_profile`. Backed up by `backend/scripts/backfill_user_profiles.py`.
*   **Procedural (`procedural_memory.py`):** `CASE_WORKFLOWS` per case_type (inheritance/land/marriage/business/traffic/other) → `workflow_block()` scaffold injected into the agent system prompt when the user's `case_type` is known.
*   **ReAct recall (MemGPT pattern):** the `recall_user_memory_tool` (threshold 0.65) lets the **agent decide** when to recall long-term memory, instead of auto-injecting episodic facts into every RAG prompt (FLAW 3, disabled by `RAG_AUTO_INJECT_DISABLED=1`).
*   **Online monitoring:** Prometheus counters in `memory_metrics.py` (`memory_short_term_hits/misses`, `episodic_extractions{result}`, `episodic_pollution`, `react_recall{hit}`, `profile_merge{field}`).

### 5b. Semantic Tool Router (`tool_router.py`)
*   **Embedding-based selection:** the query is embedded against each tool's description; top-k above `TOOL_SIM_THRESHOLD=0.30` are kept. Robust to paraphrase — *"công ty cho tôi nghỉ"* matches `severance_pay_tool` with no keyword overlap. The tool-description index is built once per process.
*   **Graceful fallback:** returns `None` when disabled or the embedding service is down → `filter_tools_for_query` falls back to the keyword path, so behavior never regresses.
*   **Always-include contract:** `legal_disclaimer` + `current_time` + `tavily` are always attached; `recall_user_memory` is added when conversation history exists — matching the old keyword-path contract.

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
Generation routes by `LLM_PROVIDER` (`groq` | `ollama` | `openai`) with graceful fallback (Vietnamese LLM API → Groq; Ollama main → Groq). Every provider path is wrapped in `FallbackLLM` so transient calls (429/5xx/timeout) retry 3× with exponential backoff before cross-provider fallback — single-provider configs use `primary==fallback` so the retry itself is the resilience. Token usage is accumulated in-process via a `usage_accumulator` contextvar for cost metrics — no external tracing service required. See [Failure handling §7.1](docs/ARCHITECTURE.md#71-agent-lifecycle-retry--resilience-4-layers) for the full 4-layer retry lifecycle (LLM call → embedding → graph self-correction → Celery task).

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
*   **Agent Memory:** CoALA / MemGPT 5-layer (Redis short-term + MySQL episodic/semantic + Qdrant episodic + procedural scaffold)
*   **Tool Selection:** Semantic embedding router (query↔tool-description cosine) with keyword fallback
*   **Input Validation:** Pydantic v2 `fn_schema` on every arithmetic tool
*   **Monitoring:** Prometheus counters (memory + tool ops, no-op safe) at `/metrics`
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
# Memory architecture golden gate (CoALA 5-layer, FLAW 1-4)
python -m pytest tests/test_short_term_memory_redis.py tests/test_episodic_extraction.py \
                  tests/test_rag_context_no_contamination.py tests/test_structured_profile.py \
                  tests/test_working_memory_state.py tests/test_memory_eval.py -v

# Tool-calling: semantic router + Pydantic validation + keyword fallback
python -m pytest tests/test_tool_router.py tests/test_calc_tool_validation.py \
                  tests/test_tool_selection.py -v
```

Current suite: **656 passing** (offline gate) — incl. memory rebuild (48: short-term 8, episodic 8, no-contamination 5, structured-profile 14, working-memory 5, eval gate 8), tool-calling (21: semantic router 8+1, Pydantic calc 10, keyword fallback 3), plus Phase 1 verify_answer 10, Phase 2 metacognitive 13, Phase 3 graph memory 12, Phase 4 RLHF 12, and the security/RBAC/approval/auth/checkpoint/trace/SSE suites. The `test_tool_router.py::test_real_paraphrase_matches_severance` integration test (real embedding, ~105s) is deselected by the offline gate.

See [`docs/TESTING.md`](docs/TESTING.md) for full test suite coverage (including security layer, checkpointing, trace tables, CRAG flows, SSE streaming, memory leakage, PEV verify, metacognitive escalation, Neo4j graph memory, RLHF feedback, CoALA 5-layer memory, and semantic tool router tests).

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