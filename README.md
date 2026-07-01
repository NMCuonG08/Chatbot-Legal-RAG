# ⚖️ Vietnamese Legal Assistant (RAG & Agentic Chatbot)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-orange.svg)](https://www.llamaindex.ai/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-brightgreen.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)

An intelligent virtual assistant for looking up Vietnamese legal documents, calculating legal costs/penalties, and verifying civil legal conditions. Built on an **Advanced RAG** architecture combined with an **Agentic Workflow** (LangGraph router → ReAct agent), protected by multi-layered safety guardrails and a hardened admin surface.

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

    GenRag & Agent & Web & Gen -->|Output guardrails| Out[Final response]
    Out -->|save + trace run_end| Broker
    Out -->|conversation history| SQL[("MySQL")]
```

**Key Graph Properties:**
*   **Self-corrective RAG (CRAG):** `retrieve → grade_documents → {generate | rewrite_query → retrieve (loop) | web_search}`, guarded by `REFLECTION_MAX=2`. Documents are graded by rerank `relevance_score` (threshold `DOC_GRADE_THRESHOLD=0.35`) with an LLM-as-judge batch fallback for borderline docs.
*   **Multi-agent handoff:** `Command(goto=...)` edges — `agent_tools → retrieve` (agent needs legal docs), `generate → web_search` (canned "not found"), `web_search → agent_tools` (needs tool use). Three once-per-run guard flags prevent cycles.
*   **Checkpointing:** `RedisSaver` (requires Redis Stack / RedisJSON) with `MemorySaver` auto-fallback; isolation by `thread_id = conversation_id`, so multi-turn follow-ups resume state.
*   **Trace (self-hosted):** every node emits `node_end`/`handoff` events → MySQL `agent_steps` + Redis pub/sub `graph_trace_events`. One `GraphRun` row per turn (`run_id`, route, final response, reflection_count, tool_calls). No LangSmith/Langfuse, no cloud egress — Vietnamese legal data stays local.
*   **SSE streaming:** `GET /chat/stream/{task_id}` subscribes to the pub/sub channel, filters by `run_id`, and closes on `run_end`. The Streamlit UI renders a live `Agent trace` expander.
*   **ReAct tool-call surfacing:** the `agent_tool_calls` contextvar is reset per turn, populated by `@track_tool_call`, and lifted through the graph → Celery result → async poll as an optional `tool_calls` array.
*   **Per-conversation memory:** ReAct agent memory is keyed by `(user_id, conversation_id)` in an LRU cache (cap 32) — fixing a prior global-memory cross-user leak.

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
│   │   ├── models.py             # Pydantic data models & GraphRun/AgentStep DB models
│   │   ├── search.py             # Hybrid search index + BM25 retriever
│   │   ├── semantic_cache.py     # Qdrant-based vector caching
│   │   ├── config.py             # CRAG constants (REFLECTION_MAX, DOC_GRADE_THRESHOLD, ...)
│   │   ├── database.py           # Database connection & session orchestration
│   │   ├── cache.py              # Redis cache integration
│   │   ├── tasks.py              # Celery tasks + LangGraph StateGraph (CRAG loop, handoff, trace, checkpoint)
│   │   ├── trace.py              # Self-hosted trace: MySQL agent_steps/graph_runs + Redis pub/sub emit_*
│   │   ├── import_data.py        # Legacy JSONL importer (incremental)
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
│   └── test_trace_tables.py      # MySQL trace tables validation tests
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

### 5. Episodic Memory
*   **Long-Term Memory:** extracts key facts from sessions and stores them as vectors in Qdrant.
*   **Contextual Retrieval:** dual-retrieval (laws + conversation context history) for personalized answers.

### 6. Multi-Layered Safety Guardrails
*   **Input Protection:** detects jailbreaks, prompt injections, and politically sensitive queries (NVIDIA NeMo Guardrails).
*   **Output Groundedness:** verifies generated answers against source documents to prevent hallucinations and appends legal disclaimers.

### 7. Hardened Admin Surface
*   **API-key auth:** admin endpoints (`collection/create`, `document/create`, `data/import`, `pipeline/ingest`, `collections/.../clean`) require `X-API-Key` matching `ADMIN_API_KEY`. When unset, endpoints are **refused** unless `ALLOW_UNSAFE_ADMIN=1` (dev only).
*   **Path-traversal guard:** ingestion paths are resolved safely under the data dir (`IMPORT_DATA_DIR`) — `../../etc/passwd`-style doc_ids/paths cannot escape.
*   **No data leakage:** the Vietnamese LLM endpoint has no hardcoded public IP default — it must be configured explicitly; internal errors return a generic user-facing message (details logged server-side only).
*   **Collection name validation:** FastAPI field patterns constrain collection/source identifiers.

### 8. Multi-Provider LLM Routing
Generation routes by `LLM_PROVIDER` (`groq` | `ollama` | `openai`) with graceful fallback (Vietnamese LLM API → Groq; Ollama main → Groq). Token usage is accumulated in-process via a `usage_accumulator` contextvar for cost metrics — no external tracing service required.

### 9. Comprehensive RAG Evaluation Suite
A 4-pillar framework tracking operational metrics (token count, API cost, latency TTFT/TTLT), quality metrics (LLM-as-judge faithfulness/relevance), agentic metrics (tool-call success via the `agent_tool_calls` contextvar, router accuracy), and failure-mode analysis (Retrieval / Routing / Hallucination / Execution).

> **Observability note (tracing):** Tracing is **self-hosted** by design: every graph run is persisted as a `GraphRun` + `AgentStep` rows in MySQL and published to a Redis pub/sub channel (`graph_trace_events`) for live SSE streaming. Tool-call tracking and token-usage accumulation are in-process contextvars feeding the eval suite. **No LangSmith / Langfuse / OpenTelemetry, no cloud egress** — Vietnamese legal data stays local. If external tracing is ever desired, add a LangSmith/Langfuse callback; it is off by default and the self-hosted trace keeps working independently.

---

## V. Technology Stack

*   **Frontend UI:** Streamlit (Python)
*   **Backend API Framework:** FastAPI
*   **Background Tasks & Queue:** Celery + Redis
*   **Vector Database:** Qdrant DB
*   **Relational Database:** PostgreSQL / MySQL (for logs, chat history, and status checks)
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
```

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

See [`docs/TESTING.md`](docs/TESTING.md) for full test suite coverage (including security layer, checkpointing, trace tables, CRAG flows, SSE streaming, and memory leakage tests).

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