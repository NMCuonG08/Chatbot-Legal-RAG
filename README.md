# Legal RAG & Agentic Workflow (MVP Scaffold)

This project is a base scaffold for your architecture:

- Frontend: Streamlit
- Backend: FastAPI
- Async: Celery + Redis
- Data layer placeholders: MariaDB, Qdrant
- Router: Legal RAG / React Agent / Web Search

## 1. Run with Docker

```bash
docker compose up --build
```

Services:

- Backend API: <http://localhost:8000>
- Frontend UI: <http://localhost:8501>
- Qdrant: <http://localhost:6333>

## 2. API Quick Test

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"session_id\":\"s1\",\"message\":\"soan hop dong lao dong\"}"
```

Then poll task result:

```bash
curl http://localhost:8000/tasks/<task_id>
```

## 3. Current status

Implemented:

- Async request flow (FastAPI -> Celery -> Redis)
- Router with 3 paths
- Placeholder logic for Legal RAG, ReAct Agent, Web Search
- Streamlit chat UI with polling

Not yet implemented (next steps):

- LlamaIndex + Qdrant hybrid retrieval (BM25 + dense)
- Cohere reranking
- Tavily real integration
- MariaDB chat history persistence
- Auth, observability, CI/CD

## 4. Suggested next module to build

1. Hybrid Search module with LlamaIndex + Qdrant.
2. DB persistence layer for chat history.
3. Streaming response path (SSE/WebSocket).
