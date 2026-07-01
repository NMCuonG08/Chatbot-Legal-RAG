import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from celery.result import AsyncResult
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sse_starlette.sse import EventSourceResponse

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from models import (
    delete_user_conversations,
    ensure_database_schema,
    insert_document,
    list_documents,
    list_user_conversations,
)
from cache import clear_conversation_id
from security import (
    get_cors_origins,
    get_legal_collection_name,
    require_api_key,
    resolve_safe_data_path,
)
from tasks import index_document_v2, llm_handle_message
from database import settings as db_settings
from utils import setup_logging
from vectorize import create_collection, list_collection_points, list_collections
from semantic_cache import init_semantic_cache

# Constants
TASK_TIMEOUT = 60
POLLING_INTERVAL = 0.5

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ensure_database_schema()
    init_semantic_cache()
    try:
        from vectorize import create_collection, list_collections
        existing = [c["name"] for c in list_collections()]
        if "user_episodes" not in existing:
            create_collection("user_episodes")
            logger.info("Created Qdrant collection 'user_episodes' for Episodic Memory.")
    except Exception as e:
        logger.warning(f"Could not initialize user_episodes collection on startup: {e}")
    yield
    # Shutdown: no explicit cleanup needed currently.


app = FastAPI(lifespan=lifespan)

# Expose Prometheus metrics (accessible at /metrics)
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# CORS allowlist: deny-all by default. Set CORS_ALLOWED_ORIGINS in .env.
_cors_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)



class CompleteRequest(BaseModel):
    bot_id: Optional[str] = "botLawyer"
    user_id: Optional[str] = "anonymous"
    user_message: str
    sync_request: Optional[bool] = False


class ImportRequest(BaseModel):
    data_file_path: Optional[str] = None
    collection_name: Optional[str] = Field(default="llm", pattern=r"^[a-z0-9_-]{1,64}$")
    batch_size: Optional[int] = 50
    limit: Optional[int] = None


class CreateCollectionRequest(BaseModel):
    collection_name: str = Field(..., pattern=r"^[a-z0-9_-]{1,64}$")


class CreateDocumentRequest(BaseModel):
    id: Optional[str] = None
    question: Optional[str] = None
    content: str = Field(..., min_length=1, max_length=200_000)


class PipelineIngestRequest(BaseModel):
    """Multi-source pipeline ingestion request.

    ``source_type`` selects a connector; ``path`` is a file (jsonl) or
    directory (markdown/html/pdf) resolved safely under the data dir.
    """

    source_type: str = Field(..., pattern=r"^(jsonl|markdown|html|pdf)$")
    path: Optional[str] = None
    collection_name: Optional[str] = Field(default="llm", pattern=r"^[a-z0-9_-]{1,64}$")
    limit: Optional[int] = None
    use_semantic: bool = True


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Vietnamese Legal Chatbot Backend"}


@app.get("/collections")
async def get_collections():
    return {"collections": list_collections()}


@app.get("/documents")
async def get_documents(limit: int = 100, offset: int = 0):
    return {
        "documents": list_documents(limit=limit, offset=offset),
        "limit": limit,
        "offset": offset,
    }


@app.get("/history/{user_id}")
async def get_user_history(user_id: str, limit: int = 100, offset: int = 0):
    return {
        "user_id": user_id,
        "history": list_user_conversations(user_id, limit=limit, offset=offset),
        "limit": limit,
        "offset": offset,
    }


@app.delete("/history/{user_id}")
async def delete_history(user_id: str, bot_id: str = "botLawyer"):
    """Clear all chat history for a user, including from cache"""
    logger.info(f"Deleting history for user {user_id} and bot {bot_id}")
    
    # Clear session from cache (Redis)
    cache_cleared = clear_conversation_id(bot_id, user_id)
    
    # Clear records from database
    db_cleared = delete_user_conversations(user_id)
    
    if not db_cleared:
        raise HTTPException(status_code=500, detail="Failed to delete history from database")
        
    return {
        "user_id": user_id,
        "status": "success",
        "cache_cleared": cache_cleared,
        "db_cleared": db_cleared
    }


@app.get("/collections/{collection_name}/points")
async def get_collection_points(
    collection_name: str,
    limit: int = 20,
    offset: int = 0,
    include_vectors: bool = False,
):
    return list_collection_points(
        collection_name,
        limit=limit,
        offset=offset,
        include_vectors=include_vectors,
    )


@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    logger.info(f"Complete chat from user {user_id} to {bot_id}: {user_message}")

    if not user_message or not user_id:
        raise HTTPException(
            status_code=400, detail="User id and user message are required"
        )

    if data.sync_request:
        # llm_handle_message is sync (Celery task called directly). Run in a
        # worker thread so the uvicorn event loop is NOT blocked for the whole
        # generation latency.
        response = await asyncio.to_thread(llm_handle_message, bot_id, user_id, user_message)
        return {
            "response": response.get("content", ""),
            "sources": response.get("sources", [])
        }
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}


@app.get("/chat/complete/{task_id}")
async def get_response(task_id: str):
    task_result = AsyncResult(task_id)
    task_status = task_result.status
    logger.info(f"Polling task {task_id} status: {task_status}")
    
    result = {
        "task_id": task_id,
        "task_status": task_status,
        "task_result": task_result.result if task_status != "PENDING" else None
    }
    return result


async def _resolve_run_id(task_id: str, timeout: float = 15.0) -> Optional[str]:
    """Resolve the trace run_id for a Celery task_id from Redis (set by the worker).

    The worker sets `trace:run:{task_id}` at run start. Short-poll until it
    appears (the SSE client may connect before the worker has started the run).
    """
    client = aioredis.from_url(db_settings.trace_redis_url, decode_responses=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run_id = await client.get(f"trace:run:{task_id}")
            if run_id:
                return run_id
            await asyncio.sleep(0.2)
        return None
    finally:
        await client.aclose()


@app.get("/chat/stream/{task_id}")
async def chat_stream(task_id: str):
    """Server-Sent Events stream of agent trace for an async chat task.

    Subscribes to the self-hosted Redis pub/sub trace channel, filters events
    by the run_id bound to this task_id, and closes the stream on `run_end`.
    Contract: the existing `/chat/complete` + `/chat` endpoints are unchanged.
    """
    run_id = await _resolve_run_id(task_id)
    if not run_id:
        raise HTTPException(status_code=404, detail=f"No trace run found for task {task_id}")

    channel = db_settings.trace_redis_channel

    async def event_generator():
        client = aioredis.from_url(db_settings.trace_redis_url, decode_responses=True)
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        try:
            # Send a ready frame so the client knows the stream is live.
            yield {"event": "ready", "data": json.dumps({"run_id": run_id})}
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if message is None:
                    continue
                if message.get("type") != "message":
                    continue
                try:
                    payload = json.loads(message.get("data", ""))
                except (json.JSONDecodeError, TypeError):
                    continue
                if payload.get("run_id") != run_id:
                    continue
                yield {"event": payload.get("event_type", "step"), "data": json.dumps(payload, ensure_ascii=False)}
                if payload.get("event_type") == "run_end":
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await client.aclose()

    return EventSourceResponse(event_generator())


@app.post("/collection/create", dependencies=[Depends(require_api_key)])
async def create_vector_collection(data: CreateCollectionRequest):
    collection_name = get_legal_collection_name(data.collection_name)
    create_status = create_collection(collection_name)
    logging.info(f"Create collection {collection_name} status: {create_status}")
    return {"status": create_status is not None}



@app.delete("/collections/{collection_name}/clean", dependencies=[Depends(require_api_key)])
async def clean_collection(collection_name: str):
    """Wipe all vector points in Qdrant collection and clean chunk metadata mapping in MySQL."""
    collection_name = get_legal_collection_name(collection_name)
    from vectorize import wipe_collection
    from models import _new_db_session, DocumentChunk

    # 1. Recreate the collection (which wipes all vectors)
    success = wipe_collection(collection_name)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to wipe collection {collection_name}")

    # 2. Clean MySQL mapping chunks
    db = _new_db_session()
    try:
        db.execute(delete(DocumentChunk))
        db.commit()
    except Exception as e:
        logger.error(f"Failed to clean document chunks metadata in database: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to clean database metadata")
    finally:
        db.close()

    return {"status": "success", "message": f"Collection {collection_name} and all mapping metadata cleaned successfully."}


@app.post("/document/create", dependencies=[Depends(require_api_key)])
async def create_document(data: CreateDocumentRequest):
    create_status = insert_document(data.question, data.content)
    logging.info(f"Create document status: {create_status}")
    # index_document_v2 runs embedding + Qdrant + DB sync; offload to thread
    # so the event loop is not held for seconds.
    index_status = await asyncio.to_thread(
        index_document_v2, data.id, data.question, data.content
    )
    return {"status": create_status is not None, "index_status": index_status}


@app.post("/data/import", dependencies=[Depends(require_api_key)])
async def import_qa_data_endpoint(data: ImportRequest):
    from import_data import import_qa_data

    # Resolve path safely (prevents path traversal).
    safe_path = resolve_safe_data_path(data.data_file_path)

    # import_qa_data is a long-running sync ingestion; offload to thread.
    success = await asyncio.to_thread(
        import_qa_data,
        str(safe_path),
        data.collection_name,
        data.batch_size or 50,
        data.limit,
    )
    return {"success": success}


@app.post("/pipeline/ingest", dependencies=[Depends(require_api_key)])
async def pipeline_ingest_endpoint(data: PipelineIngestRequest):
    """Run the multi-source data pipeline (fetch → parse → chunk → embed).

    Picks a connector by ``source_type`` and processes every doc through the
    shared core. Idempotent: already-embedded docs are skipped.
    """
    from pipeline.connectors import (
        HtmlConnector,
        JsonlQaConnector,
        MarkdownConnector,
        PdfConnector,
    )
    from pipeline.orchestrator import run_pipeline

    collection_name = get_legal_collection_name(data.collection_name)

    # Resolve path safely under the data dir (prevents path traversal).
    # For jsonl ``path`` is a file; for the others it is a directory.
    safe_path = resolve_safe_data_path(data.path) if data.path else None

    if data.source_type == "jsonl":
        if safe_path is None or not safe_path.is_file():
            raise HTTPException(status_code=400, detail="jsonl source requires a file path")
        connector = JsonlQaConnector(safe_path, limit=data.limit)
    elif data.source_type == "markdown":
        if safe_path is None or not safe_path.is_dir():
            raise HTTPException(status_code=400, detail="markdown source requires a directory path")
        connector = MarkdownConnector(safe_path, limit=data.limit)
    elif data.source_type == "html":
        if safe_path is None or not safe_path.is_dir():
            raise HTTPException(status_code=400, detail="html source requires a directory path")
        connector = HtmlConnector(root_dir=safe_path, limit=data.limit)
    elif data.source_type == "pdf":
        if safe_path is None or not safe_path.is_dir():
            raise HTTPException(status_code=400, detail="pdf source requires a directory path")
        connector = PdfConnector(safe_path, limit=data.limit)
    else:  # unreachable: pattern-validated above
        raise HTTPException(status_code=400, detail=f"unknown source_type: {data.source_type}")

    stats = await asyncio.to_thread(
        run_pipeline,
        [connector],
        collection_name,
        data.use_semantic,
        data.limit,
    )
    return {"collection": collection_name, "source_type": data.source_type, "stats": stats}


if __name__ == "__main__":
    import os
    import uvicorn

    # On Windows, multi-worker local run can hide startup exceptions behind
    # repeated "Child process died" messages.
    workers = 1 if os.name == "nt" else 2
    uvicorn.run("app:app", host="0.0.0.0", port=8002, workers=workers, log_level="info")