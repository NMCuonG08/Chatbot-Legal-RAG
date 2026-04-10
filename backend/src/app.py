import logging
import time
from dataclasses import Field
from pathlib import Path
from typing import Dict, Optional

from celery.result import AsyncResult
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from models import (
    delete_user_conversations,
    ensure_database_schema,
    insert_document,
    list_documents,
    list_user_conversations,
)
from cache import clear_conversation_id
from tasks import index_document_v2, llm_handle_message
from utils import setup_logging
from vectorize import create_collection, list_collection_points, list_collections

# Constants
TASK_TIMEOUT = 60
POLLING_INTERVAL = 0.5

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    ensure_database_schema()


class CompleteRequest(BaseModel):
    bot_id: Optional[str] = "botLawyer"
    user_id: Optional[str] = "anonymous"
    user_message: str
    sync_request: Optional[bool] = False


class ImportRequest(BaseModel):
    data_file_path: Optional[str] = None
    collection_name: Optional[str] = None
    batch_size: Optional[int] = 50
    limit: Optional[int] = None


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
        response = llm_handle_message(bot_id, user_id, user_message)
        return {"response": str(response)}
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}


@app.get("/chat/complete/{task_id}")
async def get_response(task_id: str):
    start_time = time.time()
    while True:
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        logger.info(f"Task result: {task_result.result}")

        if task_status == "PENDING":
            if time.time() - start_time > TASK_TIMEOUT:
                return {
                    "task_id": task_id,
                    "task_status": task_result.status,
                    "task_result": task_result.result,
                    "error_message": "Service timeout, retry please",
                }
            else:
                time.sleep(POLLING_INTERVAL)  # sleep for 0.5 seconds before retrying
        else:
            result = {
                "task_id": task_id,
                "task_status": task_result.status,
                "task_result": task_result.result,
            }
            return result


@app.post("/collection/create")
async def create_vector_collection(data: Dict):
    collection_name = data.get("collection_name")
    create_status = create_collection(collection_name)
    logging.info(f"Create collection {collection_name} status: {create_status}")
    return {"status": create_status is not None}


@app.post("/document/create")
async def create_document(data: Dict):
    doc_id = data.get("id")
    question = data.get("question")
    content = data.get("content")
    create_status = insert_document(question, content)
    logging.info(f"Create document status: {create_status}")
    index_status = index_document_v2(doc_id, question, content)
    return {"status": create_status is not None, "index_status": index_status}


@app.post("/data/import")
async def import_qa_data_endpoint(data: ImportRequest):
    from import_data import import_qa_data

    success = import_qa_data(
        data_file_path=data.data_file_path,
        collection_name=data.collection_name,
        batch_size=data.batch_size or 50,
        limit=data.limit,
    )
    return {"success": success}


if __name__ == "__main__":
    import os
    import uvicorn

    # On Windows, multi-worker local run can hide startup exceptions behind
    # repeated "Child process died" messages.
    workers = 1 if os.name == "nt" else 2
    uvicorn.run("app:app", host="0.0.0.0", port=8002, workers=workers, log_level="info")