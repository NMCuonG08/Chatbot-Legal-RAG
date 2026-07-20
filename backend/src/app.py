import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from celery.result import AsyncResult
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sse_starlette.sse import EventSourceResponse

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from models import (
    create_user,
    delete_user_conversations,
    delete_user_episodes,
    ensure_database_schema,
    get_user_by_id,
    get_user_by_username,
    insert_document,
    list_documents,
    list_user_conversations,
    load_agent_steps,
)
from cache import clear_conversation_id
from guardrails_manager import LegalGuardrailsManager
from security import (
    get_cors_origins,
    get_legal_collection_name,
    require_api_key,
    resolve_safe_data_path,
)
# Phase 2 — auth / RBAC / audit / approval / seed
from auth import (
    auth_configured,
    create_access_token,
    hash_password,
    verify_password,
)
from audit import list_audit_entries, log_audit
from approval import decide_approval, fetch_pending, get_approval
from rbac import (
    Principal,
    Role,
    get_current_user,
    get_current_user_optional,
    require_admin,
)
from seed_admin import seed_admin
from tasks import index_document_v2, llm_handle_message, clear_user_runtime_caches_task
from agent import clear_user_runtime_caches
from database import settings as db_settings
from utils import setup_logging
from vectorize import create_collection, list_collection_points, list_collections, delete_vectors_by_filter
from semantic_cache import (
    clear_semantic_cache,
    get_cache_stats,
    init_semantic_cache,
    maybe_wipe_legacy_cache,
    SCOPE_USER_PREFIX,
)
from rlhf_store import init_rlhf_collection, save_feedback

# Constants
TASK_TIMEOUT = 60
POLLING_INTERVAL = 0.5

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import time
    logger.info("🚀 Starting sequential backend dependency checks...")

    # 1. Check SQL DB
    try:
        from database import engine
        # Test connection
        conn = engine.connect()
        conn.close()
        logger.info("🟢 Database (SQL) connection verified.")
    except Exception as e:
        logger.critical(f"🔴 Database (SQL) check failed: {e}")
        raise RuntimeError(f"Database check failed: {e}")

    # 2. Check Redis
    try:
        import redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)
        r.ping()
        logger.info("🟢 Redis Cache connection verified.")
    except Exception as e:
        logger.critical(f"🔴 Redis Cache check failed: {e}")
        raise RuntimeError(f"Redis check failed: {e}")

    # 3. Check Qdrant (with retries up to 30 seconds to allow for startup recovery)
    try:
        from vectorize import get_client
        qdrant_ready = False
        logger.info("⏳ Waiting for Qdrant DB to become ready...")
        for attempt in range(30):
            try:
                get_client().get_collections()
                logger.info("🟢 Qdrant DB connection verified.")
                qdrant_ready = True
                break
            except Exception as exc:
                if attempt < 29:
                    logger.warning(f"⚠️ Qdrant not ready yet (attempt {attempt + 1}/30): {exc}. Retrying in 1s...")
                    time.sleep(1)
                else:
                    raise exc
        if not qdrant_ready:
            raise RuntimeError("Qdrant check failed after 30 attempts.")
    except Exception as e:
        logger.critical(f"🔴 Qdrant DB check failed: {e}")
        raise RuntimeError(f"Qdrant check failed: {e}")

    # 4. Check Celery Worker (with retries up to 15 seconds to give dev.py time to spin it up)
    try:
        from tasks import celery_app
        worker_ready = False
        logger.info("⏳ Waiting for Celery worker to become ready...")
        for attempt in range(15):
            try:
                inspect = celery_app.control.inspect()
                ping_res = inspect.ping() if inspect else None
                if ping_res:
                    logger.info(f"🟢 Celery Worker(s) found active: {list(ping_res.keys())}")
                    worker_ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not worker_ready:
            logger.critical("🔴 Celery Worker check failed: No active workers found.")
            raise RuntimeError("Celery Worker check failed: No active workers found.")
    except Exception as e:
        logger.critical(f"🔴 Celery Worker check failed: {e}")
        raise RuntimeError(f"Celery Worker check failed: {e}")

    # 5. Run DB Schema Migration
    logger.info("⚙️ Ensuring Database Schema...")
    ensure_database_schema()

    # 6. Seed Admin
    try:
        seed_admin()
    except Exception as e:
        logger.warning(f"Seed admin failed (non-fatal): {e}")

    # 7. Init Semantic Cache
    logger.info("⚙️ Initializing Semantic Cache...")
    init_semantic_cache()

    # 8. Wipe legacy cache if requested
    try:
        wiped = maybe_wipe_legacy_cache()
        if wiped:
            logger.info(f"Legacy cache wipe deleted {wiped} points.")
    except Exception as e:
        logger.warning(f"Legacy cache wipe failed (non-fatal): {e}")

    # 9. Create Qdrant Collections
    try:
        from vectorize import create_collection, list_collections
        existing = [c["name"] for c in list_collections()]
        if "user_episodes" not in existing:
            create_collection("user_episodes")
            logger.info("Created Qdrant collection 'user_episodes' for Episodic Memory.")
    except Exception as e:
        logger.warning(f"Could not initialize user_episodes collection on startup: {e}")

    # 10. Init RLHF collection
    try:
        init_rlhf_collection()
    except Exception as e:
        logger.warning(f"Could not initialize rlhf_good_answers collection on startup: {e}")

    # 11. Pre-load local SentenceTransformer model
    try:
        from custom_embedding import get_embedding_service
        logger.info("[FASTAPI] Pre-loading local SentenceTransformer model on startup...")
        get_embedding_service()._get_sentence_transformer()
        logger.info("[FASTAPI] ✅ Local SentenceTransformer model pre-loaded successfully.")
    except Exception as e:
        logger.warning(f"[FASTAPI] Could not pre-load SentenceTransformer model: {e}")

    logger.info("✅ All sequential startup backend dependency checks completed successfully. Launching FastAPI backend...")
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
    # P6 canary/shadow knobs. variant = explicit model override; shadow = also
    # run candidate + persist for offline compare (user still gets primary).
    variant: Optional[str] = None
    shadow: Optional[bool] = False


# Phase 2 — auth / approval request schemas
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=6, max_length=128)
    email: Optional[str] = None
    role: Optional[str] = Field(default=Role.USER, pattern=r"^(admin|lawyer|user|guest)$")


class LoginRequest(BaseModel):
    username: str
    password: str


class ApprovalDecideRequest(BaseModel):
    decision: str = Field(..., pattern=r"^(approved|rejected)$")
    note: Optional[str] = None


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


class FeedbackRequest(BaseModel):
    """RLHF 👍/👎 on an assistant message (Phase 4).

    ``user_id`` is required and must be a real per-user id — sentinel/empty
    ids are rejected by ``rlhf_store.save_feedback`` (no cross-user feedback
    leak). ``question``/``response``/``sources`` are echoed back by the client
    from the message being rated so the store can reuse them as few-shot /
    rerank signal without re-fetching.
    """
    user_id: str
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    rating: str = Field(..., pattern=r"^(good|bad)$")
    question: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    sources: Optional[List[Dict[str, Any]]] = []


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Vietnamese Legal Chatbot Backend"}


@app.get("/health/detailed")
def health_detailed():
    # 1. Database check
    db_status = "healthy"
    db_error = None
    try:
        from sqlalchemy import text
        from database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = "unhealthy"
        db_error = str(e)

    # 2. Redis check
    redis_status = "healthy"
    redis_error = None
    try:
        import redis
        from database import settings as db_settings
        r = redis.from_url(db_settings.redis_url)
        r.ping()
    except Exception as e:
        redis_status = "unhealthy"
        redis_error = str(e)

    # 3. Qdrant check
    qdrant_status = "healthy"
    qdrant_error = None
    try:
        from vectorize import get_client
        qclient = get_client()
        qclient.get_collections()
    except Exception as e:
        qdrant_status = "unhealthy"
        qdrant_error = str(e)

    # 4. Celery check
    celery_status = "healthy"
    celery_error = None
    active_workers = []
    try:
        from tasks import celery_app
        ping_result = celery_app.control.ping(timeout=0.3)
        if ping_result:
            for worker_res in ping_result:
                active_workers.extend(worker_res.keys())
        else:
            celery_status = "no_workers"
            celery_error = "No active Celery workers found"
    except Exception as e:
        celery_status = "unhealthy"
        celery_error = str(e)

    # 5. Ollama check (if configured)
    ollama_status = "healthy"
    ollama_error = None
    ollama_configured = False
    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    use_ollama = os.getenv("USE_OLLAMA_AS_MAIN", "false").lower() == "true"
    if llm_provider == "ollama" or use_ollama:
        ollama_configured = True
        ollama_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        try:
            import requests
            resp = requests.get(f"{ollama_url}/api/tags", timeout=2)
            if resp.status_code != 200:
                ollama_status = "unhealthy"
                ollama_error = f"Ollama returned status {resp.status_code}"
        except Exception as e:
            ollama_status = "unhealthy"
            ollama_error = str(e)
    else:
        ollama_status = "not_configured"

    return {
        "status": "healthy" if all(s in ["healthy", "not_configured"] for s in [db_status, redis_status, qdrant_status, celery_status, ollama_status]) else "unhealthy",
        "database": {"status": db_status, "error": db_error},
        "redis": {"status": redis_status, "error": redis_error},
        "qdrant": {"status": qdrant_status, "error": qdrant_error},
        "celery": {"status": celery_status, "active_workers": active_workers, "error": celery_error},
        "ollama": {"status": ollama_status, "configured": ollama_configured, "error": ollama_error}
    }



# ---- Phase 2: auth / RBAC / approval / audit endpoints ----
def _client_ip(request: "Request") -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


@app.post("/auth/register")
async def register(req: RegisterRequest, request: "Request"):
    """Register a new user. Self-registration is restricted to non-escalating
    roles: a caller may NOT self-assign admin/lawyer via this public endpoint
    — only user/guest. Existing-role seeding is done via ``seed_admin`` at
    startup or by an admin out-of-band."""
    if req.role in (Role.ADMIN, Role.LAWYER):
        raise HTTPException(
            status_code=403,
            detail="Không thể tự đăng ký vai trò admin/lawyer. Liên hệ quản trị viên.",
        )
    if not auth_configured():
        raise HTTPException(
            status_code=500,
            detail="JWT chưa được cấu hình (JWT_SECRET). Thiết lập trước khi dùng auth.",
        )
    if get_user_by_username(req.username) is not None:
        raise HTTPException(status_code=409, detail="Tên đăng nhập đã tồn tại.")
    try:
        user = create_user(req.username, hash_password(req.password), role=req.role)
    except Exception as exc:  # IntegrityError on concurrent duplicate insert
        logger.warning(f"register duplicate/race: {exc}")
        raise HTTPException(status_code=409, detail="Tên đăng nhập đã tồn tại.")
    log_audit(
        user_id=user.id,
        action="register",
        resource="user",
        ip=_client_ip(request),
        payload={"username": req.username, "role": req.role},
    )
    token = create_access_token(
        subject=user.id, claims={"username": user.username, "role": user.role}
    )
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "username": user.username, "role": user.role}}


@app.post("/auth/login")
async def login(req: LoginRequest, request: "Request"):
    """Exchange username/password for a JWT."""
    if not auth_configured():
        raise HTTPException(status_code=500, detail="JWT chưa được cấu hình (JWT_SECRET).")
    user = get_user_by_username(req.username)
    if user is None or not verify_password(req.password, user.password_hash):
        # Same message for both branches to avoid username enumeration.
        log_audit(
            user_id=None,
            action="login_failed",
            resource="user",
            ip=_client_ip(request),
            payload={"username": req.username},
        )
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không đúng.")
    token = create_access_token(
        subject=user.id, claims={"username": user.username, "role": user.role}
    )
    log_audit(
        user_id=user.id,
        action="login",
        resource="user",
        ip=_client_ip(request),
        payload={"username": user.username},
    )
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "username": user.username, "role": user.role}}


@app.get("/auth/me")
async def me(principal: Principal = Depends(get_current_user)):
    return {"id": principal.user_id, "username": principal.username, "role": principal.role}


@app.get("/approvals", dependencies=[Depends(require_admin)])
async def list_approvals(limit: int = 100):
    """List pending tool-approval requests (admin only)."""
    return {"pending": fetch_pending(limit=limit)}


@app.post("/approvals/{approval_id}/decide", dependencies=[Depends(require_admin)])
async def approval_decide(
    approval_id: str,
    req: ApprovalDecideRequest,
    principal: Principal = Depends(get_current_user),
    request: Request = None,
):
    """Approve or reject a pending tool-approval request (admin only)."""
    existing = get_approval(approval_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy yêu cầu phê duyệt.")
    updated = decide_approval(approval_id, req.decision, decided_by=principal.user_id, note=req.note)
    log_audit(
        user_id=principal.user_id,
        action=f"approval_{req.decision}",
        resource="tool_approval",
        ip=_client_ip(request) if request else None,
        payload={"approval_id": approval_id, "tool_name": existing.tool_name},
    )
    return {"id": updated.id, "status": updated.status, "decided_by": updated.decided_by}


@app.get("/audit", dependencies=[Depends(require_admin)])
async def audit_entries(limit: int = 200, offset: int = 0, user_id: Optional[str] = None, action: Optional[str] = None):
    """Read the audit trail (admin only)."""
    return {"entries": list_audit_entries(limit=limit, offset=offset, user_id=user_id, action=action)}


@app.post("/feedback")
async def feedback(data: FeedbackRequest):
    """RLHF 👍/👎 endpoint (Phase 4). Stores feedback user-scoped.

    Sentinel/empty ``user_id`` is rejected (HTTP 400) — feedback requires a
    real per-user scope so a user's 👍 never leaks into another user's
    few-shot / rerank signal. Invalid rating is rejected by the pydantic
    pattern. All downstream persistence failures are swallowed inside
    ``save_feedback`` (returns 200 with a status string) so the frontend
    click never errors the UX.
    """
    status = save_feedback(
        user_id=data.user_id,
        conversation_id=data.conversation_id or "",
        message_id=data.message_id or "",
        question=data.question,
        response=data.response,
        sources=data.sources or [],
        rating=data.rating,
    )
    if status == "rejected_sentinel":
        raise HTTPException(status_code=400, detail="login required to submit feedback")
    return {"ok": True, "status": status}


@app.get("/stats")
async def stats():
    """Lightweight observability snapshot for ops / alerter scrape.

    Exposes in-process counters: semantic-cache hit/miss/error rates +
    error-skip (poisoning-prevention) counts, and router route distribution.
    Counters reset on process restart. Intended for an external alerter to
    detect cache-down (error spike) or hit-rate collapse, not a full metrics
    backend (Prometheus /metrics already covers HTTP-level signals).
    """
    try:
        cache = get_cache_stats()
    except Exception as e:
        logger.warning(f"/stats cache stats failed: {e}")
        cache = {"error": str(e)}
    try:
        from brain import get_route_stats

        routes = get_route_stats()
    except Exception as e:
        logger.warning(f"/stats route stats failed: {e}")
        routes = {"error": str(e)}
    return {"cache": cache, "routes": routes}


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


@app.get("/sessions")
def get_sessions(limit: int = 100):
    from models import list_unique_session_ids
    return {"sessions": list_unique_session_ids(limit=limit)}


@app.get("/history/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
    principal: Optional[Principal] = Depends(get_current_user_optional),
):
    # Phase 2 — a caller may only read their own history; admin may read any.
    if principal is None:
        allow_anon = os.getenv("ALLOW_ANONYMOUS", "1") == "1"
        if not allow_anon:
            raise HTTPException(status_code=401, detail="Đăng nhập (JWT) là bắt buộc.")
    elif principal.user_id != user_id and not principal.is_admin:
        raise HTTPException(status_code=403, detail="Không có quyền xem lịch sử của user khác.")
    return {
        "user_id": user_id,
        "history": list_user_conversations(user_id, limit=limit, offset=offset),
        "limit": limit,
        "offset": offset,
    }


@app.delete("/history")
async def delete_all_history(principal: Optional[Principal] = Depends(get_current_user_optional)):
    from models import delete_all_conversations
    db_cleared = delete_all_conversations()
    return {"status": "success", "message": "All conversations deleted", "db_cleared": db_cleared}


@app.delete("/history/message/{message_id}")
def delete_single_message(message_id: int):
    from models import _new_db_session, ChatConversation
    ensure_database_schema()
    db = _new_db_session()
    try:
        db.execute(
            delete(ChatConversation).where(ChatConversation.id == message_id)
        )
        db.commit()
        return {"status": "success", "message": f"Message {message_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.delete("/history/{user_id}")
async def delete_history(
    user_id: str,
    bot_id: str = "botLawyer",
    principal: Optional[Principal] = Depends(get_current_user_optional),
):
    """Clear all chat history for a user, including from cache.

    Purges, per user:
      - Redis conversation-id mapping (clear_conversation_id)
      - DB conversation rows (delete_user_conversations)
      - DB episodic facts (delete_user_episodes)
      - Qdrant semantic-cache entries scoped to this user
      - Qdrant user_episodes (long-term episodic facts)
      - In-process rolling-summary + memory buffers (this process + worker)
    Without the Qdrant + episodic clears the agent would keep recalling old
    facts/answers even after the user wiped their history.
    """
    # Sentinel guard: a shared placeholder id maps EVERY such client to one
    # bucket, so /history/anonymous would wipe OTHER anonymous users' facts.
    # Reject so a wipe can only touch a real per-browser identity.
    if (user_id or "").strip() in _SHARED_USER_SENTINELS:
        raise HTTPException(
            status_code=400,
            detail="Không thể xóa lịch sử cho user_id dùng chung (anonymous/demo-session). "
            "Vui lòng dùng tên người dùng riêng.",
        )

    # Phase 2 — a caller may only wipe their own history; admin may wipe any.
    if principal is None:
        if os.getenv("ALLOW_ANONYMOUS", "1") != "1":
            raise HTTPException(status_code=401, detail="Đăng nhập (JWT) là bắt buộc.")
    elif principal.user_id != user_id and not principal.is_admin:
        raise HTTPException(status_code=403, detail="Không có quyền xóa lịch sử của user khác.")

    logger.info(f"Deleting history for user {user_id} and bot {bot_id}")

    # Clear session from cache (Redis conversation-id mapping)
    cache_cleared = clear_conversation_id(bot_id, user_id)

    # Clear records from database
    db_cleared = delete_user_conversations(user_id)

    # Clear long-term episodic facts from MySQL (mirrors the Qdrant purge below
    # — without this, facts survive in the DB and can resurface later).
    db_episodes_cleared = delete_user_episodes(user_id)

    # Clear per-user semantic cache (Qdrant). Best-effort: a Qdrant outage
    # must not block the rest of the wipe.
    semantic_cleared = False
    try:
        semantic_cleared = delete_vectors_by_filter(
            "semantic_cache", {"scope": f"{SCOPE_USER_PREFIX}{user_id}"}
        )
    except Exception as e:
        logger.warning(f"[DELETE] semantic_cache clear failed for user={user_id}: {e}")

    # Clear long-term episodic facts (Qdrant user_episodes). These are the
    # "agent remembers your name / situation" entries — the most common
    # reason a deleted conversation still seems remembered.
    episodic_cleared = False
    try:
        episodic_cleared = delete_vectors_by_filter("user_episodes", {"user_id": user_id})
    except Exception as e:
        logger.warning(f"[DELETE] user_episodes clear failed for user={user_id}: {e}")

    # In-process rolling-summary + ChatMemoryBuffer. Clear locally (covers the
    # sync path where the agent runs in this process) AND dispatch a Celery
    # task so the worker's own copies are purged too.
    runtime_cleared = 0
    try:
        runtime_cleared = clear_user_runtime_caches(user_id)
    except Exception as e:
        logger.warning(f"[DELETE] local runtime cache clear failed for user={user_id}: {e}")
    try:
        clear_user_runtime_caches_task.delay(user_id)
    except Exception as e:
        logger.warning(f"[DELETE] worker runtime cache clear dispatch failed for user={user_id}: {e}")

    if not db_cleared:
        raise HTTPException(status_code=500, detail="Failed to delete history from database")

    return {
        "user_id": user_id,
        "status": "success",
        "cache_cleared": cache_cleared,
        "db_cleared": db_cleared,
        "db_episodes_cleared": db_episodes_cleared,
        "semantic_cache_cleared": semantic_cleared,
        "episodic_cleared": episodic_cleared,
        "runtime_caches_cleared": runtime_cleared,
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


_SHARED_USER_SENTINELS = {"anonymous", "demo-session", ""}


@app.post("/chat/complete")
async def complete(
    data: CompleteRequest,
    principal: Optional[Principal] = Depends(get_current_user_optional),
    request: Request = None,
):
    bot_id = data.bot_id
    # Phase 2 — trust the JWT subject over the client-supplied user_id so a
    # user cannot impersonate another by editing the payload. Fall back to the
    # payload id only for the legacy anonymous demo path (ALLOW_ANONYMOUS=1).
    if principal is not None:
        user_id = principal.user_id
    else:
        allow_anon = os.getenv("ALLOW_ANONYMOUS", "1") == "1"
        user_id = data.user_id
        if not allow_anon:
            raise HTTPException(
                status_code=401,
                detail="Đăng nhập (JWT) là bắt buộc. ALLOW_ANONYMOUS=0 đang bật.",
            )
    user_message = data.user_message
    logger.info(f"Complete chat from user {user_id} to {bot_id}: {user_message}")

    if not user_message or not user_id:
        raise HTTPException(
            status_code=400, detail="User id and user message are required"
        )

    # Sentinel guard: if the client sent a shared placeholder id, every such
    # client collapses into ONE conversation_id / memory buffer / episodic
    # store, leaking facts (e.g. names) across users. We do not reject (to
    # keep the demo working) but log loudly so ops can see the leak risk.
    if (user_id or "").strip() in _SHARED_USER_SENTINELS:
        logger.warning(
            f"[SESSION] Shared sentinel user_id={user_id!r} detected — "
            "conversation/memory/episodic state is shared across all such "
            "clients. Frontend must send a per-browser UUID."
        )

    # Fast-reject gate: deterministic Tier-1 keyword guardrails (pure string
    # match, no LLM, no NeMo init). Blocks jailbreak/political/toxic inputs
    # before any broker dispatch or DB write. Tier-2 semantic guardrails still
    # run inside the worker for non-obvious cases.
    blocked = LegalGuardrailsManager.verify_input_tier1(user_message)
    if blocked:
        return {"response": blocked, "sources": []}

    log_audit(
        user_id=user_id,
        action="chat",
        resource="message",
        ip=_client_ip(request) if request else None,
        payload={"bot_id": bot_id, "role": principal.role if principal else "anonymous"},
    )

    # Phase 2b — thread the caller's RBAC role into the graph so tool-policy
    # filtering + the approval gate apply. Anonymous (no JWT) -> None = legacy.
    role = principal.role if principal else None

    if data.sync_request:
        # llm_handle_message is sync (Celery task called directly). Run in a
        # worker thread so the uvicorn event loop is NOT blocked for the whole
        # generation latency.
        response = await asyncio.to_thread(llm_handle_message, bot_id, user_id, user_message, role, data.variant, data.shadow)
        return {
            "response": response.get("content", ""),
            "sources": response.get("sources", []),
            "route": response.get("route", ""),
            "tool_calls": response.get("tool_calls", [])
        }
    else:
        try:
            task = llm_handle_message.delay(bot_id, user_id, user_message, role, data.variant, data.shadow)
            return {"task_id": task.id}
        except Exception as broker_err:
            # Broker/broker-down: fall back to the in-process sync path so a
            # transient Redis/Celery outage does not hard-500 a chat that the
            # sync handler could still serve.
            logger.warning(f"[CHAT] broker dispatch failed, falling back to sync: {broker_err}")
            response = await asyncio.to_thread(llm_handle_message, bot_id, user_id, user_message, role, data.variant, data.shadow)
            return {
                "response": response.get("content", ""),
                "sources": response.get("sources", []),
                "route": response.get("route", ""),
                "tool_calls": response.get("tool_calls", [])
            }


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


def _make_immediate_run_end(status: str, cached: bool = False):
    async def event_generator_finished():
        yield {
            "event": "run_end",
            "data": json.dumps({
                "run_id": "completed_early",
                "node": "__ROOT__",
                "event_type": "run_end",
                "payload": {"status": status.lower(), "cached": cached, "completed_early": True}
            }, ensure_ascii=False, default=str),
        }
    return EventSourceResponse(event_generator_finished())


async def _resolve_run_id(task_id: str, timeout: float = 15.0) -> Optional[str]:
    """Resolve the trace run_id for a Celery task_id from Redis (set by the worker).

    The worker sets `trace:run:{task_id}` at run start. Short-poll until it
    appears (the SSE client may connect before the worker has started the run).
    """
    client = aioredis.from_url(db_settings.trace_redis_url, decode_responses=True)
    try:
        deadline = time.monotonic() + timeout
        task_result = AsyncResult(task_id)
        while time.monotonic() < deadline:
            run_id = await client.get(f"trace:run:{task_id}")
            if run_id:
                return run_id
            
            # If the task finishes or fails while we poll, stop waiting
            if task_result.status in ["SUCCESS", "FAILURE", "REVOKED"]:
                logger.info(f"Task {task_id} transitioned to {task_result.status} during short-poll")
                return None
                
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
    # 0. Check if the task is already completed (e.g., Semantic Cache HIT or Guardrail Block)
    # If completed, we do not need to wait for a run_id or subscribe to pub/sub
    task_result = AsyncResult(task_id)
    if task_result.status in ["SUCCESS", "FAILURE", "REVOKED"]:
        logger.info(f"⚡ Task {task_id} already completed early (status={task_result.status}). Returning immediate run_end.")
        return _make_immediate_run_end(task_result.status, cached=(task_result.status == "SUCCESS"))

    run_id = await _resolve_run_id(task_id)
    if not run_id:
        # Re-check task status in case it transitioned during poll
        if task_result.status in ["SUCCESS", "FAILURE", "REVOKED"]:
            logger.info(f"⚡ Task {task_id} completed during poll (status={task_result.status}). Returning immediate run_end.")
            return _make_immediate_run_end(task_result.status, cached=(task_result.status == "SUCCESS"))
        raise HTTPException(status_code=404, detail=f"No trace run found for task {task_id}")

    channel = db_settings.trace_redis_channel
    # Heartbeat cadence: must be shorter than the SSE client's read timeout
    # (the Streamlit UI uses requests with a 15s read timeout) so the
    # connection is never idle long enough to time out between trace events.
    heartbeat_interval = 5.0

    async def event_generator():
        # Send a ready frame so the client knows the stream is live.
        yield {"event": "ready", "data": json.dumps({"run_id": run_id})}

        # Subscribe FIRST, then replay persisted steps. Ordering matters:
        # events committed to MySQL before subscribe are delivered via replay,
        # events published after subscribe are delivered live, and dupes
        # (event in both DB snapshot and pubsub buffer) are skipped via
        # seen_indices. This closes the subscribe-after-complete race where
        # the UI opens the trace stream only after the task already finished.
        client = aioredis.from_url(db_settings.trace_redis_url, decode_responses=True)
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        seen_indices: set[int] = set()
        try:
            try:
                steps = await asyncio.to_thread(load_agent_steps, run_id)
            except Exception as e:
                logger.warning(f"trace replay load failed for {run_id}: {e}")
                steps = []
            for step in steps:
                seen_indices.add(step.step_index)
                evt = {
                    "run_id": run_id,
                    "node": step.node,
                    "step_index": step.step_index,
                    "event_type": step.event_type,
                    "payload": step.payload,
                }
                yield {
                    "event": step.event_type,
                    "data": json.dumps(evt, ensure_ascii=False, default=str),
                }
                if step.event_type == "run_end":
                    return

            # Live-tail pub/sub for a run still in progress (or events not
            # yet in the DB snapshot at replay time).
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=heartbeat_interval
                )
                if message is None:
                    # Double-check if the Celery task has finished to prevent
                    # get_message race condition hangs.
                    from celery.result import AsyncResult
                    task_result = AsyncResult(task_id)
                    if task_result.status in ["SUCCESS", "FAILURE", "REVOKED"]:
                        # Load any last-second events that might have been saved to DB
                        try:
                            final_steps = await asyncio.to_thread(load_agent_steps, run_id)
                            for step in final_steps:
                                if step.step_index not in seen_indices:
                                    seen_indices.add(step.step_index)
                                    yield {
                                        "event": step.event_type,
                                        "data": json.dumps({
                                            "run_id": run_id,
                                            "node": step.node,
                                            "step_index": step.step_index,
                                            "event_type": step.event_type,
                                            "payload": step.payload,
                                        }, ensure_ascii=False, default=str),
                                    }
                                    if step.event_type == "run_end":
                                        return
                        except Exception as e:
                            logger.warning(f"Final steps loading failed: {e}")
                        
                        # Fallback: if run_end was not recorded in DB, send it manually to close the stream
                        yield {
                            "event": "run_end",
                            "data": json.dumps({
                                "run_id": run_id,
                                "node": "__root__",
                                "event_type": "run_end",
                                "payload": {"status": "completed"}
                            }, ensure_ascii=False, default=str),
                        }
                        return

                    # Keep the connection alive between events so clients with
                    # aggressive read timeouts don't drop the stream.
                    yield {"event": "ping", "data": ""}
                    continue
                if message.get("type") != "message":
                    continue
                try:
                    payload = json.loads(message.get("data", ""))
                except (json.JSONDecodeError, TypeError):
                    continue
                if payload.get("run_id") != run_id:
                    continue
                idx = payload.get("step_index")
                if idx is not None and idx in seen_indices:
                    continue
                if idx is not None:
                    seen_indices.add(idx)
                yield {
                    "event": payload.get("event_type", "step"),
                    "data": json.dumps(payload, ensure_ascii=False, default=str),
                }
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

    # 1b. Invalidate semantic cache: cached answers may have been grounded in
    # the wiped vectors and would be stale after a re-index.
    cache_cleared = clear_semantic_cache()

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

    return {"status": "success", "message": f"Collection {collection_name} and all mapping metadata cleaned successfully.", "cache_cleared": cache_cleared}


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
    # Invalidate semantic cache: bulk (re-)ingest changes the document corpus
    # the cache answers are grounded in, so prior cached responses may be stale.
    if data.use_semantic:
        try:
            clear_semantic_cache()
        except Exception as cache_err:
            logger.warning(f"Semantic cache clear after pipeline ingest failed: {cache_err}")
    return {"collection": collection_name, "source_type": data.source_type, "stats": stats}


if __name__ == "__main__":
    import os
    import uvicorn

    # On Windows, multi-worker local run can hide startup exceptions behind
    # repeated "Child process died" messages.
    workers = 1 if os.name == "nt" else 2
    uvicorn.run("app:app", host="0.0.0.0", port=8002, workers=workers, log_level="info")