import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
import os
import re
import sys
import uuid
from copy import copy
from pathlib import Path
from typing import Dict, List, TypedDict

# Force sys.path to include the backend/src directory for Celery runtime worker threads
src_path = str(Path(__file__).resolve().parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from celery import shared_task
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.redis import RedisSaver
    _REDIS_SAVER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RedisSaver = None  # type: ignore
    _REDIS_SAVER_AVAILABLE = False

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from agent import ai_agent_handle, clear_user_runtime_caches
from verify_answer import judge_answer
from metacognitive import build_escalation, ESCALATION_PREFIX
from rlhf_store import find_similar_good
from brain import (
    detect_route,
    detect_user_intent,
    gen_doc_prompt,
    get_embedding,
    openai_chat_complete,
    vietnamese_llm_chat_complete,
)
from config import (
    DEFAULT_COLLECTION_NAME,
    DOC_GRADE_THRESHOLD,
    REFLECTION_MAX,
    ALL_IRRELEVANT_THRESHOLD,
    VERIFY_MAX_RETRIES,
    RLHF_RERANK_BOOST,
    TASK_MAX_RETRIES,
    TASK_RETRY_BACKOFF,
    TASK_RETRY_BACKOFF_MAX,
    TASK_RETRY_JITTER,
    GRAPH_RECURSION_LIMIT,
    GRAPH_RUN_TIMEOUT_S,
)
from retry_utils import with_retry, build_transient_exceptions
from graph_loop_control import GraphRunTimeout, _invoke_with_deadline

# Exception classes whose transient failure should trigger a Celery task retry.
# Built once at import; optional deps (redis/sqlalchemy/requests) imported lazily
# inside build_transient_exceptions so a missing client lib never breaks import.
_TRANSIENT_EXCEPTIONS = build_transient_exceptions()

# Shared autoretry kwargs for idempotent tasks. Applied ONLY to tasks with no
# non-idempotent side effects (or idempotent ones like uuid5-dedup episodic
# writes). Side-effectful tasks with append-only writes (llm_handle_message)
# must NOT use this — a retry would duplicate the user message + reply.
_AUTORETRY_KWARGS = dict(
    autoretry_for=_TRANSIENT_EXCEPTIONS,
    max_retries=TASK_MAX_RETRIES,
    retry_backoff=TASK_RETRY_BACKOFF,
    retry_backoff_max=TASK_RETRY_BACKOFF_MAX,
    retry_jitter=TASK_RETRY_JITTER,
)
from database import get_celery_app, SessionLocal, settings
from models import (
    ensure_database_schema,
    get_conversation_messages,
    update_chat_conversation,
    get_doc_chunks,
    save_doc_chunk,
    delete_doc_chunks_by_ids,
    clear_doc_chunks_by_doc,
    save_user_episode,
)
from query_rewriter import rewrite_query_to_multi_queries, rewrite_query_with_context
from rerank import rerank_documents
from search import hybrid_search, search_engine  # Import new hybrid search
from splitter import split_document
from summarizer import summarize_text
from tavily_tool import tavily_search_legal
from langgraph.types import Command
from utils import setup_logging
from vectorize import add_vector, search_vector, delete_vectors_by_ids, delete_vectors_by_filter

setup_logging()
logger = logging.getLogger(__name__)

import redis
redis_client = redis.from_url(settings.redis_url)

# Shared placeholder user_ids: writing episodic facts or per-user cache under
# one of these collapses every such client into a single bucket (cross-user
# leak). Skip per-user persistence when a sentinel is in use.
_SHARED_USER_SENTINELS = {"anonymous", "demo-session", ""}

# Module-level trace-redis client (lazy) so we don't open+leak a new
# connection pool on every message. The previous code did
# `redis.from_url(settings.trace_redis_url)` per chat and never closed it.
_trace_redis_client = None


def _get_trace_redis():
    global _trace_redis_client
    if _trace_redis_client is None:
        _trace_redis_client = redis.from_url(settings.trace_redis_url)
    return _trace_redis_client

# ---- Schema ensure guard: once per worker process ----
# ensure_database_schema() reflects all tables and runs idempotent CREATE IF
# NOT EXISTS. Cheap per call but unnecessary per message; a worker process lives
# across many tasks, so guard it with a module-level flag.
_SCHEMA_ENSURED = False


def _ensure_schema_once() -> None:
    global _SCHEMA_ENSURED
    if _SCHEMA_ENSURED:
        return
    try:
        ensure_database_schema()
    except Exception as e:
        logger.warning("Schema ensure failed on first call: %s", e)
        return
    _SCHEMA_ENSURED = True

from guardrails_manager import LegalGuardrailsManager
guardrails_manager = LegalGuardrailsManager()

from trace import emit_run_end, emit_run_start, emit_step


def _trace_node_end(state: "ChatGraphState", node: str, payload: Dict) -> None:
    """Best-effort node_end trace. No-op when run_id absent (e.g. legacy callers)."""
    run_id = state.get("run_id")
    if not run_id:
        return
    emit_step(run_id, state.get("thread_id", ""), node, "node_end", payload)


def _trace_handoff(state: "ChatGraphState", src: str, dst: str, reason: str) -> None:
    """Emit a handoff trace event (Phase D). Best-effort."""
    run_id = state.get("run_id")
    if not run_id:
        return
    emit_step(run_id, state.get("thread_id", ""), src, "handoff", {"from": src, "to": dst, "reason": reason})


# ---- Phase D multi-agent handoff predicates ----
# Pure string heuristics: detect canned "I don't have this" answers that should
# trigger a handoff to a more capable agent instead of ending the turn.
_HANDOFF_NOT_FOUND_MARKERS = (
    "không tìm thấy", "không tìm thấy thông tin", "không có đủ thông tin",
    "tôi không có thông tin", "không có trong tài liệu", "vượt quá khả năng",
)
_HANDOFF_NEEDS_LOOKUP_MARKERS = (
    "cần tra cứu", "tra cứu văn bản", "tôi sẽ tra cứu", "hãy tham khảo văn bản",
)


def _should_handoff_to_rag(answer: str) -> bool:
    """agent_tools -> retrieve: agent answer says it needs legal-doc lookup."""
    if not answer:
        return False
    low = answer.lower()
    return any(m in low for m in _HANDOFF_NEEDS_LOOKUP_MARKERS)


def _should_handoff_to_web(answer: str) -> bool:
    """generate -> web_search: RAG answer is a canned 'not found' message."""
    if not answer:
        return False
    low = answer.lower()
    return any(m in low for m in _HANDOFF_NOT_FOUND_MARKERS)


def _should_handoff_to_agent(answer: str) -> bool:
    """web_search -> agent_tools: web result looks like a question needing tool use."""
    if not answer:
        return False
    low = answer.lower()
    return any(m in low for m in _HANDOFF_NEEDS_LOOKUP_MARKERS)


def _planner_llm_call(prompt: str):
    """Best-effort LLM call for the planner. Returns response text or None.

    Lazily borrows the agent's Groq LLM; any failure -> None so the planner
    falls back to a route-derived 1-step plan. Kept defensive so a missing /
    misconfigured LLM never breaks routing.
    """
    try:
        from agent import _build_llm  # local import avoids heavy module load at import time

        llm = _build_llm()
        if llm is None:
            return None
        # llama-index LLM .complete returns a CompletionResponse with .text
        resp = llm.complete(prompt)
        return getattr(resp, "text", None) or str(resp)
    except Exception as exc:
        logger.warning(f"[PLANNER] LLM unavailable: {exc}")
        return None


# ---- Persistent event loop for async guardrails calls inside sync Celery tasks ----
# Previously every guardrails call did `asyncio.run(...)`, which creates and tears
# down a fresh event loop each time — wasteful and fragile under concurrency. Here
# we keep one daemon-thread loop for the worker's lifetime and schedule coroutines
# on it via run_coroutine_threadsafe.
import threading

_async_loop: asyncio.AbstractEventLoop | None = None
_async_loop_lock = threading.Lock()


def _get_persistent_loop() -> asyncio.AbstractEventLoop:
    global _async_loop
    with _async_loop_lock:
        if _async_loop is None or _async_loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True).start()
            _async_loop = loop
        return _async_loop


def run_async(coro):
    """Run a coroutine on a persistent background event loop (sync caller)."""
    loop = _get_persistent_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _compensate_index_rollback(doc_id_str: str) -> None:
    """Compensating rollback: wipe DB chunk metadata for a doc after a Qdrant
    write failed post-commit. Next index run then sees no chunks and treats the
    doc as first-time (re-cleans Qdrant + re-indexes fresh), recovering cleanly.
    """
    try:
        clear_doc_chunks_by_doc(doc_id_str)
        logger.warning(
            f"Compensating rollback done: cleared DB chunks for document {doc_id_str}. "
            "Next index run will re-embed and re-upsert from scratch."
        )
    except Exception as cleanup_err:
        logger.critical(
            f"Compensating rollback FAILED for document {doc_id_str}: {cleanup_err}. "
            "DB may contain chunk metadata pointing at missing Qdrant vectors — "
            "manual reindex recommended."
        )


class ChatGraphState(TypedDict, total=False):
    history: List[Dict]
    question: str
    route: str
    standalone_question: str
    response: str
    sources: List[Dict]
    user_id: str
    # Phase 2b — RBAC role + approval gating. role drives tool-policy filter
    # (rbac.filter_tools_by_policy) and the pre-flight approval gate
    # (approval.evaluate_tool_gate) for sensitive tools by non-exempt roles.
    role: str
    approved_tool: str
    # Phase 3 — planner + supervisor. plan = ordered specialist steps;
    # plan_step_idx = current step; supervisor_steps = handoffs executed
    # (capped at MAX_HANDOFF_STEPS to prevent infinite loops).
    plan: List[Dict]
    plan_step_idx: int
    supervisor_steps: int
    # CRAG (self-corrective RAG)
    documents: List[Dict]            # retrieved + reranked docs
    graded_docs: List[Dict]          # docs tagged with relevance: "relevant"|"irrelevant"
    rewritten_query: str             # query produced by rewrite_query_node on loop
    web_search_fallback_used: bool
    reflection_count: int            # CRAG loop guard (incremented in grade_documents_node)
    # multi-agent handoff (Phase D) + ReAct surfacing (Phase E) + trace (Phase C)
    messages: List[Dict]
    tool_calls: List[Dict]
    run_id: str
    thread_id: str
    # Phase D handoff guards — each directed edge fires at most once per run,
    # preventing agent_tools <-> web_search <-> generate cycles.
    agent_to_rag_done: bool
    generate_to_web_done: bool
    web_to_agent_done: bool
    # PEV verify_answer (final-answer groundedness gate) — Phase F.
    verify_score: float               # 0.0–1.0 groundedness from judge_answer
    verify_rationale: str
    verify_verdict: str               # "supported" | "partial" | "unsupported"
    retry_verify: int                 # verify -> rewrite_query recovery guard

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()


def follow_up_question(history, question):
    """Handle follow-up questions by rephrasing with context"""
    user_intent = detect_user_intent(history, question)
    logger.info(f"User intent (rephrased): {user_intent}")
    return user_intent


def retrieve_with_hybrid_search(queries: List[str], top_k: int = 5) -> List[Dict]:
    """
    Enhanced retrieval using hybrid search (semantic + keyword)
    This combines vector search with BM25 keyword search for better coverage.

    Queries are run concurrently (each does an embedding + Qdrant + BM25 round
    trip), so latency scales with the slowest query rather than the sum.

    Args:
        queries: List of query variations
        top_k: Number of documents to retrieve

    Returns:
        Merged and deduplicated list of documents with hybrid scores
    """
    all_docs: List[Dict] = []
    seen_contents = set()

    def _run_one(q: str) -> List[Dict]:
        logger.info(f"Hybrid search for query: {q}")
        try:
            return hybrid_search(q, limit=top_k)
        except Exception as e:
            logger.warning(f"Hybrid search failed for query '{q}': {e}")
            return []

    max_workers = min(len(queries), 4) if queries else 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        per_query_docs = list(pool.map(_run_one, queries))

    for query, docs in zip(queries, per_query_docs):
        # Deduplicate based on content
        for doc in docs:
            content_hash = hash(doc.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                # Preserve hybrid scoring information
                doc["retrieval_query"] = query
                all_docs.append(doc)

    logger.info(
        f"Retrieved {len(all_docs)} unique documents from {len(queries)} queries using hybrid search"
    )
    return all_docs


def retrieve_with_multi_query_fallback(
    queries: List[str], top_k: int = 5
) -> List[Dict]:
    """
    Fallback retrieval method using pure vector search
    Used when hybrid search is not available or fails
    """
    all_docs = []
    seen_contents = set()

    for query in queries:
        logger.info(f"Vector search fallback for query: {query}")
        # Get embedding for this query variation
        vector = get_embedding(query)

        # Search documents using pure vector search
        docs = search_vector(DEFAULT_COLLECTION_NAME, vector, top_k)

        # Deduplicate based on content
        for doc in docs:
            content_hash = hash(doc.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                doc["retrieval_query"] = query
                doc["search_method"] = "vector_fallback"
                all_docs.append(doc)

    logger.info(
        f"Retrieved {len(all_docs)} unique documents from {len(queries)} queries using vector fallback"
    )
    return all_docs


def _retrieve_episodic_context(user_id: str | None, question: str) -> str:
    """Retrieve relevant user episodic memories from the user_episodes Qdrant collection.

    Returns a newline-joined bullet list of past user facts, or "" on miss/failure.
    Extracted from generate_rag_answer so the CRAG generate_node can reuse it.
    """
    if not user_id:
        return ""
    try:
        logger.info(f"Retrieving episodic memory for user: {user_id}")
        query_vector = get_embedding(question)
        # Stricter threshold than the default 0.3: episodic context is only
        # injected when strongly relevant to the new question, so weakly-related
        # past facts don't nudge the model toward summarizing the user instead
        # of answering the new question.
        episodes = search_vector(
            collection_name="user_episodes",
            vector=query_vector,
            limit=3,
            filters={"user_id": user_id},
            score_threshold=0.5,
        )
        if not episodes:
            return ""
        logger.info(f"Retrieved {len(episodes)} episodes for user: {user_id}")
        lines = []
        for ep in episodes:
            summary = ep.get("content") or ep.get("summary") or ""
            if summary:
                lines.append(f"- {summary}")
        return "\n".join(lines)
    except Exception as ep_err:
        logger.error(f"Failed to retrieve episodic memory for user {user_id}: {ep_err}")
        return ""


def _apply_rlhf_rerank_boost(ranked_docs: List[Dict], user_id, question: str) -> List[Dict]:
    """Phase 4 — up-weight chunks whose ``doc_id`` backed a 👍-marked answer.

    Looks up this user's good answers semantically near ``question`` (same
    scope, no cross-user leak). If a match exists, any ranked chunk whose
    ``doc_id`` appears in that good answer's sources gets an additive
    ``RLHF_RERANK_BOOST`` and the list is re-sorted. Non-mutating (returns a
    new list); best-effort (any failure returns the input unchanged).
    """
    if not ranked_docs or not user_id or not question:
        return ranked_docs
    try:
        good = find_similar_good(user_id, question)
    except Exception as exc:
        logger.warning(f"RLHF rerank boost lookup failed (non-blocking): {exc}")
        return ranked_docs
    if not good or not good.get("sources"):
        return ranked_docs
    boosted_ids = {
        s.get("doc_id") for s in good["sources"] if s.get("doc_id") is not None
    }
    if not boosted_ids:
        return ranked_docs
    out = []
    for d in ranked_docs:
        if d.get("doc_id") in boosted_ids:
            d2 = dict(d)
            d2["relevance_score"] = float(d.get("relevance_score") or 0.0) + RLHF_RERANK_BOOST
            out.append(d2)
        else:
            out.append(d)
    out.sort(key=lambda x: x.get("relevance_score") or 0.0, reverse=True)
    logger.info(f"RLHF rerank boost applied to {len(boosted_ids)} doc_ids for user {user_id}")
    return out


def _llm_judge_relevance(question: str, docs: List[Dict]) -> Dict[int, str]:
    """LLM-as-judge relevance for borderline docs (those below the rerank score threshold).

    Returns a mapping {content_hash -> "relevant"|"irrelevant"} for the docs the LLM
    judged. Missing entries mean "unjudged" (caller defaults to irrelevant). Single
    batched LLM call via the existing openai_chat_complete helper — no new client.
    """
    if not docs:
        return {}
    numbered = []
    for i, doc in enumerate(docs):
        content = (doc.get("content") or "")[:500]
        numbered.append(f"[{i}] {content}")
    prompt = (
        "Đánh giá mỗi đoạn tài liệu dưới đây có liên quan đến câu hỏi pháp luật hay không.\n"
        f"Câu hỏi: {question}\n"
        f"Các đoạn:\n" + "\n".join(numbered) + "\n"
        "Trả lời duy nhất 1 dòng cho mỗi đoạn theo định dạng: [index] yes  hoặc  [index] no\n"
        "Không giải thích thêm."
    )
    try:
        resp = openai_chat_complete([
            {"role": "system", "content": "Bạn là bộ đánh giá relevance tài liệu pháp luật Việt Nam."},
            {"role": "user", "content": prompt},
        ])
    except Exception as e:
        logger.warning(f"LLM relevance judge failed, defaulting borderline docs to irrelevant: {e}")
        return {}

    verdicts: Dict[int, str] = {}
    for line in resp.splitlines():
        m = re.match(r"\[?\s*(\d+)\s*\]?\s*(yes|no)", line.strip().lower())
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx < len(docs):
            verdicts[hash(docs[idx].get("content", ""))] = "relevant" if m.group(2) == "yes" else "irrelevant"
    return verdicts


def _tag_documents_relevance(documents: List[Dict], verdicts: Dict[int, str]) -> List[Dict]:
    """Pure: tag each doc "relevant"|"irrelevant" using rerank score + LLM verdicts.

    Docs with relevance_score >= DOC_GRADE_THRESHOLD -> relevant.
    Otherwise consult verdicts (content_hash -> verdict); missing -> irrelevant.
    Returns NEW dicts (no mutation of input).
    """
    graded = []
    for doc in documents:
        score = doc.get("relevance_score")
        base = {**doc}
        if score is not None and score >= DOC_GRADE_THRESHOLD:
            base["relevance"] = "relevant"
        else:
            v = verdicts.get(hash(doc.get("content", "")))
            base["relevance"] = v if v in ("relevant", "irrelevant") else "irrelevant"
        graded.append(base)
    return graded


def _decide_after_grade(graded_docs: List[Dict], reflection_count: int) -> str:
    """Pure routing decision after grading. Returns "generate"|"rewrite_query"|"web_search"."""
    relevant_count = sum(1 for d in graded_docs if d.get("relevance") == "relevant")
    if relevant_count >= ALL_IRRELEVANT_THRESHOLD:
        return "generate"
    if reflection_count >= REFLECTION_MAX:
        return "web_search"
    return "rewrite_query"


def generate_rag_answer(history, question, user_id=None):
    """Pure function for RAG answer generation.

    Expects an already-normalized question.
    """
    standalone_question = question
    logger.info(f"Standalone question: {standalone_question}")

    query_variations = rewrite_query_to_multi_queries(
        standalone_question, num_queries=3
    )
    logger.info(f"Query variations: {query_variations}")

    try:
        retrieved_docs = retrieve_with_hybrid_search(query_variations, top_k=4)
        logger.info(f"Hybrid search retrieved {len(retrieved_docs)} documents")
    except Exception as e:
        logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
        retrieved_docs = retrieve_with_multi_query_fallback(query_variations, top_k=4)

    logger.info(f"Retrieved {len(retrieved_docs)} documents before reranking")
    ranked_docs = rerank_documents(retrieved_docs, standalone_question, top_n=5)
    logger.info(f"Top {len(ranked_docs)} documents after reranking")

    # Episodic memory retrieval (delegated to shared helper, reused by CRAG generate_node)
    episodic_context = _retrieve_episodic_context(user_id, standalone_question)

    system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp
2. Trích dẫn chính xác các điều khoản, khoản, điểm từ văn bản pháp luật
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu thông tin không đủ trong tài liệu, hãy nói rõ điều đó
5. Luôn đưa ra câu trả lời có căn cứ pháp lý

QUAN TRỌNG: Chỉ sử dụng thông tin từ các tài liệu được cung cấp bên dưới.
LUÔN trả lời trực tiếp câu hỏi MỚI của người dùng. KHÔNG bao giờ tóm tắt, nhắc lại hay mô tả lại thông tin lịch sử của người dùng; ngữ cảnh lịch sử (nếu có) chỉ là gợi ý phụ để cá nhân hóa, tuyệt đối không dùng làm nội dung chính của câu trả lời.""" + _date_context_block()

    doc_context = gen_doc_prompt(ranked_docs)

    # Episodic context moved INTO the system prompt as a clearly-labeled
    # background hint (not in the user message). Putting it in the user message
    # caused weak models to mimic it and output a summary OF the user instead
    # of answering the new question.
    background_block = ""
    if episodic_context:
        background_block = (
            "\n\n[Ngữ cảnh phụ — sự kiện người dùng từ các cuộc trò chuyện trước, "
            "chỉ dùng để cá nhân hóa, KHÔNG được tóm tắt hay lặp lại trong câu trả lời]:\n"
            f"{episodic_context}\n"
        )

    openai_messages = (
        [{"role": "system", "content": system_prompt + background_block}]
        + history
        + [
            {
                "role": "user",
                "content": f"Tài liệu tham khảo:\n{doc_context}\n\nCâu hỏi: {question}\n\nHãy trả lời dựa trên các tài liệu pháp luật trên.",
            }
        ]
    )

    logger.info(f"Sending {len(openai_messages)} messages to Vietnamese LLM")
    assistant_answer = vietnamese_llm_chat_complete(openai_messages)
    logger.info("Bot RAG reply generated successfully")
    
    # RAG Hallucination Guardrail Check
    if guardrails_manager.initialized:
        try:
            logger.info("Verifying RAG groundedness using NeMo Guardrails")
            assistant_answer = run_async(
                guardrails_manager.verify_output_rag(assistant_answer, doc_context)
            )
        except Exception as e:
            logger.error(f"Error running output RAG guardrails: {e}")
            
    return assistant_answer, ranked_docs


def generate_agent_answer(history, question, user_id=None, conversation_id=None, role=None,
                          run_id=None, thread_id=None):
    logger.info("Using ReAct agent with tools")
    answer, tool_calls = ai_agent_handle(
        question, user_id, conversation_id, history=history, role=role,
        run_id=run_id, thread_id=thread_id,
    )
    return answer, tool_calls


def _maybe_block_on_approval(state, question, history, role, user_id, run_id):
    """Phase 2b approval gate for ``agent_tools_node``.

    Returns a Vietnamese "chờ phê duyệt" response string when a non-exempt
    role is anticipated to call a sensitive tool not yet approved for this
    run, else ``None`` (proceed). Exempt roles (admin/lawyer) and the legacy
    anonymous path (``role`` None) always proceed.

    Anticipated tools are derived from ``filter_tools_for_query`` (same
    selection the ReAct agent will use), so the gate predicts the agent's
    actual tool set. Only the first blocking sensitive tool surfaces; once
    approved and the client re-posts, the gate re-runs and may surface the
    next one.
    """
    if not role:
        return None
    try:
        from rbac import SENSITIVE_TOOLS, Principal, Role, _tool_name
        from approval import evaluate_tool_gate, await_approval_response

        principal = Principal(user_id=user_id or "", username="", role=role)
        if principal.is_approval_exempt:
            return None
        anticipated = filter_tools_for_query(question, history, role=role)
        names = {_tool_name(t) for t in anticipated}
        decision, approval = evaluate_tool_gate(principal, names, run_id=run_id)
        if decision == "await_approval" and approval is not None:
            logger.info(
                f"[GATE] Blocking run {run_id}: tool {approval.tool_name} needs approval "
                f"(user={user_id}, role={role})"
            )
            return await_approval_response(approval)
    except Exception as exc:
        # Gate failure must never break chat; fall through to the agent.
        logger.warning(f"[GATE] approval gate failed (proceeding): {exc}")
        return None
    return None


def _date_context_block() -> str:
    """Grounding block: injects current date so the LLM never hallucinates
    a stale year (e.g. answering 'năm nay' with a training-cutoff year).
    Used by every generation node that may answer time-dependent questions."""
    from datetime import datetime
    today = datetime.now()
    return (
        f"\n\nTHÔNG TIN THỜI GIAN: Hôm nay là {today.strftime('%d/%m/%Y')} "
        f"(năm {today.year}). Khi câu hỏi phụ thuộc thời gian — tính tuổi, "
        f"thời hiệu khởi kiện, hiệu lực văn bản, lương tối thiểu vùng, văn bản "
        f"mới ban hành — PHẢI dùng mốc này, tuyệt đối KHÔNG dùng năm khác."
    )


def generate_web_search_answer(history, question):
    logger.info("Using Tavily web search for query")
    search_results = tavily_search_legal(question, max_results=5)

    system_prompt = """Bạn là trợ lý AI giúp tìm kiếm thông tin pháp luật trên internet.
    Hãy tổng hợp và trả lời câu hỏi dựa trên kết quả tìm kiếm được cung cấp.""" + _date_context_block()

    openai_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [
            {
                "role": "user",
                "content": f"Kết quả tìm kiếm:\n{search_results}\n\nCâu hỏi: {question}\n\nHãy tổng hợp thông tin và trả lời.",
            }
        ]
    )

    return openai_chat_complete(openai_messages)


def generate_general_answer(history, question):
    logger.info("Using general chat")

    system_prompt = """Bạn là trợ lý AI thân thiện của hệ thống tư vấn pháp luật Việt Nam.
Hãy trả lời lịch sự và hướng dẫn người dùng về các câu hỏi pháp luật bạn có thể giúp đỡ.

Bạn có thể:
- Trả lời câu hỏi về luật pháp Việt Nam
- Tính toán phí phạt, chia thừa kế, kiểm tra tuổi pháp lý
- Tìm kiếm thông tin pháp luật mới trên internet
- Hướng dẫn thủ tục pháp lý""" + _date_context_block()

    openai_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

    return vietnamese_llm_chat_complete(openai_messages)


@lru_cache(maxsize=1)
def _get_checkpointer():
    """Build (once) the LangGraph checkpoint saver.

    RedisSaver when langgraph-checkpoint-redis is installed AND Redis is reachable
    AND use_memory_saver is False; otherwise in-process MemorySaver (dev/test).
    Isolation between conversations is by thread_id in the run config, not by
    saver instance, so a single saver serves all threads.
    """
    if settings.use_memory_saver or not _REDIS_SAVER_AVAILABLE:
        logger.info("Using MemorySaver checkpointer (dev/test fallback).")
        return MemorySaver()

    try:
        probe = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        probe.ping()
        # RedisSaver requires the RedisJSON module (shipped with Redis Stack),
        # not just a plain Redis server. Verify JSON.SET actually works before
        # committing to RedisSaver, otherwise graph.invoke will fail mid-run.
        probe.json().set("checkpoint:probe", "$", {"ok": True})
        probe.delete("checkpoint:probe")
        probe.close()
    except Exception as e:
        logger.warning(
            f"Redis checkpointer unavailable or missing RedisJSON/Redis Stack module ({e}); "
            "falling back to MemorySaver."
        )
        return MemorySaver()

    ttl_minutes = max(1, settings.langgraph_checkpoint_ttl_seconds // 60)
    saver = RedisSaver(
        redis_url=settings.redis_url,
        ttl={"default_ttl": ttl_minutes},
    )
    logger.info(f"Using RedisSaver checkpointer (ttl={ttl_minutes}min, url={settings.redis_url}).")
    return saver


def _build_chat_graph():
    graph = StateGraph(ChatGraphState)

    def route_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("question", "")
        route = detect_route(history, question)
        standalone_question = question

        if route != "general_chat":
            standalone_question = follow_up_question(history, question)

        _trace_node_end(state, "route", {"route": route})
        return {
            "route": route,
            "standalone_question": standalone_question,
        }

    # ---- Phase 3: planner ----
    # Produces an ordered plan (list of {specialist, goal}) and records it in
    # state + trace. Falls back to a 1-step plan derived from the route when
    # the LLM is unavailable or returns an unparseable response. The plan is
    # observational + drives supervisor handoff; the existing route->specialist
    # edges still execute the first step.
    def planner_node(state: ChatGraphState):
        question = state.get("standalone_question") or state.get("question", "")
        route = state.get("route", "general_chat")
        try:
            from planner import build_plan_prompt, parse_plan, validate_plan, fallback_plan

            plan = fallback_plan(route)
            try:
                prompt = build_plan_prompt(question, history_summary="")
                raw = _planner_llm_call(prompt)
                if raw:
                    parsed = validate_plan(parse_plan(raw))
                    if parsed:
                        plan = parsed
            except Exception as exc:
                logger.warning(f"[PLANNER] LLM plan failed, using route fallback: {exc}")
            _trace_node_end(state, "planner", {"plan_steps": len(plan), "plan": plan})
            return {"plan": plan, "plan_step_idx": 0, "supervisor_steps": 0}
        except Exception as exc:
            logger.warning(f"[PLANNER] node failed (non-fatal): {exc}")
            _trace_node_end(state, "planner", {"plan_steps": 1, "fallback": True})
            return {"plan": [{"specialist": "chat", "goal": "fallback"}], "plan_step_idx": 0}

    # ---- Phase 3: supervisor handoff helper ----
    # specialist name -> graph node name
    _SPEC_TO_NODE = {
        "rag": "retrieve",
        "tool": "agent_tools",
        "web": "web_search",
        "chat": "general_chat",
    }

    def _supervisor_next(state, current_specialist, answer, already_done_flag):
        """Ask the supervisor for the next step. Returns a Command(goto=...)
        when a handoff is decided (and the per-edge guard not already fired),
        else None (caller falls through to its default -> metacognitive edge).

        Replaces the bare _should_handoff_to_* heuristics: supervisor_decide
        tries the LLM first, then falls back to those same heuristics, and
        enforces the MAX_HANDOFF_STEPS loop guard.
        """
        from supervisor import supervisor_decide, END as SUP_END

        steps = state.get("supervisor_steps", 0)
        plan = state.get("plan", [])
        question = state.get("standalone_question") or state.get("question", "")
        decision = supervisor_decide(
            question, current_specialist, answer, plan, steps_taken=steps,
            llm_call=_planner_llm_call,
        )
        nxt = decision.get("next", SUP_END)
        if nxt == SUP_END or nxt not in _SPEC_TO_NODE:
            return None
        if state.get(already_done_flag):
            return None  # per-edge guard: this handoff already fired this run
        _trace_handoff(state, current_specialist, _SPEC_TO_NODE[nxt], decision.get("rationale", ""))
        return Command(
            goto=_SPEC_TO_NODE[nxt],
            update={"supervisor_steps": steps + 1, already_done_flag: True},
        )

    # ---- CRAG (self-corrective RAG) nodes ----
    def retrieve_node(state: ChatGraphState):
        # On a reflection loop, prefer the rewritten query; otherwise the standalone question.
        question = (
            state.get("rewritten_query")
            or state.get("standalone_question")
            or state.get("question", "")
        )
        user_id = state.get("user_id")
        query_variations = rewrite_query_to_multi_queries(question, num_queries=3)
        try:
            retrieved = retrieve_with_hybrid_search(query_variations, top_k=4)
        except Exception as e:
            logger.warning(f"Hybrid search failed, vector fallback: {e}")
            retrieved = retrieve_with_multi_query_fallback(query_variations, top_k=4)
        # Rerank here so docs carry relevance_score for the grade node.
        ranked = rerank_documents(retrieved, question, top_n=5)
        # Phase 4 — RLHF rerank boost (user-scoped 👍-marked source up-weight).
        ranked = _apply_rlhf_rerank_boost(ranked, user_id, question)
        logger.info(f"Retrieve node: {len(ranked)} reranked docs for query: {question[:80]}")
        _trace_node_end(state, "retrieve", {"query": question[:200], "doc_count": len(ranked)})
        return {"documents": ranked}

    def grade_documents_node(state: ChatGraphState):
        documents = state.get("documents", [])
        question = state.get("standalone_question") or state.get("question", "")
        borderline = [
            d for d in documents
            if d.get("relevance_score") is None or d.get("relevance_score") < DOC_GRADE_THRESHOLD
        ]
        verdicts = _llm_judge_relevance(question, borderline) if borderline else {}
        graded = _tag_documents_relevance(documents, verdicts)
        # reflection_count is incremented ONLY here; rewrite_query does not increment,
        # so each retrieve->grade pass counts one reflection.
        reflection_count = state.get("reflection_count", 0) + 1
        relevant_n = sum(1 for g in graded if g.get("relevance") == "relevant")
        logger.info(
            f"Grade node: {relevant_n}/{len(graded)} relevant, reflection_count={reflection_count}"
        )
        _trace_node_end(state, "grade_documents", {
            "relevant": relevant_n,
            "total": len(graded),
            "reflection_count": reflection_count,
        })
        return {"graded_docs": graded, "reflection_count": reflection_count}

    def decide_after_grade(state: ChatGraphState):
        return _decide_after_grade(state.get("graded_docs", []), state.get("reflection_count", 0))

    def rewrite_query_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question") or state.get("question", "")
        rewritten = rewrite_query_with_context(question, history)
        logger.info(f"Rewrite query (reflection loop): {rewritten}")
        # NOTE: do not increment reflection_count here; grade_documents_node owns the counter.
        _trace_node_end(state, "rewrite_query", {"rewritten": rewritten[:200]})
        return {"rewritten_query": rewritten}

    def generate_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question") or state.get("question", "")
        user_id = state.get("user_id")
        graded = state.get("graded_docs", [])
        ranked_docs = [d for d in graded if d.get("relevance") == "relevant"] or list(graded)

        episodic_context = _retrieve_episodic_context(user_id, question)

        system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp
2. Trích dẫn chính xác các điều khoản, khoản, điểm từ văn bản pháp luật
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu thông tin không đủ trong tài liệu, hãy nói rõ điều đó
5. Luôn đưa ra câu trả lời có căn cứ pháp lý

QUAN TRỌNG: Chỉ sử dụng thông tin từ các tài liệu được cung cấp bên dưới.
LUÔN trả lời trực tiếp câu hỏi MỚI của người dùng. KHÔNG bao giờ tóm tắt, nhắc lại hay mô tả lại thông tin lịch sử của người dùng; ngữ cảnh lịch sử (nếu có) chỉ là gợi ý phụ để cá nhân hóa, tuyệt đối không dùng làm nội dung chính của câu trả lời.""" + _date_context_block()

        doc_context = gen_doc_prompt(ranked_docs)

        background_block = ""
        if episodic_context:
            background_block = (
                "\n\n[Ngữ cảnh phụ — sự kiện người dùng từ các cuộc trò chuyện trước, "
                "chỉ dùng để cá nhân hóa, KHÔNG được tóm tắt hay lặp lại trong câu trả lời]:\n"
                f"{episodic_context}\n"
            )

        openai_messages = (
            [{"role": "system", "content": system_prompt + background_block}]
            + history
            + [
                {
                    "role": "user",
                    "content": (
                        f"Tài liệu tham khảo:\n{doc_context}\n\n"
                        f"Câu hỏi: {question}\n\nHãy trả lời dựa trên các tài liệu pháp luật trên."
                    ),
                }
            ]
        )

        # Phase 4 — RLHF few-shot injection: if this user previously 👍-marked an
        # answer to a semantically similar question (score >= 0.85, same scope),
        # surface it as a system few-shot example to steer the LLM toward the
        # style/grounding they endorsed. Never cross-user (scope-filtered).
        try:
            good = find_similar_good(user_id, question)
            if good and good.get("response"):
                few_shot = (
                    f"\n\n[Ví dụ trả lời tốt cho câu tương tự đã được người dùng đánh giá cao — "
                    f"chỉ tham khảo về phong cách/cách dẫn chứng, KHÔNG chép nguyên văn]:\n"
                    f"Q: {good['question']}\nA: {good['response']}"
                )
                openai_messages[0]["content"] += few_shot
                logger.info("RLHF few-shot example injected (score=%.3f).", good.get("score", 0.0))
        except Exception as exc:
            logger.warning("RLHF few-shot injection failed (non-blocking): %s", exc)

        assistant_answer = vietnamese_llm_chat_complete(openai_messages)
        logger.info("CRAG generate_node reply generated")

        # RAG groundedness guardrail + legal disclaimer (replaces old legal_rag_node behavior)
        if guardrails_manager.initialized:
            try:
                assistant_answer = run_async(
                    guardrails_manager.verify_output_rag(assistant_answer, doc_context)
                )
            except Exception as e:
                logger.error(f"Error running output RAG guardrails: {e}")
            assistant_answer = guardrails_manager.add_legal_disclaimer(assistant_answer, question)

        _trace_node_end(state, "generate", {"doc_count": len(ranked_docs), "answer_len": len(assistant_answer)})

        # Phase D handoff: canned "not found" RAG answer -> escalate to web_search.
        if _should_handoff_to_web(assistant_answer) and not state.get("generate_to_web_done"):
            _trace_handoff(state, "generate", "web_search", "rag_not_found")
            return Command(goto="web_search", update={"generate_to_web_done": True})

        return {"response": assistant_answer, "sources": ranked_docs}

    def agent_tools_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question", state.get("question", ""))
        user_id = state.get("user_id")
        thread_id = state.get("thread_id")
        role = state.get("role")
        run_id = state.get("run_id")

        # Phase 2b — pre-flight approval gate. Compute the tool set the agent
        # is anticipated to call, intersect with SENSITIVE_TOOLS, and if a
        # non-exempt role needs one not yet approved this run, block with a
        # "chờ phê duyệt" response carrying the approval_id. The client re-
        # posts after an admin decides; the gate then sees the tool as allowed
        # and proceeds. Exempt roles (admin/lawyer) and the legacy anonymous
        # path (role None) skip the gate entirely.
        gate_resp = _maybe_block_on_approval(state, question, history, role, user_id, run_id)
        if gate_resp is not None:
            _trace_node_end(state, "agent_tools", {"approval_gate": True})
            return {"response": gate_resp, "sources": [], "tool_calls": []}

        resp, tool_calls = generate_agent_answer(
            history, question, user_id=user_id, conversation_id=thread_id, role=role,
            run_id=run_id, thread_id=thread_id,
        )
        if guardrails_manager.initialized:
            resp = guardrails_manager.add_legal_disclaimer(resp, question)
        _trace_node_end(state, "agent_tools", {"answer_len": len(resp), "tool_calls": len(tool_calls)})

        # Phase 3: supervisor-driven handoff (LLM decision with heuristic
        # fallback). Replaces the bare _should_handoff_to_rag heuristic.
        cmd = _supervisor_next(state, "tool", resp, "agent_to_rag_done")
        if cmd is not None:
            # Preserve tool_calls + standalone_question for the retrieve node.
            update = dict(cmd.update)
            update["standalone_question"] = question
            update["tool_calls"] = tool_calls
            return Command(goto=cmd.goto, update=update)

        return {"response": resp, "sources": [], "tool_calls": tool_calls}

    def web_search_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question", state.get("question", ""))
        resp = generate_web_search_answer(history, question)
        if guardrails_manager.initialized:
            resp = guardrails_manager.add_legal_disclaimer(resp, question)
        _trace_node_end(state, "web_search", {"answer_len": len(resp)})

        # Phase 3: supervisor-driven handoff (LLM + heuristic fallback),
        # replacing the bare _should_handoff_to_agent heuristic.
        cmd = _supervisor_next(state, "web", resp, "web_to_agent_done")
        if cmd is not None:
            return cmd

        return {"response": resp, "sources": [], "web_search_fallback_used": True}

    def general_chat_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("question", "")
        _trace_node_end(state, "general_chat", {})
        return {"response": generate_general_answer(history, question), "sources": []}

    # ---- PEV verify_answer (Phase F) ----
    # Judge the final answer for citation groundedness + hallucination.
    # Only the RAG route carries real sources; agent/web/general return
    # sources=[] and judge_answer short-circuits to "supported".
    def verify_answer_node(state: ChatGraphState):
        question = state.get("standalone_question") or state.get("question", "")
        answer = state.get("response", "")
        sources = state.get("sources", [])
        verdict = judge_answer(question, answer, sources)
        retry = state.get("retry_verify", 0)
        _trace_node_end(state, "verify_answer", {
            "score": verdict["score"],
            "verdict": verdict["verdict"],
            "retry_verify": retry + 1,
        })
        return {
            "verify_score": verdict["score"],
            "verify_rationale": verdict["rationale"],
            "verify_verdict": verdict["verdict"],
            "retry_verify": retry + 1,
        }

    def verify_router(state: ChatGraphState) -> str:
        verdict = state.get("verify_verdict", "supported")
        retry = state.get("retry_verify", 0)
        if verdict == "supported":
            return "metacognitive"
        if retry >= VERIFY_MAX_RETRIES:
            return "metacognitive"          # degrade — never infinite-loop
        return "rewrite_query"              # recovery -> CRAG retrieve loop

    # ---- Metacognitive escalation (Phase 2) ----
    # Graph-level safety gate before END: combine verify_answer confidence
    # with a tiered stakes classifier; prepend a lawyer-escalation prefix
    # when the stakes are high, or medium with low confidence. The original
    # answer is kept intact below the prefix (auditable in the trace).
    def metacognitive_node(state: ChatGraphState):
        question = state.get("standalone_question") or state.get("question", "")
        confidence = state.get("verify_score", 1.0)
        decision = build_escalation(question, confidence)
        _trace_node_end(state, "metacognitive", decision)
        if decision["escalate"]:
            response = state.get("response", "")
            return {"response": ESCALATION_PREFIX + response}
        return {}

    def choose_route(state: ChatGraphState):
        return state.get("route", "general_chat")

    graph.add_node("route", route_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("agent_tools", agent_tools_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("verify_answer", verify_answer_node)
    graph.add_node("metacognitive", metacognitive_node)

    graph.add_edge(START, "route")
    # Phase 3: route -> planner (records plan + trace) -> specialist via choose_route.
    graph.add_edge("route", "planner")
    graph.add_conditional_edges(
        "planner",
        choose_route,
        {
            "legal_rag": "retrieve",
            "agent_tools": "agent_tools",
            "web_search": "web_search",
            "general_chat": "general_chat",
        },
    )

    # CRAG loop: retrieve -> grade -> {generate | rewrite_query (-> retrieve) | web_search}
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")

    # PEV verify_answer gate: generate -> verify -> {rewrite_query (recovery) | metacognitive}.
    # agent_tools / web_search / general_chat carry no sources -> skip verify,
    # go straight to metacognitive.
    graph.add_edge("generate", "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        verify_router,
        {
            "rewrite_query": "rewrite_query",
            "metacognitive": "metacognitive",
        },
    )
    graph.add_edge("agent_tools", "metacognitive")
    graph.add_edge("web_search", "metacognitive")
    graph.add_edge("general_chat", "metacognitive")
    graph.add_edge("metacognitive", END)

    return graph.compile(checkpointer=_get_checkpointer())


@lru_cache(maxsize=1)
def get_chat_graph():
    return _build_chat_graph()


def run_chat_graph(history, question, user_id=None, conversation_id=None,
                   run_id=None, role=None, variant=None, shadow=False):
    """Invoke the chat graph.

    P6 extensions (opt-in, default off):
    - ``variant``: explicit model override (sets LLM_MODEL_CONTEXTVAR for this call).
    - Cost routing: when ``config.COST_ROUTING_ENABLED`` and no explicit variant,
      the route's chosen model is applied via the same contextvar. The route is
      inferred from a keyword guess (best-effort: falls back to general_chat).
    - ``shadow``: when true (and ``config.SHADOW_MODE_ENABLED``), also run the
      candidate variant graph in a try/except and persist its trace under a
      ``<run_id>:shadow`` run id for offline comparison. The user always
      receives the primary result.
    """
    from brain import LLM_MODEL_CONTEXTVAR
    try:
        from config import COST_ROUTING_ENABLED, SHADOW_MODE_ENABLED
    except Exception:
        COST_ROUTING_ENABLED = False
        SHADOW_MODE_ENABLED = False

    override_model = variant
    if override_model is None and COST_ROUTING_ENABLED:
        try:
            from evaluation.cost_routing import select_model_for_route
            route_guess = _guess_route(question)
            override_model = select_model_for_route(route_guess).model
        except Exception as exc:
            logger.debug(f"cost routing skipped: {exc}")
            override_model = None

    token = None
    if override_model is not None:
        token = LLM_MODEL_CONTEXTVAR.set(override_model)
    try:
        graph = get_chat_graph()
        thread_id = conversation_id or (f"user:{user_id}" if user_id else "default")
        config = {
            "configurable": {"thread_id": thread_id},
            # Hard recursion cap — runaway supervisor/ReAct/verify loops raise
            # GraphRecursionError instead of spinning forever. Bounded above by
            # GRAPH_RECURSION_LIMIT (env-tunable). Caught + degraded, NOT retried.
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        result = _invoke_with_deadline(
            graph,
            {
                "history": history,
                "question": question,
                "user_id": user_id,
                "thread_id": thread_id,
                "run_id": run_id,
                "role": role,
            },
            config=config,
            timeout_s=GRAPH_RUN_TIMEOUT_S,
        )
    finally:
        if token is not None:
            LLM_MODEL_CONTEXTVAR.reset(token)

    primary = {
        "response": result.get("response", ""),
        "sources": result.get("sources", []),
        "route": result.get("route", ""),
        "reflection_count": result.get("reflection_count", 0),
        "run_id": run_id,
        "tool_calls": result.get("tool_calls", []),
        "verify_score": result.get("verify_score", 0.0),
        "verify_verdict": result.get("verify_verdict", ""),
        "retry_verify": result.get("retry_verify", 0),
        "variant": variant,
    }

    if shadow and SHADOW_MODE_ENABLED:
        _run_shadow(history, question, user_id, conversation_id, role, run_id)

    return primary


def _guess_route(question: str) -> str | None:
    """Lightweight keyword route guess for cost routing (no LLM call)."""
    q = (question or "").lower()
    if any(k in q for k in ("điều", "bộ luật", "nghị định", "luật")):
        return "legal_rag"
    if any(k in q for k in ("tính", "calculator", "bao nhiêu phần trăm")):
        return "agent_tools"
    if any(k in q for k in ("tìm kiếm", "tin tức", "search the web")):
        return "web_search"
    return "general_chat"


def _run_shadow(history, question, user_id, conversation_id, role, run_id) -> None:
    """Run the candidate (big) model variant and persist a trace marker.

    Fire-and-forget: never surfaces to the user, never raises into primary path.
    """
    try:
        from trace import emit_step
        shadow_run_id = f"{run_id}:shadow" if run_id else None
        shadow_result = run_chat_graph(
            history, question, user_id=user_id,
            conversation_id=conversation_id, run_id=shadow_run_id,
            role=role, variant=None, shadow=False,
        )
        if shadow_run_id:
            emit_step(shadow_run_id, conversation_id or "default",
                      "__shadow__", "shadow_complete",
                      {"primary_route": None,
                       "shadow_response_len": len(shadow_result.get("response", ""))})
    except Exception as exc:
        logger.warning(f"shadow run failed (non-fatal): {exc}")


@shared_task(**_AUTORETRY_KWARGS)
def bot_rag_answer_message(history, question):
    standalone_question = follow_up_question(history, question)
    ans, sources = generate_rag_answer(history, standalone_question)
    return ans


def index_document_v2(id, question, content, collection_name=DEFAULT_COLLECTION_NAME):
    import hashlib
    import uuid
    from custom_embedding import get_custom_embedding

    doc_id_str = str(id)
    text = question + " " + content
    doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    full_doc_cid = f"{doc_id_str}#full_doc"

    # Acquire Redis Distributed Lock for this document ID to prevent race conditions
    lock_name = f"lock:doc:{doc_id_str}"
    lock = None
    try:
        lock = redis_client.lock(lock_name, timeout=120, blocking_timeout=30)
        if not lock.acquire():
            logger.warning(f"Could not acquire lock for document {doc_id_str} within timeout.")
            return [{"action": "error", "reason": "lock_timeout"}]
    except Exception as lock_err:
        logger.warning(f"Redis distributed lock service unavailable, proceeding without locking: {lock_err}")
        lock = None

    status_list = []
    db = SessionLocal()
    try:
        # Fetch old chunks from Metadata Store
        old_chunks = get_doc_chunks(doc_id_str)
        old_chunks_dict = {c.chunk_id: c.chunk_hash for c in old_chunks}

        # If this is the FIRST time indexing this document in the new system (MySQL has no chunks),
        # we clean up any old vectors belonging to this doc_id from Qdrant to avoid duplicates.
        if not old_chunks:
            logger.info(f"First-time indexing for document {doc_id_str} in the new system. Cleaning old Qdrant vectors...")
            delete_vectors_by_filter(collection_name, {"doc_id": doc_id_str})
            try:
                numeric_id = int(id)
                delete_vectors_by_filter(collection_name, {"doc_id": numeric_id})
            except (ValueError, TypeError):
                pass

        # Fast-path check: If full document hash matches, skip splitting and embedding entirely!
        if full_doc_cid in old_chunks_dict and old_chunks_dict[full_doc_cid] == doc_hash:
            logger.info(f"Skipped entire document {doc_id_str} - full content hash matches.")
            status_list.append({"action": "skip", "reason": "no_changes"})
            return status_list

        nodes = split_document(text)

        new_chunk_ids = set()
        to_upsert = []
        to_delete = []

        # Classify chunks
        for i, node in enumerate(nodes):
            # Generate deterministic UUID chunk ID
            cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{doc_id_str}_chunk_{i}"))
            new_chunk_ids.add(cid)
            chash = hashlib.md5(node.text.encode("utf-8")).hexdigest()

            if cid not in old_chunks_dict or old_chunks_dict[cid] != chash:
                to_upsert.append((cid, node.text, chash))

        # Chunks in DB that are no longer in new document should be deleted
        for old_cid in old_chunks_dict.keys():
            # Do not delete the special #full_doc hash record yet, it will be updated or deleted anyway
            if old_cid != full_doc_cid and old_cid not in new_chunk_ids:
                to_delete.append(old_cid)

        # 1. Stage orphan deletions in DB (Qdrant delete happens AFTER commit).
        if to_delete:
            logger.info(f"Staging deletion of {len(to_delete)} orphaned chunks for document {doc_id_str}")
            delete_doc_chunks_by_ids(to_delete, db=db)
            status_list.append({"action": "delete", "count": len(to_delete)})

        # 2. Embed and stage DB inserts (no Qdrant writes yet).
        vectors_payload = None
        if to_upsert:
            logger.info(f"Embedding and upserting {len(to_upsert)} chunks for document {doc_id_str}")

            # Batch embedding optimization!
            texts_to_embed = [item[1] for item in to_upsert]
            try:
                embeddings = get_custom_embedding(texts_to_embed)
                if not isinstance(embeddings, list) or (embeddings and not isinstance(embeddings[0], list)):
                    # If single embedding returned (fallback or error)
                    embeddings = [embeddings]
            except Exception as e:
                logger.error(f"Failed to generate embeddings in batch: {e}")
                raise e

            vectors_payload = {}
            for (cid, text_val, chash), vector in zip(to_upsert, embeddings):
                vectors_payload[cid] = {
                    "vector": vector,
                    "payload": {
                        "question": question,
                        "content": text_val,
                        "source": "document",
                        "doc_id": id,
                    }
                }
                save_doc_chunk(doc_id_str, cid, chash, db=db)

            # Stage full document hash record (so next run can fast-path skip).
            save_doc_chunk(doc_id_str, full_doc_cid, doc_hash, db=db)
        else:
            logger.info(f"Skipped indexing for document {doc_id_str} - no changes detected.")
            status_list.append({"action": "skip", "reason": "no_changes"})

        # 3. Commit metadata FIRST. DB is the source of truth: a DB failure here
        # leaves Qdrant completely untouched (no orphan vectors, no mismatch).
        # Previously Qdrant was upserted BEFORE commit, so a commit failure left
        # Qdrant holding vectors with no MySQL mapping — the bug we fix here.
        db.commit()
        logger.info(f"Committed DB metadata for document {doc_id_str}")

        # 4. Apply Qdrant changes AFTER the DB commit succeeds. If Qdrant fails,
        # run a compensating DB rollback so metadata never points at missing vectors.
        try:
            if to_delete:
                delete_vectors_by_ids(collection_name, to_delete)
            if to_upsert and vectors_payload:
                add_vector_status = add_vector(
                    collection_name=collection_name,
                    vectors=vectors_payload,
                )
                status_list.append({"action": "upsert", "status": add_vector_status})
        except Exception as qdrant_err:
            logger.error(
                f"Qdrant write failed AFTER DB commit for document {doc_id_str}; "
                f"running compensating DB rollback: {qdrant_err}"
            )
            _compensate_index_rollback(doc_id_str)
            raise

    except Exception as e:
        logger.error(f"Index transaction failed for document {doc_id_str}, rolling back changes: {e}")
        db.rollback()
        raise e
    finally:
        db.close()
        # Release the Redis distributed lock
        if lock:
            try:
                if lock.owned():
                    lock.release()
            except Exception as release_err:
                logger.error(f"Failed to release Redis lock for document {doc_id_str}: {release_err}")

    return status_list


def get_summarized_response(response):
    output = summarize_text(response)
    logger.info("Summarized response: %s", output)
    return output


@shared_task(**_AUTORETRY_KWARGS)
def bot_route_answer_message(history, question):
    return run_chat_graph(history, question)


@shared_task()
def save_episodic_memory_task(user_id: str, conversation_id: str):
    """
    Asynchronously extract key personal facts, situation, and preferences from conversation history,
    then save to MySQL and Qdrant.
    """
    try:
        from models import get_conversation_messages, save_user_episode
        from vectorize import add_vector
        import uuid

        messages = get_conversation_messages(conversation_id)
        if not messages or len(messages) < 2:
            logger.info("Not enough messages to summarize episodic memory.")
            return "skipped_empty"

        # Standard format conversation string for prompt
        conversation_str = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_str += f"{role.upper()}: {content}\n"

        summary_prompt = f"""Bạn là một AI phân tích thông tin hội thoại. Dưới đây là cuộc đối thoại giữa người dùng (USER) và trợ lý pháp luật (ASSISTANT):
---
{conversation_str}
---
Hãy tóm tắt ngắn gọn các chi tiết thực tế (facts) quan trọng về người dùng hoặc tình huống pháp lý riêng của họ được đề cập trong cuộc trò chuyện trên (ví dụ: họ có tài sản gì, đang tranh chấp ở đâu, đã đóng bảo hiểm bao nhiêu năm, nghề nghiệp gì).
QUAN TRỌNG:
- Chỉ tập trung vào thông tin của người dùng (USER). KHÔNG tóm tắt lời giải thích luật của ASSISTANT.
- Trả lời bằng các câu gạch đầu dòng ngắn gọn, súc tích (tiếng Việt).
- Nếu không có thông tin cá nhân/tình huống cụ thể nào của người dùng (chỉ hỏi luật chung chung), hãy trả lời duy nhất từ: "NONE".
"""

        summary_result = openai_chat_complete([
            {"role": "system", "content": "Bạn là chuyên gia trích xuất thông tin hội thoại pháp luật."},
            {"role": "user", "content": summary_prompt}
        ]).strip()

        if not summary_result or summary_result.upper() == "NONE":
            logger.info(f"No meaningful episodic memories extracted for user {user_id}.")
            return "skipped_none"

        logger.info(f"Extracted episodic memories for user {user_id}:\n{summary_result}")

        # Compute embedding once; reused for dedup check and Qdrant upsert.
        vector = get_embedding(summary_result)

        # Dedup: skip if a near-identical fact is already stored for this user
        # (e.g. "sinh năm 2004" saved every turn). High threshold to only catch
        # true duplicates, not related-but-distinct facts.
        try:
            existing = search_vector(
                collection_name="user_episodes",
                vector=vector,
                limit=1,
                filters={"user_id": user_id},
                score_threshold=0.88,
            )
            if existing:
                logger.info(f"Episodic dedup: similar fact already stored for user {user_id}, skipping save.")
                return "skipped_duplicate"
        except Exception as dedup_err:
            logger.warning(f"Episodic dedup check failed (proceeding to save): {dedup_err}")

        # Save to MySQL
        save_user_episode(user_id, summary_result)

        # Save to Qdrant. Deterministic point id (uuid5 of user_id + summary)
        # makes the upsert idempotent: if the dedup check was skipped (e.g. a
        # transient Qdrant error) and the same fact is saved again, the
        # second write overwrites the same point instead of creating a
        # duplicate (uuid4 would create a new point every turn).
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}|{summary_result}"))

        vectors_payload = {
            point_id: {
                "vector": vector,
                "payload": {
                    "user_id": user_id,
                    "summary": summary_result,
                    "conversation_id": conversation_id,
                    "content": summary_result  # Keep content to be compatible with search_vector payload format
                }
            }
        }

        add_vector(
            collection_name="user_episodes",
            vectors=vectors_payload
        )
        logger.info(f"Successfully saved episodic memory vector for user {user_id}")
        return "success"
    except Exception as e:
        logger.error(f"Failed in save_episodic_memory_task: {e}")
        return f"error: {str(e)}"


@shared_task()
def clear_user_runtime_caches_task(user_id: str, conversation_id: str | None = None) -> int:
    """Worker-side purge of in-process memory caches for a user.

    ``clear_user_runtime_caches`` only clears the caches in the process it
    runs in. The agent runs in the Celery worker, so a delete issued from the
    web process would not reach the worker's ``_conv_summaries`` /
    ``_agent_memories``. This task runs IN the worker so a history delete
    actually purges the rolling-summary + memory buffers the agent is using.
    """
    try:
        return clear_user_runtime_caches(user_id, conversation_id)
    except Exception as e:
        logger.error(f"Failed in clear_user_runtime_caches_task: {e}")
        return 0


@shared_task(bind=True)
def llm_handle_message(self, bot_id, user_id, question, role=None,
                       variant=None, shadow=False):
    """
    Main message handler with intelligent routing.

    Flow:
    1. Save user message to conversation history
    2. Load conversation context
    3. Route to appropriate handler (RAG, web search, or general chat)
    4. Generate and save response
    """
    logger.info("Start handle message")
    # Celery task id (None when called directly in the sync path). Used to map
    # task_id -> run_id so the SSE stream endpoint can resolve a run before the
    # Celery result is ready.
    celery_task_id = getattr(self, "request", None) and getattr(self.request, "id", None)

    # Ensure required tables exist before worker writes conversation records.
    # Guarded to run once per worker process (schema create is idempotent but
    # reflects all tables = a non-trivial DB roundtrip we don't want per message).
    # The FastAPI lifespan also runs this on startup; this covers a worker that
    # starts before the API ever received a request.
    _ensure_schema_once()

    # Update chat conversation
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)

    # 1. Check Semantic Cache BEFORE loading history. Cache lookup only needs
    # the raw question, not the conversation history, so on a HIT we skip the
    # extra DB read (get_conversation_messages) entirely.
    try:
        import sys
        import os
        _dir = os.path.dirname(os.path.abspath(__file__))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        from semantic_cache import get_cached_response, set_cached_response
        if "nocache" in (user_id or ""):
            cached = None
        else:
            cached = get_cached_response(question, user_id)
        if cached:
            logger.info("Semantic Cache HIT - returning cached response directly")
            update_chat_conversation(bot_id, user_id, cached["response"], False)
            return {
                "role": "assistant",
                "content": cached["response"],
                "sources": cached["sources"],
                "cached": True,
                "route": "cached"
            }
    except Exception as cache_err:
        logger.exception("Semantic Cache check failed")

    # Convert history to list messages (only needed on cache miss)
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[:-1]

    # 2. Run Input Guardrails check
    blocked_response = None
    if guardrails_manager.initialized:
        try:
            logger.info("Running input guardrails check...")
            blocked_response = run_async(guardrails_manager.verify_input(question))
        except Exception as e:
            logger.error(f"Error running input guardrails: {e}")
            
    if blocked_response:
        response_text = blocked_response
        sources = []
        run_id = None
        tool_calls = []
        logger.info("Chatbot response generated (Blocked by Input Guardrails)")
    else:
        # 3. Use LangGraph-based routing to handle the question.
        run_id = uuid.uuid4().hex
        emit_run_start(run_id, conversation_id, user_id, question)
        # Map celery task_id -> run_id (TTL 10m) so /chat/stream/{task_id} can
        # resolve the run and subscribe to its trace events before the Celery
        # result is ready. Best-effort: never block the chat on Redis.
        if celery_task_id:
            try:
                trace_redis = _get_trace_redis()
                trace_redis.setex(f"trace:run:{celery_task_id}", 600, run_id)
            except Exception as e:
                logger.warning(f"trace:run key set failed: {e}")
        # Retry transient infra failures (RedisSaver/DB/Qdrant hiccups) on the
        # graph invoke itself, NOT at Celery task level. Retrying here re-invokes
        # with the same thread_id, so LangGraph resumes from its last checkpoint
        # — no duplicate user message (the user msg was saved BEFORE the graph),
        # no duplicate reply. We do NOT use Celery autoretry for this task because
        # it would re-run the append-only user-message save and duplicate the turn.
        _run_graph_retried = with_retry(max_attempts=3, base_delay=1.0, max_delay=8.0)(run_chat_graph)
        graph_failed = False
        try:
            graph_result = _run_graph_retried(
                history, question, user_id=user_id,
                conversation_id=conversation_id, run_id=run_id, role=role,
                variant=variant, shadow=shadow,
            )
            response_text = graph_result.get("response", "")
            sources = graph_result.get("sources", [])
            tool_calls = graph_result.get("tool_calls", [])
            logger.info(f"Chatbot response generated")
            emit_run_end(
                run_id, conversation_id, status="completed",
                final_response=response_text,
                route=graph_result.get("route"),
                reflection_count=graph_result.get("reflection_count"),
                tool_calls_json=tool_calls if tool_calls else None,
            )
        except Exception as graph_err:
            # All retries exhausted (or a non-transient error). Degrade
            # gracefully: return a user-facing error instead of raising, so the
            # Celery task SUCCEEDS (no dead-task, no client exception) and the
            # user gets a clear retry prompt. Skip cache + episodic (an error
            # message must not be cached as an answer, nor extracted as a fact).
            graph_failed = True
            emit_run_end(run_id, conversation_id, status="error")
            logger.exception("Chat graph failed after retries; degrading gracefully: %s", graph_err)
            graph_result = {
                "response": (
                    "Xin lỗi, hệ thống đang tạm thời quá tải hoặc không kết nối được với "
                    "hạ tầng lưu trữ (Redis/DB/Qdrant). Vui lòng thử lại sau ít phút."
                ),
                "sources": [],
                "tool_calls": [],
                "route": "error",
                "reflection_count": 0,
                "verify_score": 0.0,
                "verify_verdict": "",
                "retry_verify": 0,
            }
            response_text = graph_result["response"]
            sources = []
            tool_calls = []

    # Save full response to history (to preserve all details like the specific age)
    update_chat_conversation(bot_id, user_id, response_text, False)

    # 4. Save to Semantic Cache if not blocked. Skip for shared sentinel
    # user_ids: the answer may carry this client's private facts, and caching
    # it under either user:<sentinel> or common would replay it to other
    # clients. No cache is better than a leaking cache here. Also skip when the
    # graph degraded to an error reply — never cache an error as an answer.
    if (
        not blocked_response
        and not graph_failed
        and (user_id or "").strip() not in _SHARED_USER_SENTINELS
        and "nocache" not in (user_id or "")
    ):
        try:
            import sys
            import os
            _dir = os.path.dirname(os.path.abspath(__file__))
            if _dir not in sys.path:
                sys.path.insert(0, _dir)
            from semantic_cache import set_cached_response, SCOPE_COMMON
            # Scope the cache per-user for privacy: legal_rag/agent_tools/web_search
            # answers may carry the user's private facts, so they must never leak to
            # another user. Only general_chat greetings opt into the shared
            # "common" scope to keep hit rate for non-private small talk.
            route = graph_result.get("route") if not blocked_response else None
            cache_scope = SCOPE_COMMON if route == "general_chat" else None
            set_cached_response(question, response_text, sources, user_id, scope=cache_scope)
        except Exception as cache_err:
            logger.exception("Semantic Cache save failed")

    # Trigger background episodic memory ingestion task asynchronously.
    # Skip when: (a) the input was blocked by guardrails — never persist a
    # blocked/jailbreak message as a "user fact"; (b) user_id is a shared
    # sentinel — writing would leak one client's private facts into the
    # shared bucket every other sentinel client reads from; (c) the graph
    # degraded to an error — an error reply carries no real user facts.
    if not blocked_response and not graph_failed and (user_id or "").strip() not in _SHARED_USER_SENTINELS:
        try:
            save_episodic_memory_task.delay(user_id, conversation_id)
            logger.info("Triggered save_episodic_memory_task asynchronously")
        except Exception as t_err:
            logger.error(f"Failed to trigger save_episodic_memory_task: {t_err}")

    # Return full response (tool_calls + run_id are additive optional keys;
    # sync /chat/complete callers ignore them, async poll passes them through).
    result = {"role": "assistant", "content": response_text, "sources": sources}
    if tool_calls:
        result["tool_calls"] = tool_calls
    if run_id:
        result["run_id"] = run_id
    if 'graph_result' in locals() and "route" in graph_result:
        result["route"] = graph_result["route"]
    elif blocked_response:
        result["route"] = "guardrails"

    # Flush Langfuse traces to ensure they are sent before the task terminates
    try:
        from agent import flush_langfuse
        flush_langfuse()
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse at task exit: {e}")

    return result