"""ReAct agent runtime: LLM, system prompt, per-conversation memory/cache, and
the ``ai_agent_handle`` Celery task.

Tool wrappers + FunctionTool instances live in ``agent_tool_wrappers.py``; the
per-run tool-call tracker (``agent_tool_calls``, ``track_tool_call``) lives in
``agent_tool_tracking.py``. Both are re-exported here so legacy callers that do
``from agent import ...`` / ``agent.track_tool_call`` keep working.
"""

import asyncio
import threading
import inspect
import logging
import os
import traceback

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List

from celery import shared_task
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, CustomLLM, CompletionResponse, ChatResponse, LLMMetadata, MessageRole
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

from retry_utils import with_retry, awith_retry

import config  # AGENT_MAX_ITERATIONS, AGENT_RUN_TIMEOUT_S — ReAct latency guards

# Re-export tracking symbols for back-compat (tests/eval import via `agent.*`).
from agent_tool_tracking import (  # noqa: F401
    agent_tool_calls,
    track_tool_call,
    agent_user_id,
    agent_run_id,
    agent_thread_id,
    agent_sources,
    agent_empty_streak,
    agent_prev_tool_args,
)
from citations import strip_trailing_references
from memory_short_term import (
    clear_short_term,
    get_rolling_summary,
    get_summarized_count,
    set_rolling_summary,
    set_summarized_count,
)

logger = logging.getLogger(__name__)

# Initialize the LLM LAZILY based on environment variables. Building at import
# time required env vars / network at import, which broke test collection and
# any importer that does not need the agent. We now build on first use via
# _get_ai_agent(user_id, conversation_id), which caches a per-conversation agent.
_llm = None

logger.info("LlamaIndex Agent LLM will be initialized lazily on first use")


class FallbackLLM(CustomLLM):
    primary_llm: object
    fallback_llm: object

    def __init__(self, primary_llm, fallback_llm):
        super().__init__(primary_llm=primary_llm, fallback_llm=fallback_llm)

    @property
    def metadata(self) -> LLMMetadata:
        return self.primary_llm.metadata

    @llm_chat_callback()
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        # Retry transient primary errors (429/5xx/timeout) before falling back,
        # so a brief rate-limit burst does NOT immediately switch providers.
        # Non-retryable errors skip retry and fall back at once.
        @with_retry(max_attempts=3, base_delay=1.0, max_delay=8.0)
        def _primary():
            logger.info(f"[FallbackLLM] Attempting chat using primary LLM ({type(self.primary_llm).__name__})...")
            return self.primary_llm.chat(messages, **kwargs)

        try:
            return _primary()
        except Exception as e:
            logger.warning(f"[FallbackLLM] Primary LLM failed: {e}. Falling back to ({type(self.fallback_llm).__name__})...")
            return self.fallback_llm.chat(messages, **kwargs)

    @llm_chat_callback()
    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        @awith_retry(max_attempts=3, base_delay=1.0, max_delay=8.0)
        async def _primary():
            logger.info(f"[FallbackLLM] Attempting async chat using primary LLM ({type(self.primary_llm).__name__})...")
            return await self.primary_llm.achat(messages, **kwargs)

        try:
            return await _primary()
        except Exception as e:
            logger.warning(f"[FallbackLLM] Primary LLM failed: {e}. Falling back to ({type(self.fallback_llm).__name__})...")
            return await self.fallback_llm.achat(messages, **kwargs)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        @with_retry(max_attempts=3, base_delay=1.0, max_delay=8.0)
        def _primary():
            logger.info(f"[FallbackLLM] Attempting complete using primary LLM ({type(self.primary_llm).__name__})...")
            return self.primary_llm.complete(prompt, **kwargs)

        try:
            return _primary()
        except Exception as e:
            logger.warning(f"[FallbackLLM] Primary LLM failed: {e}. Falling back to ({type(self.fallback_llm).__name__})...")
            return self.fallback_llm.complete(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        @awith_retry(max_attempts=3, base_delay=1.0, max_delay=8.0)
        async def _primary():
            logger.info(f"[FallbackLLM] Attempting async complete using primary LLM ({type(self.primary_llm).__name__})...")
            return await self.primary_llm.acomplete(prompt, **kwargs)

        try:
            return await _primary()
        except Exception as e:
            logger.warning(f"[FallbackLLM] Primary LLM failed: {e}. Falling back to ({type(self.fallback_llm).__name__})...")
            return await self.fallback_llm.acomplete(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        """Stream chat via primary LLM, fall back on error.

        Workflow ReActAgent uses streaming (_get_streaming_response), so this
        MUST be implemented - raising NotImplementedError crashes every agent run.
        """
        try:
            logger.info(f"[FallbackLLM] Streaming chat via primary ({type(self.primary_llm).__name__})...")
            yield from self.primary_llm.stream_chat(messages, **kwargs)
        except Exception as e:
            logger.warning(f"[FallbackLLM] Primary stream failed: {e}. Falling back to ({type(self.fallback_llm).__name__})...")
            yield from self.fallback_llm.stream_chat(messages, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        try:
            logger.info(f"[FallbackLLM] Streaming complete via primary ({type(self.primary_llm).__name__})...")
            yield from self.primary_llm.stream_complete(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"[FallbackLLM] Primary stream failed: {e}. Falling back to ({type(self.fallback_llm).__name__})...")
            yield from self.fallback_llm.stream_complete(prompt, **kwargs)


def _build_llm():
    """Build the LlamaIndex LLM from env vars. Returns None if config missing.

    ALWAYS wraps the chosen provider in FallbackLLM so transient call errors
    (429 rate limits, 5xx, connection/timeout) are retried with exponential
    backoff before surfacing — even when only ONE provider is configured. With a
    single provider, primary==fallback, so the wrapper's value is the retry, not
    provider switching. With Ollama-as-main + Groq configured, primary=Ollama /
    fallback=Groq gives genuine cross-provider failover on top of the retry.
    """
    # 1. Build Groq (as fallback or primary)
    from llama_index.llms.groq import Groq
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    if groq_model and "cloud" in groq_model.lower():
        groq_model = "llama-3.3-70b-versatile"
    groq_llm = None
    if groq_api_key:
        groq_llm = Groq(model=groq_model, api_key=groq_api_key, temperature=0.1)

    # 2. Build Ollama
    from llama_index.llms.ollama import Ollama
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")
    ollama_key = os.environ.get("OLLAMA_API_KEY")
    ollama_model = os.environ.get("OLLAMA_LLM_MODEL", "glm-5.2:cloud")
    headers = {"Authorization": f"Bearer {ollama_key}"} if ollama_key else None
    ollama_llm = Ollama(
        model=ollama_model,
        base_url=ollama_url,
        temperature=0.1,
        request_timeout=30.0,
        headers=headers
    )

    # Check routing
    use_ollama_as_main = os.environ.get("USE_OLLAMA_AS_MAIN", "false").lower() == "true"

    def _single(llm):
        """Wrap a lone provider as FallbackLLM(primary=llm, fallback=llm).

        primary==fallback is intentional: there is no second provider to switch
        to, so the point of the wrapper is NOT provider fallback — it is the
        transient-call retry (``with_retry``/``awith_retry`` baked into
        FallbackLLM.chat/complete/achat/acomplete). On exhaustion the "fallback"
        simply re-runs the same provider once more, i.e. a second retry burst.
        Returns None when ``llm`` is None (no provider configured at all).
        """
        if llm is None:
            return None
        logger.info(f"Using FallbackLLM (single-provider retry): {type(llm).__name__}")
        return FallbackLLM(primary_llm=llm, fallback_llm=llm)

    if use_ollama_as_main:
        if groq_llm:
            logger.info("Using FallbackLLM: Primary = Ollama, Fallback = Groq")
            return FallbackLLM(primary_llm=ollama_llm, fallback_llm=groq_llm)
        logger.info("Ollama is main, but Groq is not configured. Using Ollama only (retry-wrapped).")
        return _single(ollama_llm)

    # Default routing behavior based on LLM_PROVIDER
    llm_provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if llm_provider == "ollama":
        return _single(ollama_llm)
    elif llm_provider == "openai":
        from llama_index.llms.openai import OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        openai_base = os.environ.get("OPENAI_API_BASE")
        openai_llm = OpenAI(
            model=os.environ.get("LLM_MODEL"),
            api_key=openai_key,
            api_base=openai_base,
            temperature=0.1,
        )
        return _single(openai_llm)
    else:
        # Default provider = Groq. Return None (not a wrapped None) when no key
        # is configured so callers' `if _llm is None` guard still works.
        if groq_llm is None:
            return None
        return _single(groq_llm)


def _get_agent_system_prompt(history_summary: str = "", user_id: Optional[str] = None) -> str:
    from datetime import datetime
    now = datetime.now()
    current_year = now.year
    current_date = now.strftime("%d/%m/%Y")
    # LLMOps Tier 1.2: system prompt loaded from backend/prompts/agent_system.<ver>.txt
    # Version = PROMPT_AGENT_SYSTEM env > models.yaml prompts.agent_system > "v1".
    from prompt_loader import load_prompt
    base = load_prompt("agent_system", current_date=current_date, current_year=current_year)

    if history_summary:
        base += (
            "\nNGỮ CẢNH HỘI THOẠI TRƯỚC (CHỈ THAM KHẢO - KHÔNG PHẢI CÂU HỎI HIỆN TẠI):\n"
            f"{history_summary}\n"
            "\nDùng ngữ cảnh trên chỉ để bổ sung факт nền (năm sinh, giới tính, tình huống). "
            "Câu hỏi chính là câu mới nhất của người dùng - trả lời trực tiếp câu đó."
        )

    # P4b — Procedural memory: inject the per-case-type workflow scaffold when
    # the user's case_type is known. Cheap KV lookup (UserProfile), not a tool.
    # Gate behind PROCEDURAL_WORKFLOW_ENABLED (default true) for rollback.
    if os.environ.get("PROCEDURAL_WORKFLOW_ENABLED", "true").lower() == "true" and user_id:
        try:
            from procedural_memory import workflow_block
            from models import get_user_profile
            profile = get_user_profile(user_id)
            if profile:
                base += workflow_block(profile.get("case_type"))
        except Exception as wf_err:
            logger.warning(f"[AGENT] Procedural workflow block injection skipped: {wf_err}")
    return base


# Per-conversation agent memory (Phase E). Previously a single global
# ChatMemoryBuffer was shared across ALL users/conversations - a cross-user
# data leak. Now keyed by (user_id, conversation_id) with an LRU cap.
_agent_memories: "OrderedDict[Tuple[Optional[str], Optional[str]], ChatMemoryBuffer]" = OrderedDict()
_AGENT_MEMORY_CAP = 32


def get_agent_memory(user_id: Optional[str], conversation_id: Optional[str]) -> ChatMemoryBuffer:
    """Return (creating if needed) the ChatMemoryBuffer for a (user, conversation).

    LRU-evicts the oldest entry once the cap is exceeded. Thread-safe via
    ``_memory_lock`` so concurrent runs cannot corrupt the LRU order.
    """
    key = (user_id, conversation_id)
    with _memory_lock:
        mem = _agent_memories.get(key)
        if mem is None:
            mem = ChatMemoryBuffer.from_defaults(token_limit=8192)
            _agent_memories[key] = mem
            _agent_memories.move_to_end(key)
            if len(_agent_memories) > _AGENT_MEMORY_CAP:
                evicted_key, _ = _agent_memories.popitem(last=False)
                logger.info(f"[AGENT] Evicted oldest agent memory for key={evicted_key}")
        else:
            # LRU: touch on read so a long-lived, frequently-read conversation
            # is not evicted while cold conversations occupy the tail.
            _agent_memories.move_to_end(key)
    return mem


def clear_user_runtime_caches(user_id: Optional[str], conversation_id: Optional[str] = None) -> int:
    """Drop in-process + Redis memory entries for a user (and optional conversation).

    Clears the rolling-summary/count cache (Redis-backed, centralized) and the
    legacy ChatMemoryBuffer for the given (user_id, conversation_id) — or ALL
    conversations for that user when ``conversation_id`` is None. Returns the
    number of cache entries removed.

    The rolling summary now lives in Redis (see ``memory_short_term``), so a
    clear from the web process also clears what the Celery worker reads. The
    worker's local LRU read-through cache is dropped by dispatching
    ``clear_user_runtime_caches_task`` (runs in the worker).
    """
    removed = 0
    # Legacy ChatMemoryBuffer (in-process only, used by the from_tools branch).
    with _memory_lock:
        keys_to_drop = [
            (u, c) for (u, c) in list(_agent_memories.keys())
            if u == user_id and (conversation_id is None or c == conversation_id)
        ]
        for k in keys_to_drop:
            _agent_memories.pop(k, None)
            removed += 1
    # Rolling summary + count (Redis-backed, cross-worker).
    removed += clear_short_term(user_id, conversation_id)
    logger.info(
        f"[AGENT] Cleared runtime caches for user={user_id} "
        f"conv={conversation_id} entries_removed={removed}"
    )
    return removed


# Per-conversation rolling summary of chat history. Injected into the agent
# system prompt as grounding context so the agent stays focused on the CURRENT
# question while still knowing prior facts (birth year, gender, situation).
#
# The summary + summarized-count now live in Redis (cross-worker source of
# truth) — see ``memory_short_term``. They used to be module-level OrderedDicts,
# which were RAM-local to one process and caused multi-worker drift ("amnesia"
# across turns). The ``_memory_lock`` below still guards the legacy
# ``_agent_memories`` ChatMemoryBuffer (from_tools branch only).

# Guards the legacy ``_agent_memories`` OrderedDict: ai_agent_handle runs in a
# thread (asyncio.to_thread in the web process, or a Celery thread/gevent
# pool) and CPython OrderedDict mutations (move_to_end/popitem/pop/__setitem__)
# are not atomic, so concurrent runs could corrupt LRU order or lose entries.
_memory_lock = threading.Lock()

# Memory strategy constants (senior-grade hybrid: short-term raw + rolling
# structured summary + long-term RAG recall via recall_user_memory_tool).
_SHORT_TERM_RAW_TURNS = 2   # last N prior turns passed verbatim as chat_history
_RESUMMARY_INTERVAL = 10    # full re-summarize from raw every N turns to fix drift
_MSG_CHAR_CAP = 4000        # hard per-message char cap to bound token budget


def _truncate_raw(text: str, limit: int = 600) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _history_to_chat_messages(history: Optional[List[Dict]]) -> List[ChatMessage]:
    """Convert DB history (list of {role, content} dicts) to ChatMessage list.

    Skips system messages (the agent has its own system prompt). Hard-truncates
    each message content to ``_MSG_CHAR_CAP`` to bound token budget.
    """
    if not history:
        return []
    msgs: List[ChatMessage] = []
    for d in history:
        if not isinstance(d, dict):
            continue
        role_str = str(d.get("role", "user")).lower()
        if role_str == "system":
            continue  # system msg is agent's own system prompt, not conversation history
        if role_str == "assistant":
            role = MessageRole.ASSISTANT
        else:
            role = MessageRole.USER
        content = str(d.get("content", ""))
        if len(content) > _MSG_CHAR_CAP:
            content = content[:_MSG_CHAR_CAP] + "..."
        msgs.append(ChatMessage(role=role, content=content))
    return msgs


def _summarize_chat_history(
    history: Optional[List[Dict]],
    user_id: Optional[str],
    conversation_id: Optional[str],
) -> Tuple[str, List[ChatMessage]]:
    """Build grounding context from PRIOR conversation turns (hybrid memory).

    Takes ``history`` = list of ``{role, content}`` dicts from the DB (source of
    truth). The in-process ChatMemoryBuffer is NOT used for read — it was never
    populated (generate_agent_answer passed history but the old code ignored it),
    so relying on it made short-term/summary dead silent.

    Returns ``(rolling_summary, short_term_raw)``:

    - ``short_term_raw``: last ``_SHORT_TERM_RAW_TURNS`` prior turns verbatim.
      Passed as ``chat_history`` so the agent keeps reference-resolution ability
      ("việc đó", "như đã nói") for the immediately preceding turns.
    - ``rolling_summary``: structured LLM summary (FACTS + TOPICS) of OLDER turns
      beyond the raw window. Threshold-triggered (only when older turns exist),
      incremental, with periodic full re-summary every ``_RESUMMARY_INTERVAL``
      turns to fix accumulated drift. Injected into the system prompt as
      grounding context only.

    First turn (no prior history) -> ``("", [])`` (no LLM call).
    """
    # history is PRIOR turns only (current question is passed separately as
    # user_msg and not yet in DB), so no trailing-drop needed.
    prior_msgs = _history_to_chat_messages(history)
    if not prior_msgs:
        return "", []

    short_term_raw = prior_msgs[-_SHORT_TERM_RAW_TURNS:]
    old_turns = prior_msgs[:-_SHORT_TERM_RAW_TURNS]

    # No older turns beyond the raw window -> nothing to summarize.
    if not old_turns:
        return "", short_term_raw

    # Rolling summary + count now live in Redis (cross-worker). Replaces the
    # old in-process OrderedDicts that drifted across uvicorn/Celery workers.
    already = get_summarized_count(user_id, conversation_id)
    prev_summary = get_rolling_summary(user_id, conversation_id)
    # Periodic full re-summary from raw to correct incremental drift.
    full_resummary = len(prior_msgs) % _RESUMMARY_INTERVAL == 0
    needs_update = full_resummary or len(old_turns) > already

    if not needs_update:
        # No change -> return the cached summary verbatim. Redis is the source
        # of truth; no in-process LRU touch needed (memory_short_term caches it).
        return prev_summary, short_term_raw
    if full_resummary:
        # Re-summarize ALL older turns from raw (drift correction).
        new_turn_text = "\n".join(
            f"{getattr(m.role, 'value', m.role)}: {m.content}" for m in old_turns
        )
        base_summary = "(chưa có)"
        logger.info(f"[AGENT] Full re-summary from {len(old_turns)} old turns (drift fix)")
    else:
        new_msgs = old_turns[already:]
        new_turn_text = "\n".join(
            f"{getattr(m.role, 'value', m.role)}: {m.content}" for m in new_msgs
        )
        base_summary = prev_summary or "(chưa có)"

    global _llm
    if _llm is None:
        _llm = _build_llm()
    if _llm is None:
        return prev_summary or _truncate_raw(new_turn_text), short_term_raw

    prompt = (
        "Bạn là bộ tóm tắt hội thoại pháp lý. Tóm tắt CẤU TRÚC, NGẮN (<200 từ) các lượt hội thoại.\n"
        "Format BẮT BUỘC:\n"
        "FACTS:\n"
        "- <fact nền của user dạng key-value: năm sinh, giới tính, nghề, tình huống pháp lý — không lan man>\n"
        "TOPICS:\n"
        "- <chủ đề đã hỏi + kết luận ngắn 1 dòng>\n\n"
        "Quy tắc:\n"
        "- Dedup: không lặp fact đã có trong tóm tắt trước.\n"
        "- Bỏ câu trả lời dài, chỉ giữ kết luận chính.\n"
        "- Giữ nguyên giá trị sự thật (năm, số tiền, tên luật, số điều).\n\n"
        f"Tóm tắt trước đó:\n{base_summary}\n\n"
        f"Các lượt hội thoại cần tóm tắt:\n{new_turn_text}\n\n"
        "Tóm tắt cập nhật (FACTS + TOPICS):"
    )
    try:
        resp = _llm.complete(prompt)
        summary = str(getattr(resp, "text", resp) or "").strip()
        if summary:
            # Persist to Redis (cross-worker). LRU cap + TTL handled in
            # memory_short_term; no manual eviction needed here.
            set_rolling_summary(user_id, conversation_id, summary)
            set_summarized_count(user_id, conversation_id, len(old_turns))
        return summary or prev_summary, short_term_raw
    except Exception as e:
        logger.warning(f"[AGENT] History summary failed: {e}")
        return prev_summary or _truncate_raw(new_turn_text), short_term_raw


def filter_tools_for_query(
    query: str,
    history: Optional[List[Dict]] = None,
    role: Optional[str] = None,
) -> list:
    """Select a subset of tools relevant to the user's query to minimize prompt tokens.

    Phase 2b — when ``role`` is supplied, the result is further narrowed by
    RBAC tool policy (``rbac.filter_tools_by_policy``) so a guest never sees
    web/search/sensitive tools in the agent's prompt, and a user never sees
    admin-only escalation paths. ``role=None`` keeps legacy behavior (no
    policy filter) for the anonymous demo path.

    Phase 3 — semantic tool router (``tool_router.select_tools_semantic``):
    embed query↔tool-description top-k cosine, robust to paraphrase. Tried
    FIRST; on ``None`` (disabled / embedding service down / empty result) it
    falls through to the keyword path below, so behavior degrades gracefully
    to the pre-Phase-3 contract with zero regression risk.
    """
    try:
        from tool_router import select_tools_semantic
        semantic = select_tools_semantic(query, history=history)
        if semantic:
            return _apply_role_policy(semantic, role)
    except Exception as exc:
        logger.warning("[AGENT] semantic tool router failed, using keyword path: %s", exc)

    query_lower = query.lower()

    # Accumulate query and last conversation turns to find relevant keywords
    scan_text = query_lower
    has_prior_history = False
    if history:
        try:
            history_msgs = _history_to_chat_messages(history)
            has_prior_history = len(history_msgs) > 1
            if history_msgs:
                # Scan last 2 messages for keyword context
                for msg in history_msgs[-2:]:
                    scan_text += " " + str(msg.content).lower()
        except Exception as e:
            logger.warning(f"Failed to load history for tool filtering: {e}")

    selected = []
    
    # 1. Legal calculation tools
    if any(kw in scan_text for kw in ["phạt", "chậm trễ", "vi phạm", "trễ hạn", "bồi thường hợp đồng"]):
        selected.append("contract_penalty_tool")
    if any(kw in scan_text for kw in ["tuổi", "sinh năm", "năm sinh", "kết hôn", "lao động", "hình sự", "bao nhiêu tuổi"]):
        selected.append("legal_age_tool")
    if any(kw in scan_text for kw in ["thừa kế", "di sản", "chia", "tài sản cha mẹ", "để lại"]):
        selected.append("inheritance_tool")
    if any(kw in scan_text for kw in ["doanh nghiệp", "tên", "đặt tên", "công ty"]):
        selected.append("business_name_tool")
    if any(kw in scan_text for kw in ["thời hiệu", "khởi kiện", "khiếu nại", "hết hạn kiện"]):
        selected.append("statute_tool")
        
    # 2. Knowledge / data-driven legal tools
    if any(kw in scan_text for kw in ["thôi việc", "sa thải", "trợ cấp", "bị đuổi", "đuổi việc", "cho nghỉ", "mất việc", "nghỉ việc"]):
        selected.append("severance_pay_func_tool")
    if any(kw in scan_text for kw in ["làm thêm", "tăng ca", "ot", "giờ", "lương đêm", "làm ngày lễ"]):
        selected.append("overtime_pay_func_tool")
    if any(kw in scan_text for kw in ["thuế", "tncn", "thu nhập", "giảm trừ"]):
        selected.append("pit_monthly_func_tool")
    if any(kw in scan_text for kw in ["trước bạ", "đất đai", "nhà đất", "lệ phí", "sang tên", "sổ đỏ"]):
        selected.append("land_registration_fee_func_tool")
        selected.append("vehicle_registration_fee_func_tool")
    if any(kw in scan_text for kw in ["án phí", "tòa án", "lệ phí tòa", "tiền án phí"]):
        selected.append("court_fee_func_tool")
    if any(kw in scan_text for kw in ["phạt hành chính", "vi phạm", "giao thông", "nồng độ cồn", "vượt đèn đỏ"]):
        selected.append("admin_fine_lookup_func_tool")
    if any(kw in scan_text for kw in ["cấp dưỡng", "nuôi con", "ly hôn", "chia tài sản ly hôn"]):
        selected.append("child_support_func_tool")
    if any(kw in scan_text for kw in ["thủ tục", "quy trình", "bước", "hướng dẫn làm"]):
        selected.append("procedure_wizard_func_tool")
    if any(kw in scan_text for kw in ["thẩm quyền", "tòa án nào", "thụ lý", "nộp đơn ở đâu"]):
        selected.append("jurisdiction_resolver_func_tool")
    if any(kw in scan_text for kw in ["mẫu đơn", "văn bản mẫu", "hợp đồng mẫu", "mẫu hợp đồng"]):
        selected.append("generate_document_template_func_tool")
    if any(kw in scan_text for kw in ["hiệu lực", "phiên bản", "sửa đổi", "bổ sung", "còn dùng không"]):
        selected.append("law_version_func_tool")
        
    # 3. Retrieval-backed legal tools (always useful for general legal questions)
    if any(kw in scan_text for kw in ["điều", "khoản", "luật", "án lệ", "trích dẫn"]):
        selected.append("article_lookup_func_tool")
        selected.append("precedent_lookup_func_tool")
        selected.append("cross_reference_func_tool")
        selected.append("verify_citation_func_tool")

    # 3b. Neo4j graph recall — multi-hop citation traversal (Phase 3). Better
    # fit than vector search for "điều nào dẫn chiếu", "còn hiệu lực không",
    # "bác bỏ bởi luật nào", or enumerating a statute's articles.
    if any(kw in scan_text for kw in ["dẫn chiếu", "dẫn chứng", "bác bỏ",
                                     "còn hiệu lực", "điều nào", "tham chiếu",
                                     "chuỗi dẫn chiếu"]):
        selected.append("recall_legal_graph_func_tool")
        
    # Always include safety disclaimer (crucial for legal agent)
    selected.append("legal_disclaimer_func_tool")

    # 4. Search and Utility tools
    selected.append("tavily_tool")
    selected.append("current_time_tool")

    # 5. Long-term memory recall — include when the query references prior
    # context OR there is conversation history. Agent decides whether to call.
    recall_keywords = ["việc đó", "như đã nói", "câu trước", "trước đó", "lúc nãy",
                       "còn nhớ", "anh/chị còn nhớ", "vừa rồi", "nhắc lại", "lúc trước"]
    if has_prior_history or any(kw in scan_text for kw in recall_keywords):
        selected.append("recall_user_memory_func_tool")

    # Import dynamically to avoid circular imports
    import agent_tool_wrappers

    tools_list = []
    for name in selected:
        tool_obj = getattr(agent_tool_wrappers, name, None)
        if tool_obj:
            tools_list.append(tool_obj)

    # Fallback: if no specific tools matched, return a sensible default set.
    # Retrieval-first — most unmatched legal questions are lookup-style ("xin
    # chào", "cho hỏi chút"), not arithmetic. The old calc trio
    # (contract_penalty/legal_age/inheritance) was useless for a generic query
    # and starved the agent of any retrieval capability. unified_legal_search
    # fans out article_lookup + cross_reference + precedent in one call.
    if not tools_list or len(tools_list) <= 3:
        fallback = [
            getattr(agent_tool_wrappers, "article_lookup_func_tool"),
            getattr(agent_tool_wrappers, "unified_legal_search_func_tool"),
            getattr(agent_tool_wrappers, "legal_disclaimer_func_tool"),
            getattr(agent_tool_wrappers, "tavily_tool"),
            getattr(agent_tool_wrappers, "current_time_tool"),
        ]
        if has_prior_history:
            fallback.append(getattr(agent_tool_wrappers, "recall_user_memory_func_tool"))
        return _apply_role_policy(fallback, role)

    logger.info(f"[AGENT] Dynamically selected {len(tools_list)} tools for prompt reduction (out of 28)")
    return _apply_role_policy(tools_list, role)


def _apply_role_policy(tools: list, role: Optional[str]) -> list:
    """Narrow ``tools`` by RBAC policy when a role is supplied (Phase 2b).

    No-op for ``role=None`` (legacy anonymous path). A principal with no
    matching tools after filtering returns an empty list; callers should fall
    back to a non-tool generation path rather than run a toolless ReAct agent.
    """
    if not role:
        return tools
    try:
        from rbac import Principal, Role, filter_tools_by_policy

        principal = Principal(user_id="", username="", role=role)
        kept = filter_tools_by_policy(tools, principal)
        if len(kept) < len(tools):
            logger.info(
                f"[AGENT] RBAC policy (role={role}) kept {len(kept)}/{len(tools)} tools"
            )
        return kept
    except Exception as exc:
        logger.warning(f"[AGENT] RBAC policy filter failed (proceeding unfiltered): {exc}")
        return tools


def _build_react_agent(
    llm,
    memory: ChatMemoryBuffer,
    tools: list,
    history_summary: str = "",
    user_id: Optional[str] = None,
) -> ReActAgent | None:
    """Build ReAct agent with compatibility across llama-index versions."""
    if llm is None:
        return None

    prompt = _get_agent_system_prompt(history_summary=history_summary, user_id=user_id)

    # Older versions expose from_tools, newer workflow-based versions use constructor.
    if hasattr(ReActAgent, "from_tools"):
        return ReActAgent.from_tools(
            tools,
            llm=llm,
            memory=memory,
            verbose=True,
            max_iterations=config.AGENT_MAX_ITERATIONS,
            context=prompt,
        )

    # Newer workflow-based ReActAgent (llama_index >=0.12): __init__ does NOT
    # accept `memory` or `max_iterations`. Memory/chat_history must be passed
    # to run()/arun() at call time. Passing them here silently lands in
    # **kwargs and the agent runs with no memory -> empty exception.
    return ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=prompt,
        verbose=True,
    )


@shared_task()
def ai_agent_handle(
    question: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    history: Optional[List[Dict]] = None,
    role: Optional[str] = None,
    run_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Handle user question using ReAct agent with tools.

    Args:
        question: User's question
        user_id: Optional user id for per-conversation memory isolation.
        conversation_id: Optional conversation id for per-conversation memory.
        history: Prior conversation turns as list of ``{role, content}`` dicts
            from the DB (source of truth). Used to build short-term raw context
            + rolling summary. The current question is NOT in history.

    Returns:
        ``(response_text, tool_calls, sources)`` where tool_calls is a list of
        ``{tool_name, args, status, error, result}`` dicts captured during this
        run via the ``agent_tool_calls`` contextvar, and sources is the list of
        source chunks surfaced by retrieval tools (article_lookup /
        cross_reference / verify_citation / precedent_lookup) captured via the
        ``agent_sources`` contextvar — handed to verify_answer so the agent
        route no longer carries ``sources=[]`` and skips citation groundedness.
    """
    # Reset the tool-call accumulator for this run so @track_tool_call records
    # only this turn's calls. contextvars propagate into asyncio.run coroutines.
    token = agent_tool_calls.set([])
    user_token = agent_user_id.set(user_id)
    # Reset the source accumulator so retrieval tools record only this turn's
    # sources (see agent_tool_tracking.record_agent_source).
    src_token = agent_sources.set([])
    # Reset the empty-result no-retry guard state so streak/prev-args start clean
    # (see agent_tool_tracking.should_block_repeat / mark_tool_empty).
    streak_token = agent_empty_streak.set(0)
    prev_args_token = agent_prev_tool_args.set({})
    # Trace identity for per-tool-call events (None outside a graph run => silent).
    rid_token = agent_run_id.set(run_id)
    tid_token = agent_thread_id.set(thread_id)
    try:
        global _llm
        if _llm is None:
            _llm = _build_llm()
        if _llm is None:
            return (
                "Xin lỗi, hệ thống AI agent chưa sẵn sàng do thiếu cấu hình GROQ_API_KEY. "
                "Vui lòng kiểm tra biến môi trường và thử lại.",
                [],
                [],
            )

        # memory is kept only for the legacy from_tools branch of _build_react_agent.
        # The workflow ReActAgent (llama_index >=0.12) ignores it; history
        # (DB-sourced) drives short-term raw + rolling summary instead. Only
        # populate the buffer when the legacy branch will actually read it, so
        # we don't mutate/evict _agent_memories for a buffer that is never used.
        memory = get_agent_memory(user_id, conversation_id) if hasattr(ReActAgent, "from_tools") else None

        # Hybrid memory: short-term raw (last turns verbatim) + rolling structured
        # summary of older turns (grounding only). The current question is the
        # main user_msg; short_term_raw keeps reference-resolution for the
        # immediately preceding turns; older context is condensed in the summary.
        history_summary, short_term_raw = _summarize_chat_history(history, user_id, conversation_id)
        if history_summary:
            logger.info(
                f"[AGENT] Grounding summary injected ({len(history_summary)} chars); "
                f"short-term raw turns={len(short_term_raw)}"
            )
        elif short_term_raw:
            logger.info(f"[AGENT] Short-term raw turns={len(short_term_raw)} (no summary needed yet)")

        # Select tools dynamically based on current query and conversation history
        tools = filter_tools_for_query(question, history, role=role)

        ai_agent = _build_react_agent(_llm, memory, tools, history_summary=history_summary, user_id=user_id)
        if ai_agent is None:
            return (
                "Xin lỗi, lỗi khi khởi tạo hệ thống AI Agent.",
                [],
                [],
            )

        logger.info(f"[AGENT] Processing question: {question}")

        # Robust execution wrapper to handle different LlamaIndex versions and event loops
        async def _safe_execute_agent():
            # chat_history = short_term_raw (last turns verbatim) for reference
            # resolution. Older context is condensed in the system prompt summary.
            # The current question is the main user_msg -> agent stays focused.
            run_kwargs = {
                "user_msg": question,
                "chat_history": short_term_raw,
                "max_iterations": config.AGENT_MAX_ITERATIONS,
            }

            # Try async methods first (modern llama-index)
            # Newer ReActAgent (Workflow) uses run/arun which often returns a WorkflowHandler
            if hasattr(ai_agent, "arun"):
                # Workflow arun accepts user_msg + chat_history kwargs.
                try:
                    result = await ai_agent.arun(**run_kwargs)
                except Exception:
                    # Fallback: positional call (user_msg, chat_history)
                    try:
                        result = await ai_agent.arun(question, chat_history)
                    except Exception:
                        result = await ai_agent.arun(question)

                # If result is a WorkflowHandler, we need to await it to get the actual response
                if inspect.isawaitable(result) or hasattr(result, "__await__"):
                    result = await result
                return result

            if hasattr(ai_agent, "achat"):
                return await ai_agent.achat(question)

            # Sync fallbacks
            if hasattr(ai_agent, "run"):
                try:
                    result = ai_agent.run(**run_kwargs)
                except Exception:
                    try:
                        result = ai_agent.run(question, chat_history)
                    except Exception:
                        result = ai_agent.run(question)
                if inspect.isawaitable(result) or hasattr(result, "__await__"):
                    result = await result
                return result

            if hasattr(ai_agent, "chat"):
                return ai_agent.chat(question)

            raise AttributeError(f"Agent {type(ai_agent)} has no execution method (run/arun/chat/achat)")

        try:
            # Use asyncio.run for clean loop management in each Celery task.
            # Wrap in asyncio.wait_for with AGENT_RUN_TIMEOUT_S so a thrashing
            # ReAct loop (empty-result retries + reasoning tokens) cannot lũy kế
            # past the cap. On timeout the agent bails to a graceful fallback
            # instead of compounding to 60-90s.
            logger.info(
                f"[AGENT] Running agent (Type: {type(ai_agent)}, "
                f"max_iter={config.AGENT_MAX_ITERATIONS}, "
                f"timeout={config.AGENT_RUN_TIMEOUT_S}s)"
            )
            result = asyncio.run(
                asyncio.wait_for(_safe_execute_agent(), timeout=config.AGENT_RUN_TIMEOUT_S)
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[AGENT] Timed out after {config.AGENT_RUN_TIMEOUT_S}s — returning "
                f"graceful fallback (empty-retry thrash capped)"
            )
            tool_calls = list(agent_tool_calls.get() or [])
            sources = list(agent_sources.get() or [])
            fallback = (
                "Xin lỗi, quá trình tra cứu mất nhiều thời gian và chưa trả được kết quả "
                "trong thời gian cho phép. Vui lòng đặt câu hỏi cụ thể hơn (ghi rõ tên luật, "
                "số điều) hoặc thử lại sau."
            )
            return fallback, tool_calls, sources
        except Exception as e:
            if "already running" in str(e).lower():
                logger.info("[AGENT] Loop already running, applying nest_asyncio")
                import nest_asyncio
                nest_asyncio.apply()
                # Use current loop — keep the same timeout cap on the fallback path.
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    asyncio.wait_for(_safe_execute_agent(), timeout=config.AGENT_RUN_TIMEOUT_S)
                )
            else:
                raise e

        logger.info(f"[AGENT] Response generated successfully")

        # Extract text content and ensure it's a clean string (no 'assistant:' prefix)
        res_text = ""
        if isinstance(result, str):
            res_text = result
        elif hasattr(result, "response"):
            # Result is often an AgentOutput/Response object
            inner = result.response
            if hasattr(inner, "content"):
                res_text = str(inner.content)
            elif hasattr(inner, "text"):
                res_text = str(inner.text)
            else:
                res_text = str(inner)
        elif hasattr(result, "content"):
            res_text = str(result.content)
        else:
            res_text = str(result)

        # Clean up common prefixes and trailing reference lists
        if res_text.startswith("assistant: "):
            res_text = res_text[len("assistant: "):]
        res_text = strip_trailing_references(res_text)

        logger.info(f"[AGENT] Clean response extracted (length: {len(res_text)})")
        tool_calls = list(agent_tool_calls.get() or [])
        sources = list(agent_sources.get() or [])
        logger.info(f"[AGENT] Captured {len(tool_calls)} tool calls, {len(sources)} sources")
        return res_text, tool_calls, sources
    except Exception as e:
        logger.error(
            f"[AGENT] Error: type={type(e).__name__} msg={e!r} "
            f"traceback={traceback.format_exc()}"
        )
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi ({type(e).__name__}).", [], []
    finally:
        agent_tool_calls.reset(token)
        agent_user_id.reset(user_token)
        agent_run_id.reset(rid_token)
        agent_thread_id.reset(tid_token)
        agent_sources.reset(src_token)
        agent_empty_streak.reset(streak_token)
        agent_prev_tool_args.reset(prev_args_token)


def get_agent_tools_summary() -> Dict:
    """Get summary of available agent tools"""
    return {
        "total_tools": len(all_tools),
        "legal_calc_tools": [
            "contract_penalty_calculator - Tính phạt hợp đồng",
            "legal_age_checker - Kiểm tra tuổi pháp lý",
            "inheritance_calculator - Tính thừa kế",
            "business_name_validator - Kiểm tra tên DN",
            "statute_lookup - Tra cứu thời hiệu",
            "severance_pay_tool - Trợ cấp thôi việc (Đ48 BLLĐ 2019)",
            "overtime_pay_tool - Tiền làm thêm giờ (Đ107 BLLĐ 2019)",
            "pit_monthly_tool - Thuế TNCN tháng (lũy tiến)",
            "land_registration_fee_tool - Lệ phí trước bạ nhà đất",
            "vehicle_registration_fee_tool - Lệ phí trước bạ xe",
            "court_fee_tool - Án phí dân sự (NQ 326/2016)",
            "admin_fine_lookup_tool - Phạt vi phạm hành chính",
            "child_support_tool - Ước lượng cấp dưỡng con (Đ82 HNGĐ)",
        ],
        "legal_retrieval_tools": [
            "article_lookup_tool - Tra một điều luật trong corpus",
            "precedent_lookup_tool - Tra án lệ theo tình huống",
            "cross_reference_tool - Tìm dẫn chiếu đến một điều",
            "verify_citation_tool - Xác minh trích dẫn (anti-hallucination)",
        ],
        "legal_knowledge_tools": [
            "procedure_wizard_tool - Hướng dẫn thủ tục pháp lý",
            "jurisdiction_resolver_tool - Xác định cơ quan thụ lý",
            "generate_document_template_tool - Sinh văn bản mẫu",
            "law_version_tool - Phiên bản/lịch sử hiệu lực luật",
            "legal_disclaimer_tool - Disclaimer + escalate luật sư",
        ],
        "search_tools": [
            "google_search_tool - Tìm kiếm Google",
            "tavily_search_tool - Tìm kiếm Tavily AI",
            "quick_answer_tool - Trả lời nhanh",
        ],
        "utility_tools": [
            "get_current_time - Thời gian hệ thống",
        ],
    }