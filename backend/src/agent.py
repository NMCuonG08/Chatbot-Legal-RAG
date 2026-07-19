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

# Re-export tracking symbols for back-compat (tests/eval import via `agent.*`).
from agent_tool_tracking import (  # noqa: F401
    agent_tool_calls,
    track_tool_call,
    agent_user_id,
    agent_run_id,
    agent_thread_id,
)
from agent_tool_wrappers import all_tools  # noqa: F401

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


def _get_agent_system_prompt(history_summary: str = "") -> str:
    from datetime import datetime
    now = datetime.now()
    current_year = now.year
    current_date = now.strftime("%d/%m/%Y")
    base = f"""Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam với khả năng sử dụng các công cụ.
Thời gian hệ thống hiện tại: Ngày {current_date} (Năm {current_year}).
LƯU Ý QUAN TRỌNG VỀ THỜI GIAN: Năm hiện tại là {current_year}. Hãy dùng năm {current_year} để tính toán tuổi hoặc kiểm tra thời hiệu khởi kiện!

NHIỆM VỤ:
1. Trả lời câu hỏi pháp luật chính xác dựa trên các công cụ có sẵn
2. Sử dụng công cụ phù hợp cho từng loại câu hỏi
3. Giải thích rõ ràng kết quả từ công cụ cho người dùng

CÁC CÔNG CỤ KHẢ DỤNG:
- Tính toán pháp lý (cơ bản): phạt hợp đồng, thừa kế, tuổi pháp lý
- Tính toán pháp lý (mở rộng): trợ cấp thôi việc, làm thêm giờ, thuế TNCN,
  lệ phí trước bạ nhà đất/xe, án phí, cấp dưỡng con, phạt hành chính
- Tra cứu văn bản luật: tra một điều (article_lookup), tra án lệ (precedent_lookup),
  tìm dẫn chiếu (cross_reference), xác minh trích dẫn (verify_citation)
- Tra cứu bảng: thời hiệu khởi kiện, phiên bản luật (law_version), thủ tục pháp lý
  (procedure_wizard), cơ quan thụ lý (jurisdiction_resolver)
- Tạo văn bản mẫu: đơn khởi kiện/khiếu nại, hợp đồng mua bán, đơn ly hôn
- Tìm kiếm web: Google Search và Tavily AI cho thông tin mới
- Nhớ lại fact cũ: recall_user_memory (năm sinh, giới tính, tình huống đã trao đổi) - chỉ gọi khi câu hỏi tham chiếu ngữ cảnh cũ
- Tuân thủ: legal_disclaimer (kiểm tra escalate + disclaimer), tiện ích thời gian

HƯỚNG DẪN SỬ DỤNG CÔNG CỤ:
- Câu hỏi "Điều X Luật Y nói gì?" → article_lookup; nếu cần chính xác cho user, gọi thêm verify_citation.
- "Trường hợp của tôi bị xử sao / có án lệ nào?" → precedent_lookup.
- "Điều X dẫn chiếu điều nào?" → cross_reference.
- Tính tiền/chi phí → công cụ tính toán phù hợp (thôi việc, OT, TNCN, trước bạ, án phí, cấp dưỡng).
- "Làm thủ tục Z cần gì / nộp ở đâu / mất bao lâu?" → procedure_wizard + jurisdiction_resolver.
- "Sinh đơn/hợp đồng mẫu" → generate_document_template (luôn kèm cảnh báo mẫu tham khảo).
- "Luật này còn hiệu lực không / phiên bản nào?" → law_version.
- Cần thông tin mới/cập nhật → web search (google/tavily).
- Khi cần xác minh thời gian/năm hiện tại → get_current_time.
- Sau khi có kết quả → giải thích bằng tiếng Việt dễ hiểu.

LƯU Ý AN TOÀN PHÁP LÝ (BẮT BUỘC):
- Trước khi trả lời câu hỏi pháp luật, gọi legal_disclaimer để lấy disclaimer gắn cuối câu trả lời.
- Nếu legal_disclaimer trả escalate=True (topic hình sự/bào chữa): KHÔNG tư vấn chi tiết,
  chỉ giải thích khái niệm chung + chuyển user đến luật sư / Trung tâm trợ giúp pháp lý.
- Trước khi khẳng định nội dung một điều luật, gọi verify_citation; nếu verdict != 'consistent',
  không khẳng định - dùng web_search xác minh hoặc nói rõ chưa chắc chắn.
- Luôn trích dẫn căn cứ pháp lý từ kết quả công cụ.
- Nếu công cụ trả về lỗi, giải thích và đề xuất giải pháp khác.
- Trả lời chính xác, chuyên nghiệp, dễ hiểu.

GROUNDING MẠNH (BẮT BUỘC):
- Câu hỏi mới nhất của người dùng (ở dưới, là user_msg) LUÔN là chủ đề chính. Trả lời TRỰC TIẾP nó.
- Mọi thông tin khác (tóm tắt ngữ cảnh dưới đây, kết quả công cụ) chỉ là PHỤ, dùng để bổ sung факт nền.
- KHÔNG để ngữ cảnh cũ đổi chủ đề câu trả lời. Nếu câu hỏi hiện tại hỏi chủ đề mới, trả lời chủ đề mới.
- Chỉ dùng ngữ cảnh cũ khi câu hỏi hiện tại explicitly tham chiếu ("việc đó", "như đã nói", "câu trước").
"""

    if history_summary:
        base += (
            "\nNGỮ CẢNH HỘI THOẠI TRƯỚC (CHỈ THAM KHẢO - KHÔNG PHẢI CÂU HỎI HIỆN TẠI):\n"
            f"{history_summary}\n"
            "\nDùng ngữ cảnh trên chỉ để bổ sung факт nền (năm sinh, giới tính, tình huống). "
            "Câu hỏi chính là câu mới nhất của người dùng - trả lời trực tiếp câu đó."
        )
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
    """Drop in-process memory entries for a user (and optional conversation).

    Clears the rolling-summary cache and the ChatMemoryBuffer for the given
    (user_id, conversation_id) — or ALL conversations for that user when
    ``conversation_id`` is None. Returns the number of cache entries removed.

    Note: only affects the process it runs in. With a separate Celery worker,
    the worker's own copies are NOT cleared from the web process call site;
    call this from the worker too (or restart it) for a full purge.
    """
    removed = 0
    with _memory_lock:
        for store in (_agent_memories, _conv_summaries, _conv_summarized_counts):
            keys_to_drop = [
                (u, c) for (u, c) in list(store.keys())
                if u == user_id and (conversation_id is None or c == conversation_id)
            ]
            for k in keys_to_drop:
                store.pop(k, None)
                removed += 1
    logger.info(
        f"[AGENT] Cleared runtime caches for user={user_id} "
        f"conv={conversation_id} entries_removed={removed}"
    )
    return removed


# Per-conversation rolling summary of chat history. Injected into the agent
# system prompt as grounding context so the agent stays focused on the CURRENT
# question while still knowing prior facts (birth year, gender, situation).
# Keyed by (user_id, conversation_id) with LRU cap, mirroring _agent_memories.
_conv_summaries: "OrderedDict[Tuple[Optional[str], Optional[str]], str]" = OrderedDict()
_conv_summarized_counts: "OrderedDict[Tuple[Optional[str], Optional[str]], int]" = OrderedDict()
_CONV_SUMMARY_CAP = 32

# Guards the three module-level OrderedDicts below: ai_agent_handle runs in a
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
    key = (user_id, conversation_id)
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

    with _memory_lock:
        already = _conv_summarized_counts.get(key, 0)
        prev_summary = _conv_summaries.get(key, "")
    # Periodic full re-summary from raw to correct incremental drift.
    full_resummary = len(prior_msgs) % _RESUMMARY_INTERVAL == 0
    needs_update = full_resummary or len(old_turns) > already

    if not needs_update:
        # LRU touch on read so the active conversation is not evicted while
        # cold ones occupy the tail; keeps the two stores' ordering in sync.
        with _memory_lock:
            _conv_summaries.move_to_end(key)
            _conv_summarized_counts.move_to_end(key)
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
            with _memory_lock:
                _conv_summaries[key] = summary
                _conv_summaries.move_to_end(key)
                _conv_summarized_counts[key] = len(old_turns)
                _conv_summarized_counts.move_to_end(key)
                if len(_conv_summaries) > _CONV_SUMMARY_CAP:
                    ev_k, _ = _conv_summaries.popitem(last=False)
                    _conv_summarized_counts.pop(ev_k, None)
                # Cap the counts store symmetrically so it cannot drift past
                # _CONV_SUMMARY_CAP (it has no independent cap otherwise).
                if len(_conv_summarized_counts) > _CONV_SUMMARY_CAP:
                    _conv_summarized_counts.popitem(last=False)
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
    """
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
    if any(kw in scan_text for kw in ["phạt", "chậm trễ", "vi phạm hợp đồng"]):
        selected.append("contract_penalty_tool")
    if any(kw in scan_text for kw in ["tuổi", "sinh năm", "năm sinh", "kết hôn", "lao động", "hình sự"]):
        selected.append("legal_age_tool")
    if any(kw in scan_text for kw in ["thừa kế", "di sản", "chia"]):
        selected.append("inheritance_tool")
    if any(kw in scan_text for kw in ["doanh nghiệp", "tên", "đặt tên"]):
        selected.append("business_name_tool")
    if any(kw in scan_text for kw in ["thời hiệu", "khởi kiện", "khiếu nại"]):
        selected.append("statute_tool")
        
    # 2. Knowledge / data-driven legal tools
    if any(kw in scan_text for kw in ["thôi việc", "sa thải", "trợ cấp"]):
        selected.append("severance_pay_func_tool")
    if any(kw in scan_text for kw in ["làm thêm", "tăng ca", "ot", "giờ"]):
        selected.append("overtime_pay_func_tool")
    if any(kw in scan_text for kw in ["thuế", "tncn", "thu nhập"]):
        selected.append("pit_monthly_func_tool")
    if any(kw in scan_text for kw in ["trước bạ", "đất đai", "nhà đất", "lệ phí"]):
        selected.append("land_registration_fee_func_tool")
        selected.append("vehicle_registration_fee_func_tool")
    if any(kw in scan_text for kw in ["án phí", "tòa án", "lệ phí tòa"]):
        selected.append("court_fee_func_tool")
    if any(kw in scan_text for kw in ["phạt hành chính", "vi phạm", "giao thông"]):
        selected.append("admin_fine_lookup_func_tool")
    if any(kw in scan_text for kw in ["cấp dưỡng", "nuôi con", "ly hôn"]):
        selected.append("child_support_func_tool")
    if any(kw in scan_text for kw in ["thủ tục", "quy trình", "bước"]):
        selected.append("procedure_wizard_func_tool")
    if any(kw in scan_text for kw in ["thẩm quyền", "tòa án nào", "thụ lý"]):
        selected.append("jurisdiction_resolver_func_tool")
    if any(kw in scan_text for kw in ["mẫu đơn", "văn bản mẫu", "hợp đồng mẫu"]):
        selected.append("generate_document_template_func_tool")
    if any(kw in scan_text for kw in ["hiệu lực", "phiên bản", "sửa đổi", "bổ sung"]):
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

    # Fallback: if no specific tools matched, return a sensible default set
    if not tools_list or len(tools_list) <= 3:
        fallback = [
            getattr(agent_tool_wrappers, "contract_penalty_tool"),
            getattr(agent_tool_wrappers, "legal_age_tool"),
            getattr(agent_tool_wrappers, "inheritance_tool"),
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
) -> ReActAgent | None:
    """Build ReAct agent with compatibility across llama-index versions."""
    if llm is None:
        return None

    prompt = _get_agent_system_prompt(history_summary=history_summary)

    # Older versions expose from_tools, newer workflow-based versions use constructor.
    if hasattr(ReActAgent, "from_tools"):
        return ReActAgent.from_tools(
            tools,
            llm=llm,
            memory=memory,
            verbose=True,
            max_iterations=10,
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
) -> Tuple[str, List[Dict]]:
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
        ``(response_text, tool_calls)`` where tool_calls is a list of
        ``{tool_name, args, status, error, result}`` dicts captured during this
        run via the ``agent_tool_calls`` contextvar.
    """
    # Reset the tool-call accumulator for this run so @track_tool_call records
    # only this turn's calls. contextvars propagate into asyncio.run coroutines.
    token = agent_tool_calls.set([])
    user_token = agent_user_id.set(user_id)
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

        ai_agent = _build_react_agent(_llm, memory, tools, history_summary=history_summary)
        if ai_agent is None:
            return (
                "Xin lỗi, lỗi khi khởi tạo hệ thống AI Agent.",
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
                "max_iterations": 10,
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
            # Use asyncio.run for clean loop management in each Celery task
            logger.info(f"[AGENT] Running agent (Type: {type(ai_agent)})")
            result = asyncio.run(_safe_execute_agent())
        except Exception as e:
            if "already running" in str(e).lower():
                logger.info("[AGENT] Loop already running, applying nest_asyncio")
                import nest_asyncio
                nest_asyncio.apply()
                # Use current loop
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(_safe_execute_agent())
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

        # Clean up common prefixes
        if res_text.startswith("assistant: "):
            res_text = res_text[len("assistant: "):]

        logger.info(f"[AGENT] Clean response extracted (length: {len(res_text)})")
        tool_calls = list(agent_tool_calls.get() or [])
        logger.info(f"[AGENT] Captured {len(tool_calls)} tool calls")
        return res_text, tool_calls
    except Exception as e:
        logger.error(
            f"[AGENT] Error: type={type(e).__name__} msg={e!r} "
            f"traceback={traceback.format_exc()}"
        )
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi ({type(e).__name__}).", []
    finally:
        agent_tool_calls.reset(token)
        agent_user_id.reset(user_token)
        agent_run_id.reset(rid_token)
        agent_thread_id.reset(tid_token)


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