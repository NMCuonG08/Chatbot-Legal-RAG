"""ReAct agent runtime: LLM, system prompt, per-conversation memory/cache, and
the ``ai_agent_handle`` Celery task.

Tool wrappers + FunctionTool instances live in ``agent_tool_wrappers.py``; the
per-run tool-call tracker (``agent_tool_calls``, ``track_tool_call``) lives in
``agent_tool_tracking.py``. Both are re-exported here so legacy callers that do
``from agent import ...`` / ``agent.track_tool_call`` keep working.
"""

import asyncio
import inspect
import logging
import os

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
from llama_index.core.llms import ChatMessage

# Re-export tracking symbols for back-compat (tests/eval import via `agent.*`).
from agent_tool_tracking import agent_tool_calls, track_tool_call  # noqa: F401
from agent_tool_wrappers import all_tools  # noqa: F401

logger = logging.getLogger(__name__)

# Initialize the LLM LAZILY based on environment variables. Building at import
# time required env vars / network at import, which broke test collection and
# any importer that does not need the agent. We now build on first use via
# _get_ai_agent(user_id, conversation_id), which caches a per-conversation agent.
_llm = None

logger.info("LlamaIndex Agent LLM will be initialized lazily on first use")


def _build_llm():
    """Build the LlamaIndex LLM from env vars. Returns None if config missing."""
    llm_provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    llm_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
    logger.info(f"Initializing LlamaIndex Agent LLM - Provider: {llm_provider}, Model: {llm_model}")

    if llm_provider == "groq":
        from llama_index.llms.groq import Groq
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            return Groq(model=llm_model, api_key=groq_api_key, temperature=0.1)
        logger.warning("GROQ_API_KEY not set, agent initialization may fail")
        return None
    elif llm_provider == "ollama":
        from llama_index.llms.ollama import Ollama
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        api_key = os.environ.get("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        return Ollama(
            model=llm_model,
            base_url=ollama_url,
            temperature=0.1,
            request_timeout=60.0,
            headers=headers
        )
    elif llm_provider == "openai":
        from llama_index.llms.openai import OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        openai_base = os.environ.get("OPENAI_API_BASE")
        return OpenAI(model=llm_model, api_key=openai_key, api_base=openai_base, temperature=0.1)
    else:
        logger.warning(f"Unsupported LLM provider: {llm_provider}. Falling back to Groq.")
        from llama_index.llms.groq import Groq
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            return Groq(model=llm_model, api_key=groq_api_key, temperature=0.1)
        return None


def _get_agent_system_prompt() -> str:
    from datetime import datetime
    now = datetime.now()
    current_year = now.year
    current_date = now.strftime("%d/%m/%Y")
    return f"""Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam với khả năng sử dụng các công cụ.
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
  không khẳng định — dùng web_search xác minh hoặc nói rõ chưa chắc chắn.
- Luôn trích dẫn căn cứ pháp lý từ kết quả công cụ.
- Nếu công cụ trả về lỗi, giải thích và đề xuất giải pháp khác.
- Trả lời chính xác, chuyên nghiệp, dễ hiểu."""


# Per-conversation agent memory (Phase E). Previously a single global
# ChatMemoryBuffer was shared across ALL users/conversations — a cross-user
# data leak. Now keyed by (user_id, conversation_id) with an LRU cap.
_agent_memories: "OrderedDict[Tuple[Optional[str], Optional[str]], ChatMemoryBuffer]" = OrderedDict()
_AGENT_MEMORY_CAP = 32


def get_agent_memory(user_id: Optional[str], conversation_id: Optional[str]) -> ChatMemoryBuffer:
    """Return (creating if needed) the ChatMemoryBuffer for a (user, conversation).

    LRU-evicts the oldest entry once the cap is exceeded.
    """
    key = (user_id, conversation_id)
    mem = _agent_memories.get(key)
    if mem is None:
        mem = ChatMemoryBuffer.from_defaults(token_limit=8192)
        _agent_memories[key] = mem
        _agent_memories.move_to_end(key)
        if len(_agent_memories) > _AGENT_MEMORY_CAP:
            evicted_key, _ = _agent_memories.popitem(last=False)
            logger.info(f"[AGENT] Evicted oldest agent memory for key={evicted_key}")
    return mem


def _build_react_agent(llm, memory: ChatMemoryBuffer) -> ReActAgent | None:
    """Build ReAct agent with compatibility across llama-index versions."""
    if llm is None:
        return None

    prompt = _get_agent_system_prompt()

    # Older versions expose from_tools, newer workflow-based versions use constructor.
    if hasattr(ReActAgent, "from_tools"):
        return ReActAgent.from_tools(
            all_tools,
            llm=llm,
            memory=memory,
            verbose=True,
            max_iterations=10,
            context=prompt,
        )

    return ReActAgent(
        tools=all_tools,
        llm=llm,
        memory=memory,
        verbose=True,
        max_iterations=10,
        system_prompt=prompt,
    )


# Per-(user, conversation) agent cache so each conversation gets its own memory.
# A single global agent would force a single shared memory again.
_ai_agent_cache: "OrderedDict[Tuple[Optional[str], Optional[str]], ReActAgent]" = OrderedDict()
_AI_AGENT_CACHE_CAP = 32


def _get_ai_agent(user_id: Optional[str] = None, conversation_id: Optional[str] = None):
    """Lazily build and cache a per-(user, conversation) ReAct agent.

    Building is deferred until the first ``ai_agent_handle`` call so that
    importing this module (e.g. for ``agent_tool_calls`` in eval) does not
    require LLM env vars or network access. Defaults ``(None, None)`` keep
    legacy callers working (a shared default-memory agent).
    """
    global _llm
    key = (user_id, conversation_id)
    cached = _ai_agent_cache.get(key)
    if cached is not None:
        _ai_agent_cache.move_to_end(key)
        return cached

    if _llm is None:
        _llm = _build_llm()
    memory = get_agent_memory(user_id, conversation_id)
    agent = _build_react_agent(_llm, memory)
    if agent is None:
        return None
    _ai_agent_cache[key] = agent
    _ai_agent_cache.move_to_end(key)
    if len(_ai_agent_cache) > _AI_AGENT_CACHE_CAP:
        evicted_key, _ = _ai_agent_cache.popitem(last=False)
        logger.info(f"[AGENT] Evicted oldest agent for key={evicted_key}")
    logger.info(f"Agent initialized with {len(all_tools)} tools for key={key}")
    return agent


@shared_task()
def ai_agent_handle(
    question: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Tuple[str, List[Dict]]:
    """
    Handle user question using ReAct agent with tools.

    Args:
        question: User's question
        user_id: Optional user id for per-conversation memory isolation.
        conversation_id: Optional conversation id for per-conversation memory.

    Returns:
        ``(response_text, tool_calls)`` where tool_calls is a list of
        ``{tool_name, args, status, error, result}`` dicts captured during this
        run via the ``agent_tool_calls`` contextvar.
    """
    # Reset the tool-call accumulator for this run so @track_tool_call records
    # only this turn's calls. contextvars propagate into asyncio.run coroutines.
    token = agent_tool_calls.set([])
    try:
        ai_agent = _get_ai_agent(user_id, conversation_id)
        if ai_agent is None:
            return (
                "Xin lỗi, hệ thống AI agent chưa sẵn sàng do thiếu cấu hình GROQ_API_KEY. "
                "Vui lòng kiểm tra biến môi trường và thử lại.",
                [],
            )

        logger.info(f"[AGENT] Processing question: {question}")

        # Robust execution wrapper to handle different LlamaIndex versions and event loops
        async def _safe_execute_agent():
            # Try async methods first (modern llama-index)
            # Newer ReActAgent (Workflow) uses run/arun which often returns a WorkflowHandler
            if hasattr(ai_agent, "arun"):
                # Try with 'input' keyword (modern Workflow) or positional
                try:
                    result = await ai_agent.arun(input=question)
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
        logger.error(f"[AGENT] Error: {e}")
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}", []
    finally:
        agent_tool_calls.reset(token)


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