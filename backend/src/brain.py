import contextvars
import json
import logging
import os
import re
import requests

from groq import Groq
from redis import InvalidResponse

from custom_embedding import get_custom_embedding
from retry_utils import with_retry

logger = logging.getLogger(__name__)

from langsmith import traceable


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", default=None)
# Vietnamese LLM endpoint must be configured explicitly. No hardcoded public IP
# default — sending chat traffic to an unknown third-party box is a data-leak risk.
VIETNAMESE_LLM_API_URL = os.environ.get("VIETNAMESE_LLM_API_URL", default=None)

# Stores list of dicts: {"provider": str, "model": str, "prompt_tokens": int, "completion_tokens": int}
usage_accumulator = contextvars.ContextVar("usage_accumulator", default=None)

# Per-call provider/model override (P2: judge hardening + pairwise A/B + cost
# routing). When set, get_main_provider() / build_groq_provider() honor these
# over the env defaults, so a judge call or shadow-variant run can pin a
# different model without mutating global env. None = use env default.
LLM_PROVIDER_CONTEXTVAR = contextvars.ContextVar("llm_provider_override", default=None)
LLM_MODEL_CONTEXTVAR = contextvars.ContextVar("llm_model_override", default=None)


def record_usage(provider: str, model: str, usage) -> None:
    """Append a token-usage record to the active accumulator (if any).

    Accepts BOTH attribute-style usage (Groq SDK: ``usage.prompt_tokens``) and
    dict-style usage (OpenAI/Ollama JSON: ``usage['prompt_tokens']``), collapsing
    the ~6 copy-pasted accumulator blocks that previously lived inline in each
    provider branch.
    """
    if usage is None:
        return
    acc = usage_accumulator.get()
    if acc is None:
        return
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
    else:
        prompt = getattr(usage, "prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", 0)
    acc.append({
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    })


# Generic error message returned to end users on internal failures. Exception
# details are logged server-side only (never leaked to the chat response).
_USER_FACING_ERROR = "Xin lỗi, đã xảy ra lỗi nội bộ khi xử lý yêu cầu. Vui lòng thử lại sau."


def get_groq_client():
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not configured in the environment.")
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.warning(f"Could not initialize Groq client: {e}")
        return None


client = get_groq_client()

# LangSmith auto-tracing of the raw Groq client (senior). The Groq SDK is
# OpenAI-shaped (client.chat.completions.create), so wrap_openai patches that
# method to emit a full LLM run on every call — capturing model, messages,
# token usage, and latency that the @traceable wrapper on chat() cannot infer
# from the returned content string. Without this, LangSmith shows the LLM
# span but with no token/usage/latency data, which reads as "incomplete tracing".
# Defensive: if the Groq SDK shape ever diverges from openai, skip silently
# rather than break chat.
if client is not None:
    try:
        from langsmith.wrappers import wrap_openai
        client = wrap_openai(client)
        logger.debug("LangSmith wrap_openai applied to Groq client")
    except Exception as _wrap_err:
        logger.debug(f"LangSmith wrap_openai skipped for Groq: {_wrap_err}")


# Thin retry-wrapped binding around the Groq chat completions create call.
# Retries only on transient errors (429 rate limits, 5xx, connection/timeout);
# non-retryable errors propagate to the caller's try/except fallback unchanged.
@with_retry(max_attempts=3, base_delay=1.0, max_delay=8.0)
def _groq_chat_create(model_name: str, messages, temperature: float = 0.7, max_tokens: int = 2048):
    if client is None:
        raise RuntimeError("Groq client is not initialized")
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# --------------------------------------------------------------------------- #
# LLM provider classes
# --------------------------------------------------------------------------- #
# Each provider encapsulates ONE backend + its own model knob, so the model in
# use is always unambiguous. The previous loose-function design read the model
# from different env vars in different branches — that is how a weak model
# silently took over generation (OLLAMA_MODEL vs OLLAMA_LLM_MODEL vs LLM_MODEL).
# A small router (get_main_provider) wires the classes together for dispatch.

class LLMProvider:
    """Base class for chat-completion providers."""
    name = "base"

    def __init__(self, model: str):
        self.model = model

    def is_available(self) -> bool:
        return True

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 2048, raw: bool = False):
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq via the official SDK. Model from LLM_MODEL (default weak only if unset)."""
    name = "groq"

    def is_available(self) -> bool:
        return client is not None

    @traceable(run_type="llm")
    def chat(self, messages, temperature=0.7, max_tokens=2048, raw=False):
        if not self.is_available():
            logger.warning("GROQ_API_KEY not set, cannot use Groq provider")
            return _USER_FACING_ERROR
        try:
            response = _groq_chat_create(self.model, messages, temperature, max_tokens)
            usage = getattr(response, "usage", None)
            output = response.choices[0].message
            
            if raw:
                record_usage(self.name, self.model, usage)
                return response.choices[0].message
            logger.info(f"Groq chat complete output: {output.content[:100]}")
            record_usage(self.name, self.model, usage)
            return output.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return _USER_FACING_ERROR


class OllamaProvider(LLMProvider):
    """Ollama via its OpenAI-compatible /v1 endpoint. Model from OLLAMA_LLM_MODEL
    (OLLAMA_MODEL legacy fallback) — never from LLM_MODEL, so the weak groq
    default can never leak into local generation."""
    name = "ollama"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 30):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @traceable(run_type="llm")
    def chat(self, messages, temperature=0.7, max_tokens=2048, raw=False):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        endpoint = f"{self.base_url}/v1/chat/completions"
        logger.info(
            f"Ollama request -> endpoint={endpoint} model={self.model} "
            f"api_key={'set' if self.api_key else 'MISSING'} messages={len(messages)}"
        )
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.error(
                    f"Ollama API error: {response.status_code} - {response.text} "
                    f"(model={self.model})"
                )
                return _USER_FACING_ERROR
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"Ollama chat complete output: {content[:100]}")
            
            record_usage(self.name, self.model, result.get("usage"))
            if raw:
                from types import SimpleNamespace
                return SimpleNamespace(content=content)
            return content
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return _USER_FACING_ERROR


# --------------------------------------------------------------------------- #
# Provider factory + router
# --------------------------------------------------------------------------- #

def build_groq_provider(model: str | None = None) -> GroqProvider:
    m = model or LLM_MODEL_CONTEXTVAR.get() or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    if m and "cloud" in m.lower():
        m = "llama-3.3-70b-versatile"
    return GroqProvider(m)


def build_ollama_provider(model: str | None = None, fast: bool = False) -> OllamaProvider:
    base = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")
    # Cloud Ollama (e.g. http://ollama.com) requires https. An http URL triggers
    # a 301 redirect that `requests` follows by converting POST->GET, which then
    # 405s on /v1/chat/completions. Upgrade non-localhost http to https; keep
    # localhost as http (local Ollama has no TLS).
    if base.startswith("http://") and "localhost" not in base and "127.0.0.1" not in base:
        base = "https://" + base[len("http://"):]
    if model:
        m = model
    elif fast:
        m = os.environ.get("OLLAMA_FAST_LLM_MODEL") or "qwen2.5:7b"
    else:
        m = LLM_MODEL_CONTEXTVAR.get() or os.environ.get("OLLAMA_LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or "gemma4:31b"
    return OllamaProvider(base, m, os.environ.get("OLLAMA_API_KEY"))


def get_main_provider() -> LLMProvider:
    """Router: pick the main LLM provider from LLM_PROVIDER (groq|ollama),
    honoring a per-call contextvar override first (P2). Default groq. Unknown
    values fall back to groq with a warning."""
    provider = (LLM_PROVIDER_CONTEXTVAR.get() or os.environ.get("LLM_PROVIDER", "groq")).lower()
    if provider == "ollama":
        return build_ollama_provider()
    if provider == "groq":
        return build_groq_provider()
    logger.warning(f"Unknown LLM_PROVIDER '{provider}', falling back to groq")
    return build_groq_provider()


def build_judge_fn(provider: str = "groq", model: str | None = None,
                   temperature: float = 0.0):
    """Build a judge_fn(messages) -> str closure pinned to a provider+model.

    Groq + Ollama only (no OpenAI/Anthropic key in this project). The closure
    sets the provider/model contextvars for its call so usage_accumulator
    records the right model, then restores them. Returns a callable matching
    the ``judge_fn`` contract used by metrics_generation / verify_answer.
    """
    prov = (provider or "groq").lower()
    model = model or os.environ.get("JUDGE_MODEL", "llama-3.1-8b-instant")

    def _judge(messages, *, _prov=prov, _model=model, _temp=temperature):
        pv_tok = LLM_PROVIDER_CONTEXTVAR.set(_prov)
        mv_tok = LLM_MODEL_CONTEXTVAR.set(_model)
        try:
            if _prov == "ollama":
                p = build_ollama_provider(_model)
            else:
                p = build_groq_provider(_model)
            return p.chat(messages, temperature=_temp)
        finally:
            LLM_PROVIDER_CONTEXTVAR.reset(pv_tok)
            LLM_MODEL_CONTEXTVAR.reset(mv_tok)

    _judge.provider = prov  # type: ignore[attr-defined]
    _judge.model = model  # type: ignore[attr-defined]
    return _judge


def vietnamese_llm_chat_complete(messages=(), temperature=0.7, max_tokens=2048):
    """Main user-facing answer path.

    Routing:
    - USE_OLLAMA_AS_MAIN=true -> Ollama directly (strong local model), with
      fallback to the routed provider on failure.
    - else -> custom VIETNAMESE_LLM_API_URL endpoint, fallback to routed provider.

    Model selection is centralized in the provider classes (no env var is read
    inline here), so the weak-model-by-accident class of bug cannot recur.
    """
    logger.info("Vietnamese LLM chat complete for {}".format(str(messages)[:300]))

    if os.environ.get("USE_OLLAMA_AS_MAIN", "false").lower() == "true":
        logger.info("USE_OLLAMA_AS_MAIN is True. Routing main generation to Ollama.")
        ollama = build_ollama_provider()
        content = ollama.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if content != _USER_FACING_ERROR:
            return content
        logger.error("Ollama failed for main generation; falling back to routed provider.")
        return get_main_provider().chat(messages, temperature=temperature, max_tokens=max_tokens)

    if not VIETNAMESE_LLM_API_URL:
        logger.warning("VIETNAMESE_LLM_API_URL chưa cấu hình. Fallback sang provider chính.")
        return get_main_provider().chat(messages, temperature=temperature, max_tokens=max_tokens)

    try:
        payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        response = requests.post(
            VIETNAMESE_LLM_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,  # 5 phút timeout
        )
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info("Vietnamese LLM response: {}".format(content[:200] + "..."))
            record_usage("vietnamese_llm", "vietnamese-legal-llm", result.get("usage"))
            return content
        logger.error(f"Vietnamese LLM API error: {response.status_code} - {response.text}")
        logger.info("Falling back to routed provider")
        return get_main_provider().chat(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logger.error(f"Error calling Vietnamese LLM: {e}")
        logger.info("Falling back to routed provider due to error")
        return get_main_provider().chat(messages, temperature=temperature, max_tokens=max_tokens)


def groq_chat_complete(messages=(), model=None, raw=False, temperature=0.7):
    """Backward-compatible chat completion.

    Routes through the centralized provider router (groq|ollama) so model
    selection is no longer scattered across branches. Honors an explicit
    ``model`` override by building a one-off provider for that call only.
    Kept as the public entrypoint because tasks.py / query_rewriter / eval
    call it (and the ``openai_chat_complete`` alias) by name.
    """
    base = get_main_provider()
    if model:
        provider = build_ollama_provider(model) if isinstance(base, OllamaProvider) else build_groq_provider(model)
    else:
        provider = base
    logger.info(f"Chat complete using provider: {provider.name}, model: {provider.model}, temp: {temperature}")
    return provider.chat(messages, raw=raw, temperature=temperature)


# Alias kept for backward compatibility (tasks.py / query_rewriter import this).
openai_chat_complete = groq_chat_complete


def get_embedding(text, model=None):
    """
    Get embedding using custom Vietnamese legal model
    Note: model parameter is kept for backward compatibility but not used
    """
    text = text.replace("\n", " ")
    logger.info(f"� Using custom embedding model for text: {text[:100]}...")
    return get_custom_embedding(text)


def gen_doc_prompt(docs):
    """Format retrieved docs into a numbered context block for the LLM.

    Each doc is labelled ``[Tài liệu n]`` (1-based, in input order) with its
    law_name / article_number / source metadata surfaced. The matching
    ``CITATION_RULE`` (citations.py) tells the LLM to echo ``[n]`` inline at
    each claim, so the frontend can bind each claim to its source popup.

    The 1-based index here MUST match ``citations.normalize_sources`` (both
    iterate the same list in the same order) — that is the contract that lets
    the frontend resolve an LLM-emitted ``[n]`` to the right source dict.

    Args:
        docs: list of retrieved chunk dicts. Required keys: ``content``.
            Optional: ``question``, ``law_name``, ``article_number``, ``source``.

    Returns:
        str: ``"Tài liệu tham khảo:\\n[Tài liệu 1]\\n..."``.
    """
    doc_prompt = ""
    for idx, doc in enumerate(docs, start=1):
        law_name = (doc.get("law_name") or "").strip()
        article_number = doc.get("article_number")
        clause_number = doc.get("clause_number")
        point_letter = doc.get("point_letter")
        document_type = doc.get("document_type") or ""
        effectivity = doc.get("effectivity_status") or ""
        source = (doc.get("source") or "").strip() or "—"
        meta_parts = []
        if law_name:
            meta_parts.append(f"Luật: {law_name}")
        if article_number is not None:
            meta_parts.append(f"Điều: {article_number}")
        if clause_number is not None:
            meta_parts.append(f"Khoản: {clause_number}")
        if point_letter:
            meta_parts.append(f"Điểm: {point_letter}")
        if document_type:
            meta_parts.append(f"Loại VB: {document_type}")
        if effectivity:
            meta_parts.append(f"Hiệu lực: {effectivity}")
        meta_parts.append(f"Nguồn: {source}")
        meta = " | ".join(meta_parts)
        question = (doc.get("question") or "").strip()
        content = (doc.get("content") or "").strip()
        doc_prompt += f"[Tài liệu {idx}]\n{meta}\n"
        if question:
            doc_prompt += f"Câu hỏi: {question}\n"
        doc_prompt += f"Nội dung: {content}\n\n"

    return "Danh sách tài liệu được cung cấp:\n{}".format(doc_prompt)


def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text


def detect_user_intent(history, message):
    """
    Detect user intent and rephrase follow-up questions to standalone questions.
    Improved for Vietnamese legal context with better prompt engineering.
    """
    # Convert history to list messages
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")

    # Check if this is likely a follow-up question
    follow_up_indicators = [
        "đó",
        "này",
        "kia",
        "thế",
        "vậy",
        "nữa",
        "còn",
        "như vậy",
        "như thế",
    ]
    is_follow_up = any(
        indicator in message.lower() for indicator in follow_up_indicators
    )

    # Skip rewrite when there is no history to draw from, OR when the history
    # is short and this is not a follow-up. Rewrite only when there is history
    # AND (history is long OR the message looks like a follow-up).
    # Parens explicit to avoid operator-precedence ambiguity.
    if not history or (len(history) <= 1 and not is_follow_up):
        logger.info("No context needed, returning original query")
        return message

    # Update documents to prompt with better Vietnamese legal context
    user_prompt = f"""Bạn là trợ lý AI chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là viết lại câu hỏi tiếp theo thành một câu hỏi độc lập, rõ ràng và đầy đủ ngữ cảnh.

Lịch sử hội thoại:
{history_messages}

Câu hỏi hiện tại: {message}

Hướng dẫn:
1. Viết lại câu hỏi sao cho có thể hiểu được mà KHÔNG cần đọc lịch sử hội thoại
2. Thay thế các đại từ (nó, đó, này, kia, thế, vậy) bằng danh từ hoặc cụm từ cụ thể từ ngữ cảnh
3. Bổ sung thông tin cần thiết từ lịch sử để câu hỏi trở nên đầy đủ
4. Giữ nguyên ý định hỏi về pháp luật của người dùng
5. Sử dụng thuật ngữ pháp lý chính xác và phù hợp với ngữ cảnh Việt Nam
6. CHỈ trả về câu hỏi đã viết lại, KHÔNG giải thích thêm

Ví dụ:
Lịch sử: "User: Thủ tục ly hôn như thế nào?\nAssistant: Thủ tục ly hôn theo quy định..."
Câu hỏi: "Còn chi phí thì sao?"
Kết quả: "Chi phí thủ tục ly hôn theo pháp luật Việt Nam là bao nhiêu?"

Câu hỏi đã viết lại:"""

    openai_messages = [
        {
            "role": "system",
            "content": "Bạn là chuyên gia tư vấn pháp luật Việt Nam, giỏi phân tích và làm rõ câu hỏi pháp lý.",
        },
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"Rephrase input messages: {openai_messages}")

    try:
        rephrased = groq_chat_complete(openai_messages)
        logger.info(f"Rephrased question: {rephrased}")
        return rephrased.strip()
    except Exception as e:
        logger.error(f"Error rephrasing question: {e}")
        return message


# Unambiguous tool-specific signals that map 1:1 to an agent_tools function.
# Used ONLY to force route TOWARD agent_tools (safe direction — agent_tools
# can hand off to RAG/web). Never used to force any other route. Keep this list
# narrow: only phrases that have no plausible legal_rag interpretation.
_AGENT_TOOLS_OVERRIDE_SIGNALS = [
    # Fee / money calculation tools
    "lệ phí trước bạ", "án phí", "trợ cấp thôi việc", "cấp dưỡng",
    "làm thêm giờ", "làm thêm", "tăng ca", "thuế tncn", "thuế thu nhập cá nhân",
    # Document generation
    "sinh đơn", "mẫu đơn", "hợp đồng mẫu", "đơn khởi kiện", "đơn khiếu nại",
    # Law version / validity check
    "còn hiệu lực không", "phiên bản", "hiện còn hiệu lực",
    # Jurisdiction
    "tòa cấp nào thụ lý", "tòa nào thụ lý", "thẩm quyền",
    # Age
    "sinh năm", "năm sinh", "mấy tuổi", "bao nhiêu tuổi",
    # Admin fine calc (specific phrasing, not "phạt" alone which is ambiguous)
    "phạt bao nhiêu", "bị phạt bao nhiêu",
]


def _detect_agent_tools_override(message: str) -> bool:
    """True if the message contains an unambiguous agent_tools signal.

    Substring match on normalized lowercase text. Intentionally conservative —
    only matches phrases tied to a specific tool with no legal_rag interpretation.
    """
    msg = message.lower()
    return any(sig in msg for sig in _AGENT_TOOLS_OVERRIDE_SIGNALS)


# Pure citation-lookup pattern: "Điều N <statute-type> ..." with NO calc/tool verb.
# Forces route TOWARD legal_rag (deterministic, no-LLM retrieve_node path) so a
# plain "Điều 418 Bộ luật Dân sự nói gì?" NEVER enters the ReAct agent loop —
# killing the 75s lũy kế at the source. Checked AFTER _detect_agent_tools_override
# so "Điều X luật Y còn hiệu lực không" (validity check) still goes agent_tools.
# Narrow: only statute type + article number; free-text questions don't match.
_LEGAL_RAG_OVERRIDE_RE = re.compile(
    r"điều\s+\d{1,4}\s+"
    r"(?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|pháp\s+lệnh|nghị\s+quyết)",
    re.IGNORECASE,
)


def _detect_legal_rag_override(message: str) -> bool:
    """True if the message is a pure article-citation lookup (no calc verb).

    Forces TOWARD legal_rag — the deterministic retrieve_node path (hybrid search
    + graph, no ReAct LLM loop). Safe direction: legal_rag can hand off to
    web_search/agent via the existing supervisor if retrieval comes up empty, so
    a misrouted question is still answered. The reverse (LLM dropping a pure
    lookup into agent_tools) costs a 60-90s ReAct thrash for nothing.
    """
    return bool(_LEGAL_RAG_OVERRIDE_RE.search(message or ""))


# Route distribution counters (observability). Reset on process restart.
# Exposed via the app /stats endpoint so ops can spot route skew (e.g. all
# traffic falling into one route) or override firing too often.
_route_stats = {"legal_rag": 0, "agent_tools": 0, "web_search": 0, "general_chat": 0}


def get_route_stats() -> dict:
    """Snapshot of route distribution (for the /stats endpoint)."""
    return dict(_route_stats)


def _record_route(route: str) -> None:
    if route in _route_stats:
        _route_stats[route] += 1


@traceable(name="detect_route", run_type="chain")
def detect_route(history, message):
    """
    Detect the appropriate tool/route for handling the user's query.
    Enhanced for Vietnamese legal chatbot with 4 routing options including agent tools.

    Routes:
    - legal_rag: Questions about Vietnamese laws, regulations (uses RAG system with vector search)
    - agent_tools: Questions requiring calculation, validation, or complex reasoning (uses ReAct agent)
    - web_search: Current events, recent legal changes requiring internet search
    - general_chat: Greetings, small talk, off-topic conversations
    """
    logger.info(f"Detect route on history messages: {history}")

    # Hybrid safety-net: the LLM router is non-deterministic (observed 94% one
    # run, 69% the next at temp 0.7) and tends to drop clear calculation/tool
    # questions into legal_rag. These signals map 1:1 to specific agent_tools
    # functions and are unambiguous, so we force agent_tools deterministically.
    # This is SAFE: it only ever forces TOWARD agent_tools, which can hand off
    # to RAG/web_search if a tool isn't actually needed — so a legal_rag-style
    # question misrouted here still gets answered. The reverse (LLM dropping a
    # calc question into legal_rag) loses the tool entirely. Asymmetric risk →
    # override toward agent_tools for these.
    override = _detect_agent_tools_override(message)
    if override:
        logger.info(f"Route override -> agent_tools (deterministic tool signal) for: '{message[:60]}'")
        _record_route("agent_tools")
        return "agent_tools"

    # Deterministic legal_rag override for pure citation lookups ("Điều N <luật>")
    # — skips the LLM router (non-deterministic) + the ReAct agent loop entirely,
    # routing straight to the no-LLM retrieve_node. Kills the 75s lũy kế at the
    # source for the most common legal question shape. Checked AFTER agent_tools
    # override so validity/calc queries still route to the agent.
    if _detect_legal_rag_override(message):
        logger.info(f"Route override -> legal_rag (deterministic citation lookup) for: '{message[:60]}'")
        _record_route("legal_rag")
        return "legal_rag"

    # Format history for better context
    history_text = ""
    if history and len(history) > 1:
        for msg in history[-4:]:  # Last 2 exchanges
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"Người dùng: {content}\n"
            elif role == "assistant":
                history_text += f"Trợ lý: {content}\n"

    # Prompt-engineered router: semantic decision tree + chain-of-thought +
    # structured output. No keyword hacks — LLM understands intent, scales
    # with new tools without maintaining parallel keyword dicts.
    import re
    from datetime import datetime
    current_year = datetime.now().year

    user_prompt = f"""Bạn là bộ định tuyến (router) cho chatbot TƯ VẤN PHÁP LUẬT VIỆT NAM.
Phân tích câu hỏi + ngữ cảnh hội thoại rồi xếp vào MỘT trong 4 luồng.

NĂM HIỆN TẠI: {current_year}. Dùng mốc này khi câu hỏi phụ thuộc thời gian
(tuổi, thời hiệu, hiệu lực luật, lương vùng, văn bản mới ban hành).

Lịch sử hội thoại:
{history_text}

Câu hỏi cần định tuyến:
{message}

CÂY QUYẾT ĐỊNH (theo thứ tự, dừng ở nhánh match đầu tiên):

1. agent_tools — CÂU HỎI CẦN TÍNH TOÁN / KIỂM TRA / SINH VĂN BẢN / TRA CỨU CỤ THỂ:
   - Tính số tiền/tuổi/thời hạn: "tôi sinh năm 2004 năm nay bao nhiêu tuổi",
     "tính phạt hợp đồng 100 triệu chậm 30 ngày", "tính trợ cấp thôi việc sau 3 năm",
     "chia di sản thừa kế cho 4 người", "làm thêm giờ lương 15 triệu", "thuế TNCN tháng này"
   - Kiểm tra điều kiện: "tên doanh nghiệp này có hợp lệ không", "đủ tuổi kết hôn chưa",
     "thời hiệu khởi kiện còn không", "điều khoản này có hiệu lực không"
   - Tra cứu văn bản cụ thể: "Điều 418 Bộ luật Dân sự nói gì", "có án lệ nào về trốn thuế",
     "phiên bản luật nào còn hiệu lực", "dẫn chiếu điều này tới điều nào"
   - Sinh văn bản mẫu: "sinh đơn khởi kiện đòi nợ", "tạo hợp đồng mua bán xe",
     "mẫu đơn ly hôn", "đơn khiếu nại hành chính"
   - Thẩm quyền/thủ tục theo bước: "ly hôn nộp ở đâu", "tòa nào thụ lý", "thủ tục thành lập DN cần gì"
   → Bất kỳ câu có ý "tính/kiểm tra/sinh/tra cứu điều X/án lệ/thủ tục bước/nộp ở đâu" → agent_tools
   → Mọi câu hỏi về THẨM QUYỀN/TÒA THỤ LÝ luôn là agent_tools (dù kèm "tranh chấp ..."),
     ví dụ "tranh chấp dân sự 600 triệu thì tòa cấp nào thụ lý?" → agent_tools
     (công cụ jurisdiction_resolver xác định tòa theo giá trị vụ việc).

2. legal_rag — CÂU HỎI VỀ NỘI DUNG LUẬT (giải thích, quyền, nghĩa vụ, khái niệm pháp lý):
   - "Quyền của người lao động theo Luật Lao động", "nghĩa vụ cấp dưỡng theo BLDS"
   - Giải thích khái niệm: "ủy quyền là gì", "diện trừ khỏi di sản là gì"
   - Hỏi nội dung điều khoản nói chung (KHÔNG kèm yêu cầu tính/sinh/tra cứu công cụ)

3. web_search — THÔNG TIN MỚI / THỜI SỰ (năm {current_year} hoặc vài tháng gần):
   - "Luật Đất đai mới nhất có gì thay đổi", "lương tối thiểu vùng năm nay",
     "văn bản pháp luật vừa ban hành", "số liệu GDP/lạm phát hiện nay"

4. general_chat — CHÀO HỎI, CẢM ƠN, OFF-TOPIC (không liên quan pháp luật):
   - "xin chào", "cảm ơn bạn", "bạn làm được gì", thời tiết, thể thao, giải trí

NGUYÊN TẮC XUNG ĐỘT:
- Phân vân giữa agent_tools và legal_rag → chọn agent_tools (agent có thể handoff sang RAG nếu cần văn bản; ngược lại RAG không có tool tính).
- Câu vừa hỏi nội dung luật vừa yêu cầu tính/sinh/kiểm tra → agent_tools.
- Câu follow-up tham chiếu câu trước ("cái đó tính sao", "khoản kia nữa") → nếu luồng trước là pháp luật và có ý tính → agent_tools; nếu chỉ hỏi lại nội dung → giữ route trước.
- Câu pháp luật mơ hồ (không rõ cần tính hay tra cứu) → agent_tools (an toàn hơn vì có tool + handoff).
- Câu có "mới nhất"/"năm nay"/"hiện nay" hỏi SỐ LIỆU THỜI SỰ (lương tối thiểu vùng,
  GDP, lạm phát, thống kê, văn bản vừa ban hành) → web_search, ƯU TIÊN hơn agent_tools
  (không có tool tính các số liệu này, cần dữ liệu internet cập nhật).

QUY TRÌNH: suy nghĩ 1-2 câu ngắn lý do, rồi trả về route. KHÔNG giải thích dài.

Trả về ĐÚNG định dạng:
<reasoning>1-2 câu lý do ngắn</reasoning>
<route>tên_route</route>"""

    openai_messages = [
        {
            "role": "system",
            "content": (
                "Bạn là bộ định tuyến chính xác cho chatbot pháp luật Việt Nam. "
                "Chỉ trả về đúng định dạng <reasoning>...</reasoning><route>...</route> "
                "với route là một trong: agent_tools, legal_rag, web_search, general_chat."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"Routing query: {message}")

    try:
        raw = openai_chat_complete(openai_messages, temperature=0.0).strip()

        # Structured parse: extract <route> tag (robust to LLM formatting drift).
        route_match = re.search(
            r"<route>\s*(legal_rag|agent_tools|web_search|general_chat)\s*</route>",
            raw,
            re.IGNORECASE,
        )
        if route_match:
            route = route_match.group(1).lower()
            logger.info(f"Detected route: {route} | reasoning: {raw[:200]}")
            _record_route(route)
            return route

        # Fallback: scan raw text for any valid route token (last-resort safety net).
        valid_routes = ["agent_tools", "legal_rag", "web_search", "general_chat"]
        raw_lower = raw.lower()
        for r in valid_routes:
            if r in raw_lower:
                logger.warning(f"Fallback scan route: {r} (no <route> tag) | raw: {raw[:200]}")
                _record_route(r)
                return r

        # Truly unparseable — default to agent_tools (safe: has tools + RAG handoff).
        logger.warning(f"Unparseable route output, defaulting to agent_tools | raw: {raw[:200]}")
        _record_route("agent_tools")
        return "agent_tools"

    except Exception as e:
        logger.error(f"Error detecting route: {e}")
        # Default to agent_tools: it can answer via tools and hand off to RAG/web.
        _record_route("agent_tools")
        return "agent_tools"


def get_financial_tools():
    tools = []
    logger.info(f"Financial tools: {tools}")
    return tools


def get_financial_agent_answer(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()

    # Execute the chat completion request
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    # Attempt to extract response details
    if not resp.choices:
        logger.error("No choices available in the response.")
        return {
            "role": "assistant",
            "content": "An error occurred, please try again later.",
        }

    choice = resp.choices[0]
    return choice


def convert_tool_calls_to_json(tool_calls):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "arguments": json.dumps(call.function.arguments),
                    "name": call.function.name,
                },
            }
            for call in tool_calls
        ],
    }


def get_financial_agent_handle(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()
    choice = get_financial_agent_answer(messages, model, tools)

    resp_content = choice.message.content
    resp_tool_calls = choice.message.tool_calls
    # Prepare the assistant's message
    if resp_content:
        return resp_content

    elif resp_tool_calls:
        logger.info(f"Process the tools call: {resp_tool_calls}")
        # List to hold tool response messages
        tool_messages = []
        # Iterate through each tool call and execute the corresponding function
        for tool_call in resp_tool_calls:
            # Display the tool call details
            logger.info(
                f"Tool call: {tool_call.function.name}({tool_call.function.arguments})"
            )
            # Retrieve the tool function from available tools
            tool = available_tools[tool_call.function.name]
            # Parse the arguments for the tool function
            tool_args = json.loads(tool_call.function.arguments)
            # Execute the tool function and get the result
            result = tool(**tool_args)
            tool_args["result"] = result
            # Append the tool's response to the tool_messages list
            tool_messages.append(
                {
                    "role": "tool",  # Indicate this message is from a tool
                    "content": json.dumps(tool_args),  # The result of the tool function
                    "tool_call_id": tool_call.id,  # The ID of the tool call
                }
            )
        # Update the new message to get response from LLM
        # Append the tool messages to the existing messages
        # Check here: https://platform.openai.com/docs/guides/function-calling
        next_messages = (
            messages + [convert_tool_calls_to_json(resp_tool_calls)] + tool_messages
        )
        return get_financial_agent_handle(next_messages, model, tools)
    else:
        raise InvalidResponse(f"The response is invalid: {choice}")