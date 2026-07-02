import contextvars
import json
import logging
import os
import requests

from groq import Groq
from redis import InvalidResponse

from custom_embedding import get_custom_embedding
from retry_utils import with_retry

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", default=None)
# Vietnamese LLM endpoint must be configured explicitly. No hardcoded public IP
# default — sending chat traffic to an unknown third-party box is a data-leak risk.
VIETNAMESE_LLM_API_URL = os.environ.get("VIETNAMESE_LLM_API_URL", default=None)

# Stores list of dicts: {"provider": str, "model": str, "prompt_tokens": int, "completion_tokens": int}
usage_accumulator = contextvars.ContextVar("usage_accumulator", default=None)


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

    def chat(self, messages, temperature=0.7, max_tokens=2048, raw=False):
        if not self.is_available():
            logger.warning("GROQ_API_KEY not set, cannot use Groq provider")
            return _USER_FACING_ERROR
        try:
            response = _groq_chat_create(self.model, messages, temperature, max_tokens)
            usage = getattr(response, "usage", None)
            if raw:
                record_usage(self.name, self.model, usage)
                return response.choices[0].message
            output = response.choices[0].message
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

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 120):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

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
    return GroqProvider(model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant"))


def build_ollama_provider(model: str | None = None) -> OllamaProvider:
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    # Cloud Ollama (e.g. http://ollama.com) requires https. An http URL triggers
    # a 301 redirect that `requests` follows by converting POST->GET, which then
    # 405s on /v1/chat/completions. Upgrade non-localhost http to https; keep
    # localhost as http (local Ollama has no TLS).
    if base.startswith("http://") and "localhost" not in base and "127.0.0.1" not in base:
        base = "https://" + base[len("http://"):]
    return OllamaProvider(
        base,
        model or os.environ.get("OLLAMA_LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or "llama3.1:latest",
        os.environ.get("OLLAMA_API_KEY"),
    )


def get_main_provider() -> LLMProvider:
    """Router: pick the main LLM provider from LLM_PROVIDER (groq|ollama).
    Default groq. Unknown values (including the legacy 'openai' branch) fall
    back to groq with a warning — OpenAI is not a class yet."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "ollama":
        return build_ollama_provider()
    if provider == "groq":
        return build_groq_provider()
    logger.warning(f"Unknown LLM_PROVIDER '{provider}', falling back to groq")
    return build_groq_provider()


def vietnamese_llm_chat_complete(messages=(), temperature=0.7, max_tokens=512):
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


def groq_chat_complete(messages=(), model=None, raw=False):
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
    logger.info(f"Chat complete using provider: {provider.name}, model: {provider.model}")
    return provider.chat(messages, raw=raw)


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
    """
    Document:
    Question: Trong Bộ luật Hình sự thì bao nhiêu tuổi được xem là người già...
    Answer: Người cao tuổi, người già...
    """
    doc_prompt = ""
    for doc in docs:
        doc_prompt += (
            f"Câu hỏi: {doc['question']} \n Câu trả lời: {doc['content']} \n\n"
        )

    return "Tài liệu tham khảo: \n{}".format(doc_prompt)


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

    # Improved prompt with agent_tools route
    user_prompt = f"""Bạn là hệ thống định tuyến thông minh cho chatbot tư vấn pháp luật Việt Nam. Phân tích câu hỏi và chọn công cụ phù hợp nhất.

Lịch sử hội thoại:
{history_text}

Câu hỏi hiện tại:
{message}

CÁC CÔNG CỤ KHẢ DỤNG:

1. "legal_rag" - Hệ thống RAG tra cứu văn bản pháp luật
   Sử dụng khi:
   - Hỏi về nội dung luật, nghị định, thông tư, quyết định
   - Hỏi về thủ tục pháp lý (ly hôn, thành lập DN, đăng ký đất đai)
   - Hỏi về quyền lợi, nghĩa vụ, trách nhiệm pháp lý
   - Câu hỏi về điều khoản cụ thể trong văn bản
   - Giải thích khái niệm pháp lý
   Ví dụ: "Thủ tục ly hôn theo Bộ luật Dân sự?", "Quyền của người lao động theo Luật Lao động?"

2. "agent_tools" - Agent với công cụ tính toán, tra cứu, xác thực và sinh văn bản
   Sử dụng khi:
   - Tính toán: phạt hợp đồng, chia thừa kế, trợ cấp thôi việc, làm thêm giờ,
     thuế TNCN, lệ phí trước bạ nhà đất/xe, án phí, cấp dưỡng con
   - Kiểm tra: tuổi pháp lý, tên doanh nghiệp, thời hiệu khởi kiện
   - Tra cứu văn bản: nội dung một điều luật, án lệ, dẫn chiếu, xác minh trích dẫn
   - Tra cứu bảng: thời hiệu, phiên bản luật, phạt vi phạm hành chính
   - Thủ tục & thẩm quyền: hướng dẫn thủ tục (ly hôn, thành lập DN, đăng ký đất,
     khiếu nại, khởi kiện), xác định tòa/cơ quan thụ lý
   - Sinh văn bản mẫu: đơn khởi kiện, đơn khiếu nại, hợp đồng mua bán, đơn ly hôn
   - Câu hỏi dạng "tính", "kiểm tra", "có hợp lệ không", "thủ tục...", "nộp ở đâu",
     "sinh đơn", "Điều X nói gì", "có án lệ nào"
   - Cần xử lý số liệu và logic phức tạp, nhiều bước suy luận
   Ví dụ: "Tính tiền phạt hợp đồng 100 triệu chậm 30 ngày với lãi 0.1%/ngày",
     "Thủ tục ly hôn cần giấy tờ gì, nộp ở đâu?", "Sinh đơn khởi kiện đòi nợ",
     "Điều 418 Bộ luật Dân sự nói gì?", "Trợ cấp thôi việc sau 3 năm làm việc"

3. "web_search" - Tìm kiếm web cho thông tin mới
   Sử dụng khi:
   - Tin tức, sự kiện pháp luật gần đây (trong vài tháng gần nhất)
   - Vụ án cụ thể đang diễn ra
   - Thống kê, số liệu hiện tại (GDP, lương tối thiểu, lạm phát)
   - Văn bản pháp luật MỚI vừa ban hành
   - Từ khóa: "mới nhất", "gần đây", "hiện nay", "năm 2024", "vừa ban hành"
   Ví dụ: "Luật Đất đai 2024 có gì mới?", "Lương tối thiểu vùng 1 năm 2024"

4. "general_chat" - Trò chuyện thông thường
   Sử dụng khi:
   - Chào hỏi: "xin chào", "hello", "hi"
   - Cảm ơn: "cảm ơn", "thanks"
   - Hỏi về bot: "bạn là ai", "bạn làm được gì"
   - Off-topic: không liên quan pháp luật (thời tiết, thể thao, giải trí)
   Ví dụ: "Xin chào", "Bạn có thể giúp gì?", "Cảm ơn bạn"

HƯỚNG DẪN PHÂN LOẠI:
1. Phân tích ý định chính của câu hỏi
2. Xác định xem cần tính toán/kiểm tra (→ agent_tools) hay tra cứu văn bản (→ legal_rag)
3. Nếu cần thông tin thời sự → web_search
4. Ưu tiên: agent_tools (có tính toán) > legal_rag (tra cứu) > web_search (tin mới) > general_chat
5. CHỈ trả về MỘT trong bốn giá trị: "legal_rag", "agent_tools", "web_search", "general_chat"
6. KHÔNG giải thích, KHÔNG thêm văn bản khác

Phân loại:"""

    openai_messages = [
        {
            "role": "system",
            "content": "Bạn là hệ thống định tuyến chính xác. Chỉ trả về một trong bốn giá trị: legal_rag, agent_tools, web_search, general_chat",
        },
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"Routing query: {message}")

    try:
        route = openai_chat_complete(openai_messages).strip().lower()

        # Validate route
        valid_routes = ["legal_rag", "agent_tools", "web_search", "general_chat"]
        if route not in valid_routes:
            # Try to extract valid route from response
            for valid_route in valid_routes:
                if valid_route in route:
                    route = valid_route
                    break
            else:
                # Default logic based on keywords
                message_lower = message.lower()

                # Check for calculation/validation keywords
                calc_keywords = [
                    "tính",
                    "kiểm tra",
                    "hợp lệ",
                    "đủ tuổi",
                    "chia",
                    "phạt",
                    "thời hiệu",
                ]
                if any(kw in message_lower for kw in calc_keywords):
                    logger.warning(
                        f"Invalid route '{route}', detected calculation keywords, using 'agent_tools'"
                    )
                    route = "agent_tools"
                else:
                    # Default to legal_rag for legal questions
                    logger.warning(
                        f"Invalid route '{route}', defaulting to 'legal_rag'"
                    )
                    route = "legal_rag"

        logger.info(f"Detected route: {route}")
        return route

    except Exception as e:
        logger.error(f"Error detecting route: {e}")
        # Default to legal_rag
        return "legal_rag"


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