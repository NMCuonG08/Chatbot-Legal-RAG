import logging
import re
from pathlib import Path
import asyncio
from typing import Dict, List, Any, Optional

from metacognitive import HIGH_STAKES_KEYWORDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII detection (homegrown regex, eval-time per P4). No presidio dependency.
# Returns a mapping of PII type -> list of matched spans. Pure + synchronous.
# ---------------------------------------------------------------------------
_PII_PATTERNS: Dict[str, re.Pattern] = {
    # Vietnamese ID card (CCCD): 9 or 12 digits, word-boundary guarded.
    "cccd": re.compile(r"\b(?:\d{9}|\d{12})\b"),
    # Phone: +84 or 0 prefix + 9-10 digits.
    "phone": re.compile(r"(?:\+84|0)\d{9,10}\b"),
    # Email.
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Vietnamese address cues.
    "address": re.compile(
        r"\b(?:số\s+\d+[\w/\-]*|tổ\s+\d+|khu\s+phố\s+\d+|phường\s+\w+"
        r"|xã\s+\w+|quận\s+\w+|huyện\s+\w+|tp\.?\s+\w+|thành phố\s+\w+)\b",
        re.IGNORECASE),
    # Tax code (MST): 10 digits + optional 3-digit suffix.
    "tax_code": re.compile(r"\b\d{10}(?:-\d{3})?\b"),
}


def detect_pii_vietnamese(text: str) -> Dict[str, List[str]]:
    """Return ``{pii_type: [matched_spans, ...]}`` found in ``text``.

    Pure: never logs matched values (callers log only counts). Empty dict when
    no PII is found. Spans deduped, order preserved.
    """
    if not text:
        return {}
    findings: Dict[str, List[str]] = {}
    for pii_type, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            seen = set()
            deduped = [m for m in matches if not (m in seen or seen.add(m))]
            findings[pii_type] = deduped
    return findings


def verify_output_pii(text: str) -> tuple:
    """Check agent output for PII leaks. Returns ``(blocked: bool, findings)``.

    ``blocked`` is True when any PII pattern matches. Findings map type->spans.
    """
    findings = detect_pii_vietnamese(text)
    return (bool(findings), findings)

class LegalGuardrailsManager:
    """
    Manager class for NVIDIA NeMo Guardrails in the Vietnamese Legal Chatbot.
    Protects inputs (jailbreak/off-topic/toxicity) and outputs (RAG hallucination/toxicity).
    """
    def __init__(self):
        self.rails = None
        self.initialized = False
        self._init_engine()

    def _init_engine(self):
        try:
            import os
            # Monkeypatch missing GoogleSearchAPIWrapper inside langchain_community.utilities
            # to satisfy NeMo Guardrails' buggy built-in action file imports.
            try:
                import langchain_community.utilities as utils
                if not hasattr(utils, "GoogleSearchAPIWrapper"):
                    class DummyGoogleSearchAPIWrapper:
                        def __init__(self, *args, **kwargs):
                            pass
                    setattr(utils, "GoogleSearchAPIWrapper", DummyGoogleSearchAPIWrapper)
            except Exception as mp_err:
                logger.debug(f"Langchain community monkeypatch warning: {mp_err}")

            from nemoguardrails import RailsConfig, LLMRails
            
            config_dir = Path(__file__).parent.resolve() / "guardrails"
            if not config_dir.exists():
                logger.error(f"Guardrails config directory not found: {config_dir}")
                return
                
            logger.info(f"Loading NeMo Guardrails config from {config_dir}")
            config = RailsConfig.from_path(str(config_dir))
            
            # --- Dynamic LLM Configuration via Environment Variables ---
            llm_provider = os.environ.get("LLM_PROVIDER", "groq").lower()
            llm_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
            if llm_provider == "groq" and "cloud" in llm_model.lower():
                llm_model = "llama-3.3-70b-versatile"
            
            logger.info(f"[GUARDRAILS] Configuring NeMo LLM engine - Provider: {llm_provider}, Model: {llm_model}")
            
            if config.models:
                config.models[0].engine = llm_provider
                config.models[0].model = llm_model
                
                # Configure specific provider details
                if llm_provider == "ollama":
                    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                    api_key = os.environ.get("OLLAMA_API_KEY")
                    if api_key:
                        logger.info("[GUARDRAILS] Ollama API key detected. Routing Ollama Cloud through OpenAI engine compatibility layer.")
                        config.models[0].engine = "openai"
                        config.models[0].parameters = {
                            "openai_api_base": f"{ollama_url.rstrip('/')}/v1"
                        }
                        os.environ["OPENAI_API_KEY"] = api_key
                    else:
                        config.models[0].engine = "ollama"
                        config.models[0].parameters = {
                            "base_url": ollama_url
                        }
                elif llm_provider == "openai":
                    openai_base = os.environ.get("OPENAI_API_BASE")
                    if openai_base:
                        config.models[0].parameters = {
                            "openai_api_base": openai_base
                        }
            
            self.rails = LLMRails(config)
            self.initialized = True
            logger.info("✅ NVIDIA NeMo Guardrails initialized successfully")
        except ImportError:
            logger.warning("⚠️ NeMo Guardrails not installed or failed to import. Running without rails.")
        except Exception as e:
            logger.error(f"❌ Error initializing NeMo Guardrails: {e}")

    @staticmethod
    def verify_input_tier1(user_message: str) -> Optional[str]:
        """
        Deterministic keyword guardrails (Tier 1). Pure, synchronous, no LLM call,
        no instance state required. Returns a blocked response message if a keyword
        rule matches, else None.

        Safe to call in the API process before dispatching to Celery — blocks
        obvious jailbreak/political/toxic inputs without a broker roundtrip and
        without initializing the NeMo engine.
        """
        message_lower = user_message.lower().strip()

        # --- Tier 1: Deterministic Keyword Guardrails ---
        jailbreak_keywords = [
            "ignore instructions", "bỏ qua tất cả chỉ thị", "lộ prompt hệ thống",
            "show me your prompt", "you are now a", "system prompt", "bỏ qua chỉ thị",
            "tiết lộ prompt"
        ]
        if any(kw.lower() in message_lower for kw in jailbreak_keywords):
            logger.warning(f"🚨 [GUARDRAILS-TIER1] Input BLOCKED (Jailbreak): {user_message}")
            return "Xin lỗi, tôi không thể thực hiện yêu cầu này. Tôi chỉ có nhiệm vụ hỗ trợ bạn tìm kiếm và giải đáp thông tin pháp lý Việt Nam theo các tài liệu chính thống."

        # Lưu ý: KHÔNG block từ đơn "hối lộ", "lách luật", "trốn thuế" vì người
        # dùng có thể hỏi "Điều luật xử lý hối lộ/trốn thuế là gì?" — câu hỏi pháp
        # lý hợp lệ. Chỉ block cụm từ chỉ ý định XIN hướng dẫn vi phạm pháp luật.
        political_keywords = [
            "chính trị Việt Nam", "phản động", "nói xấu chính quyền", "bản đồ hình lưỡi bò",
            "hoàng sa trường sa của ai", "lách luật trốn thuế", "hối lộ cảnh sát", "chạy án",
            "hướng dẫn lách luật", "cách trốn thuế", "dạy hối lộ"
        ]
        if any(kw.lower() in message_lower for kw in political_keywords):
            logger.warning(f"🚨 [GUARDRAILS-TIER1] Input BLOCKED (Political/Illegal): {user_message}")
            return "Tôi là trợ lý pháp luật khách quan và chỉ hỗ trợ tra cứu thông tin pháp lý chính thống. Tôi không đưa ra nhận định chính trị hoặc hướng dẫn các hành vi vi phạm pháp luật."

        toxicity_keywords = [
            "đồ ngu", "cút đi", "chửi thề", "địt", "lồn", "cặc", "đm", "vcl"
        ]
        if any(kw.lower() in message_lower for kw in toxicity_keywords):
            logger.warning(f"🚨 [GUARDRAILS-TIER1] Input BLOCKED (Toxicity): {user_message}")
            return "Tôi hỗ trợ giải đáp pháp luật dựa trên tinh thần lịch sự và chuyên nghiệp. Vui lòng đặt câu hỏi lịch sự để tôi hỗ trợ bạn tốt nhất."

        return None

    async def verify_input(self, user_message: str) -> Optional[str]:
        """
        Validate user input against safety guidelines.
        Returns a blocked response message if validation fails, or None if safe.

        Runs Tier 1 (deterministic keywords) then Tier 2 (semantic NeMo LLM).
        Tier 1 is also exposed synchronously via ``verify_input_tier1`` so the API
        layer can short-circuit obvious violations before enqueuing to Celery.
        """
        tier1 = self.verify_input_tier1(user_message)
        if tier1:
            return tier1

        # --- Tier 2: Semantic/NeMo Guardrails ---
        if not self.initialized or not self.rails:
            return None

        try:
            logger.info(f"[GUARDRAILS-TIER2] Checking input safety for: {user_message[:50]}...")
            
            messages = [{"role": "user", "content": user_message}]
            
            # Execute rails check (supporting both async and sync versions)
            if hasattr(self.rails, "generate_async"):
                response = await self.rails.generate_async(messages=messages)
            else:
                response = self.rails.generate(messages=messages)
                
            # If the rails engine intercepted and returned a canned bot message
            if response:
                if isinstance(response, dict):
                    content = response.get("content", "")
                elif hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)
                
                content = content.strip()
                # If the rails generated a response, it means it intercepted the message (e.g. safety block)
                if content:
                    logger.warning(f"🚨 [GUARDRAILS-TIER2] Input BLOCKED. Response: {content}")
                    return content
            
            return None
        except Exception as e:
            logger.error(f"Error checking input guardrails: {e}")
            return None

    async def verify_output_rag(self, response_text: str, doc_context: str) -> str:
        """
        Verify the generated RAG response against retrieved document context to prevent hallucinations.
        """
        if not self.initialized or not self.rails:
            return response_text

        try:
            logger.info("[GUARDRAILS] Running RAG Hallucination checks...")
            
            messages = [
                {"role": "context", "content": {"context": doc_context}},
                {"role": "assistant", "content": response_text}
            ]
            
            # Use generate_async with the context passed
            if hasattr(self.rails, "generate_async"):
                response = await self.rails.generate_async(messages=messages)
            else:
                response = self.rails.generate(messages=messages)
                
            if response:
                if isinstance(response, dict):
                    content = response.get("content", "")
                elif hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)
                
                if "không tìm thấy" in content or "no" in content.lower():
                    logger.warning("🚨 [GUARDRAILS] RAG output failed grounding checks (hallucination detected)")
                    return "Xin lỗi, tôi không thể tìm thấy câu trả lời chính xác có căn cứ trong tài liệu pháp lý được cung cấp."
                
            return response_text
        except Exception as e:
            logger.error(f"Error checking output hallucination guardrails: {e}")
            return response_text

    # Topics that require a licensed lawyer — the chatbot must escalate,
    # not advise. Canonical list now lives in metacognitive.HIGH_STAKES_KEYWORDS
    # (single source of truth, shared with the metacognitive graph node);
    # re-aliased here so existing self._ESCALATION_TOPICS references keep working.
    _ESCALATION_TOPICS = HIGH_STAKES_KEYWORDS

    def add_legal_disclaimer(self, text: str, question: Optional[str] = None) -> str:
        """Append a legal disclaimer; escalate to a lawyer referral when the
        user's question touches criminal-defense topics.

        Args:
            text: The assistant response to append the disclaimer to.
            question: Optional original user question. When it matches an
                escalation topic, a stronger lawyer-referral disclaimer is
                used instead of the generic informational one.
        """
        q = (question or "").lower()
        escalated = any(topic in q for topic in self._ESCALATION_TOPICS)

        if escalated:
            disclaimer = (
                "\n\n⚠️ *Đây là vấn đề thuộc diện cần Luật sư hành nghề (hình sự/bào chữa). "
                "Thông tin trên chỉ mang tính tham khảo, không thay thế ý kiến pháp lý chính thức. "
                "Vui lòng liên hệ Luật sư hoặc Trung tâm trợ giúp pháp lý nhà nước.*"
            )
        else:
            disclaimer = (
                "\n\n*Lưu ý: Thông tin trên chỉ mang tính chất tham khảo cứu pháp lý "
                "và không thay thế cho ý kiến chuyên môn của Luật sư.*"
            )

        if disclaimer not in text:
            return text + disclaimer
        return text
