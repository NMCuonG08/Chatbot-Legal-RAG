import logging
from pathlib import Path
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

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
            from nemoguardrails import RailsConfig, LLMRails
            
            config_dir = Path(__file__).parent.resolve() / "guardrails"
            if not config_dir.exists():
                logger.error(f"Guardrails config directory not found: {config_dir}")
                return
                
            logger.info(f"Loading NeMo Guardrails config from {config_dir}")
            config = RailsConfig.from_path(str(config_dir))
            self.rails = LLMRails(config)
            self.initialized = True
            logger.info("✅ NVIDIA NeMo Guardrails initialized successfully")
        except ImportError:
            logger.warning("⚠️ NeMo Guardrails not installed or failed to import. Running without rails.")
        except Exception as e:
            logger.error(f"❌ Error initializing NeMo Guardrails: {e}")

    async def verify_input(self, user_message: str) -> Optional[str]:
        """
        Validate user input against safety guidelines.
        Returns a blocked response message if validation fails, or None if safe.
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

        political_keywords = [
            "chính trị Việt Nam", "phản động", "nói xấu chính quyền", "bản đồ hình lưỡi bò",
            "hoàng sa trường sa của ai", "lách luật trốn thuế", "hối lộ cảnh sát", "chạy án",
            "hối lộ", "lách luật", "trốn thuế"
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
            
            # We pass the retrieved document context to check if output is grounded.
            context = {"context": doc_context}
            messages = [
                {"role": "context", "content": doc_context},
                {"role": "assistant", "content": response_text}
            ]
            
            # Use generate_async with the context passed
            if hasattr(self.rails, "generate_async"):
                response = await self.rails.generate_async(messages=messages, meta=context)
            else:
                response = self.rails.generate(messages=messages, meta=context)
                
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

    def add_legal_disclaimer(self, text: str) -> str:
        """Helper to append legal disclaimer."""
        disclaimer = "\n\n*Lưu ý: Thông tin trên chỉ mang tính chất tham khảo cứu pháp lý và không thay thế cho ý kiến chuyên môn của Luật sư.*"
        if disclaimer not in text:
            return text + disclaimer
        return text
