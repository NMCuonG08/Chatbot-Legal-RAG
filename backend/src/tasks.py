import asyncio
from functools import lru_cache
import logging
from copy import copy
from pathlib import Path
from typing import Dict, List, TypedDict

from celery import shared_task
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from agent import ai_agent_handle
from brain import (
    detect_route,
    detect_user_intent,
    gen_doc_prompt,
    get_embedding,
    openai_chat_complete,
    vietnamese_llm_chat_complete,
)
from config import DEFAULT_COLLECTION_NAME
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
from utils import setup_logging
from vectorize import add_vector, search_vector, delete_vectors_by_ids

setup_logging()
logger = logging.getLogger(__name__)

import redis
redis_client = redis.from_url(settings.redis_url)

from guardrails_manager import LegalGuardrailsManager
guardrails_manager = LegalGuardrailsManager()


class ChatGraphState(TypedDict, total=False):
    history: List[Dict]
    question: str
    route: str
    standalone_question: str
    response: str
    sources: List[Dict]
    user_id: str

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

    Args:
        queries: List of query variations
        top_k: Number of documents to retrieve

    Returns:
        Merged and deduplicated list of documents with hybrid scores
    """
    all_docs = []
    seen_contents = set()

    for query in queries:
        logger.info(f"Hybrid search for query: {query}")

        # Use hybrid search instead of pure vector search
        docs = hybrid_search(query, limit=top_k)

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

    # Episodic memory retrieval: Retrieve relevant user history from user_episodes collection
    episodic_context = ""
    if user_id:
        try:
            logger.info(f"Retrieving episodic memory for user: {user_id}")
            # Generate query embedding
            query_vector = get_embedding(standalone_question)
            episodes = search_vector(
                collection_name="user_episodes",
                vector=query_vector,
                limit=3,
                filters={"user_id": user_id}
            )
            if episodes:
                logger.info(f"Retrieved {len(episodes)} episodes for user: {user_id}")
                episodes_list = []
                for ep in episodes:
                    summary = ep.get("content") or ep.get("summary") or ""
                    if summary:
                        episodes_list.append(f"- {summary}")
                if episodes_list:
                    episodic_context = "\n".join(episodes_list)
        except Exception as ep_err:
            logger.error(f"Failed to retrieve episodic memory for user {user_id}: {ep_err}")

    system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp
2. Trích dẫn chính xác các điều khoản, khoản, điểm từ văn bản pháp luật
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu thông tin không đủ trong tài liệu, hãy nói rõ điều đó
5. Luôn đưa ra câu trả lời có căn cứ pháp lý

QUAN TRỌNG: Chỉ sử dụng thông tin từ các tài liệu được cung cấp bên dưới."""

    doc_context = gen_doc_prompt(ranked_docs)

    user_context_block = ""
    if episodic_context:
        user_context_block = f"\n\nThông tin lịch sử liên quan đến người dùng trong các cuộc trò chuyện trước đây:\n{episodic_context}\n"

    openai_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [
            {
                "role": "user",
                "content": f"Tài liệu tham khảo:\n{doc_context}{user_context_block}\n\nCâu hỏi: {question}\n\nHãy trả lời dựa trên các tài liệu pháp luật trên.",
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
            assistant_answer = asyncio.run(
                guardrails_manager.verify_output_rag(assistant_answer, doc_context)
            )
        except Exception as e:
            logger.error(f"Error running output RAG guardrails: {e}")
            
    return assistant_answer, ranked_docs


def generate_agent_answer(history, question):
    logger.info("Using ReAct agent with tools")
    return ai_agent_handle(question)


def generate_web_search_answer(history, question):
    logger.info("Using Tavily web search for query")
    search_results = tavily_search_legal(question, max_results=5)

    system_prompt = """Bạn là trợ lý AI giúp tìm kiếm thông tin pháp luật trên internet.
    Hãy tổng hợp và trả lời câu hỏi dựa trên kết quả tìm kiếm được cung cấp."""

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
- Hướng dẫn thủ tục pháp lý"""

    openai_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

    return vietnamese_llm_chat_complete(openai_messages)


def _build_chat_graph():
    graph = StateGraph(ChatGraphState)

    def route_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("question", "")
        route = detect_route(history, question)
        standalone_question = question

        if route != "general_chat":
            standalone_question = follow_up_question(history, question)

        return {
            "route": route,
            "standalone_question": standalone_question,
        }

    def legal_rag_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question", state.get("question", ""))
        user_id = state.get("user_id")
        resp, sources = generate_rag_answer(history, question, user_id=user_id)
        if guardrails_manager.initialized:
            resp = guardrails_manager.add_legal_disclaimer(resp)
        return {"response": resp, "sources": sources}

    def agent_tools_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question", state.get("question", ""))
        resp = generate_agent_answer(history, question)
        if guardrails_manager.initialized:
            resp = guardrails_manager.add_legal_disclaimer(resp)
        return {"response": resp, "sources": []}

    def web_search_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question", state.get("question", ""))
        resp = generate_web_search_answer(history, question)
        if guardrails_manager.initialized:
            resp = guardrails_manager.add_legal_disclaimer(resp)
        return {"response": resp, "sources": []}

    def general_chat_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("question", "")
        return {"response": generate_general_answer(history, question), "sources": []}

    def choose_route(state: ChatGraphState):
        return state.get("route", "general_chat")

    graph.add_node("route", route_node)
    graph.add_node("legal_rag", legal_rag_node)
    graph.add_node("agent_tools", agent_tools_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("general_chat", general_chat_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        choose_route,
        {
            "legal_rag": "legal_rag",
            "agent_tools": "agent_tools",
            "web_search": "web_search",
            "general_chat": "general_chat",
        },
    )

    graph.add_edge("legal_rag", END)
    graph.add_edge("agent_tools", END)
    graph.add_edge("web_search", END)
    graph.add_edge("general_chat", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_chat_graph():
    return _build_chat_graph()


def run_chat_graph(history, question, user_id=None):
    graph = get_chat_graph()
    result = graph.invoke({"history": history, "question": question, "user_id": user_id})
    return {
        "response": result.get("response", ""),
        "sources": result.get("sources", []),
        "route": result.get("route", "")
    }


@shared_task()
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

        # 1. Delete orphaned chunks from Qdrant and database
        if to_delete:
            logger.info(f"Deleting {len(to_delete)} orphaned chunks for document {doc_id_str}")
            delete_vectors_by_ids(collection_name, to_delete)
            delete_doc_chunks_by_ids(to_delete, db=db)
            status_list.append({"action": "delete", "count": len(to_delete)})

        # 2. Embed and upsert added/modified chunks
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

            add_vector_status = add_vector(
                collection_name=collection_name,
                vectors=vectors_payload,
            )
            status_list.append({"action": "upsert", "status": add_vector_status})
            
            # Update full document hash record
            save_doc_chunk(doc_id_str, full_doc_cid, doc_hash, db=db)
        else:
            logger.info(f"Skipped indexing for document {doc_id_str} - no changes detected.")
            status_list.append({"action": "skip", "reason": "no_changes"})

        # Commit everything atomically!
        db.commit()
        logger.info(f"Successfully committed index transactions for document {doc_id_str}")

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


@shared_task()
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

        # Save to MySQL
        save_user_episode(user_id, summary_result)

        # Save to Qdrant
        vector = get_embedding(summary_result)
        point_id = str(uuid.uuid4())

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
def llm_handle_message(bot_id, user_id, question):
    """
    Main message handler with intelligent routing.

    Flow:
    1. Save user message to conversation history
    2. Load conversation context
    3. Route to appropriate handler (RAG, web search, or general chat)
    4. Generate and save response
    """
    logger.info("Start handle message")

    # Ensure required tables exist before worker writes conversation records.
    ensure_database_schema()

    # Update chat conversation
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)

    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[:-1]

    # 1. Check Semantic Cache
    try:
        from semantic_cache import get_cached_response, set_cached_response
        cached = get_cached_response(question)
        if cached:
            logger.info("Semantic Cache HIT - returning cached response directly")
            update_chat_conversation(bot_id, user_id, cached["response"], False)
            return {"role": "assistant", "content": cached["response"], "sources": cached["sources"]}
    except Exception as cache_err:
        logger.error(f"Semantic Cache check failed: {cache_err}")

    # 2. Run Input Guardrails check
    blocked_response = None
    if guardrails_manager.initialized:
        try:
            logger.info("Running input guardrails check...")
            blocked_response = asyncio.run(guardrails_manager.verify_input(question))
        except Exception as e:
            logger.error(f"Error running input guardrails: {e}")
            
    if blocked_response:
        response_text = blocked_response
        sources = []
        logger.info("Chatbot response generated (Blocked by Input Guardrails)")
    else:
        # 3. Use LangGraph-based routing to handle the question.
        graph_result = run_chat_graph(history, question, user_id=user_id)
        response_text = graph_result.get("response", "")
        sources = graph_result.get("sources", [])
        logger.info(f"Chatbot response generated")

    # Save full response to history (to preserve all details like the specific age)
    update_chat_conversation(bot_id, user_id, response_text, False)

    # 4. Save to Semantic Cache if not blocked
    if not blocked_response:
        try:
            from semantic_cache import set_cached_response
            set_cached_response(question, response_text, sources)
        except Exception as cache_err:
            logger.error(f"Semantic Cache save failed: {cache_err}")

    # Trigger background episodic memory ingestion task asynchronously!
    try:
        save_episodic_memory_task.delay(user_id, conversation_id)
        logger.info("Triggered save_episodic_memory_task asynchronously")
    except Exception as t_err:
        logger.error(f"Failed to trigger save_episodic_memory_task: {t_err}")

    # Return full response
    return {"role": "assistant", "content": response_text, "sources": sources}