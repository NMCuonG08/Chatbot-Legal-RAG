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
from database import get_celery_app
from models import (
    ensure_database_schema,
    get_conversation_messages,
    update_chat_conversation,
)
from query_rewriter import rewrite_query_to_multi_queries, rewrite_query_with_context
from rerank import rerank_documents
from search import hybrid_search, search_engine  # Import new hybrid search
from splitter import split_document
from summarizer import summarize_text
from tavily_tool import tavily_search_legal
from utils import setup_logging
from vectorize import add_vector, search_vector

setup_logging()
logger = logging.getLogger(__name__)


class ChatGraphState(TypedDict, total=False):
    history: List[Dict]
    question: str
    route: str
    standalone_question: str
    response: str

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


def generate_rag_answer(history, question):
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

    system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp
2. Trích dẫn chính xác các điều khoản, khoản, điểm từ văn bản pháp luật
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu thông tin không đủ trong tài liệu, hãy nói rõ điều đó
5. Luôn đưa ra câu trả lời có căn cứ pháp lý

QUAN TRỌNG: Chỉ sử dụng thông tin từ các tài liệu được cung cấp bên dưới."""

    doc_context = gen_doc_prompt(ranked_docs)

    openai_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [
            {
                "role": "user",
                "content": f"{doc_context}\n\nCâu hỏi: {question}\n\nHãy trả lời dựa trên các tài liệu pháp luật trên.",
            }
        ]
    )

    logger.info(f"Sending {len(openai_messages)} messages to Vietnamese LLM")
    assistant_answer = vietnamese_llm_chat_complete(openai_messages)
    logger.info("Bot RAG reply generated successfully")
    return assistant_answer


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
        return {"response": generate_rag_answer(history, question)}

    def agent_tools_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question", state.get("question", ""))
        return {"response": generate_agent_answer(history, question)}

    def web_search_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("standalone_question", state.get("question", ""))
        return {"response": generate_web_search_answer(history, question)}

    def general_chat_node(state: ChatGraphState):
        history = state.get("history", [])
        question = state.get("question", "")
        return {"response": generate_general_answer(history, question)}

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


def run_chat_graph(history, question):
    graph = get_chat_graph()
    result = graph.invoke({"history": history, "question": question})
    return result.get("response", "")


@shared_task()
def bot_rag_answer_message(history, question):
    standalone_question = follow_up_question(history, question)
    return generate_rag_answer(history, standalone_question)


def index_document_v2(id, question, content, collection_name=DEFAULT_COLLECTION_NAME):
    text = question + " " + content
    nodes = split_document(text)
    status_list = []
    for node in nodes:
        vector = get_embedding(node.text)
        add_vector_status = add_vector(
            collection_name=collection_name,
            vectors={
                id: {
                    "vector": vector,
                    "payload": {"question": question, "content": node.text},
                }
            },
        )
        status_list.append(add_vector_status)
    logger.info(f"Add vector status: {status_list}")
    return status_list


def get_summarized_response(response):
    output = summarize_text(response)
    logger.info("Summarized response: %s", output)
    return output


@shared_task()
def bot_route_answer_message(history, question):
    return run_chat_graph(history, question)


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

    # Use LangGraph-based routing to handle the question.
    response = run_chat_graph(history, question)
    logger.info(f"Chatbot response generated")

    # Summarize response for storage (optional, can be disabled if not needed)
    try:
        summarized_response = get_summarized_response(response)
    except Exception as e:
        logger.warning(f"Failed to summarize response: {e}, using original")
        summarized_response = response

    # Save response to history
    update_chat_conversation(bot_id, user_id, summarized_response, False)

    # Return full response
    return {"role": "assistant", "content": response}