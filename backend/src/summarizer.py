try:
    from langchain_core.messages import HumanMessage
except ImportError:
    # Backward compatibility for older langchain versions.
    from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
import os

# load the model with Groq
groq_api_key = os.environ.get("GROQ_API_KEY")
summarizer_model = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0
) if groq_api_key else None


def summarize_text(text):
    if not summarizer_model:
        return "Error: Groq API key not configured for summarization"
    
    # prepare template for prompt
    template = """You are a very good assistant that summarizes text.

    Always keep important key points in the summary.

    ==================
    {text}
    ==================

    Write a summary of the content in Vietnamese.
    """

    prompt = template.format(text=text)

    messages = [HumanMessage(content=prompt)]
    try:
        # Sử dụng .invoke hoặc .chat tuỳ theo API của ChatGroq
        if hasattr(summarizer_model, "invoke"):
            summary = summarizer_model.invoke(messages)
        elif hasattr(summarizer_model, "chat"):
            summary = summarizer_model.chat(messages=messages)
        else:
            raise Exception("summarizer_model does not support .invoke or .chat method")
        return summary.content if hasattr(summary, "content") else str(summary)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Summarization error: {e}")
        return f"Error: Unable to summarize - {str(e)}"