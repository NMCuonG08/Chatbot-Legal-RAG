
import os
import sys

import gradio as gr
import requests

# Make frontend/ importable so we can reuse the shared citation renderer.
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

from citation_render import CITATION_CSS, render_answer_html, render_sources_panel

# Đổi endpoint cho đúng backend FastAPI
API_URL = "http://localhost:8002/chat/complete"  # Đúng port backend

def format_sources_markdown(content, sources):
    """Render the answer with inline [n] citation links + a grouped sources
    panel (split by kind: corpus vs web search) below the answer. Replaces the
    old disconnected "Tài liệu 1/2/3..." dump — each [n] is keyed to its source
    card in the panel, and web-search results are visually separated from
    in-corpus legal documents."""
    html_body = render_answer_html(content, sources or [])
    panel = render_sources_panel(sources or [])
    return (
        html_body
        + '<hr style="margin:14px 0;border:none;border-top:1px solid #e5e7eb;">'
        + '<div style="font-weight:700;margin-bottom:8px;">📚 Nguồn & Dẫn chứng</div>'
        + panel
    )

def chat_fn(message, history):
    # Chuyển history sang list các dict role/content nếu chưa đúng
    formatted_history = []
    if history:
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                formatted_history.append(item)
            elif isinstance(item, list) and len(item) == 2:
                formatted_history.append({"role": "user", "content": item[0]})
                formatted_history.append({"role": "assistant", "content": item[1]})

    payload = {
        "bot_id": "botLawyer",
        "user_id": "anonymous",
        "user_message": message,
        "sync_request": False,  # Để backend xử lý qua Celery
        # Nếu backend cần history thì truyền thêm, còn không thì bỏ dòng này
        # "history": formatted_history,
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            # Nếu trả về trực tiếp (sync), lấy luôn response và nguồn
            if "response" in data:
                answer_text = data["response"]
                sources = data.get("sources", [])
                answer = format_sources_markdown(answer_text, sources)
            # Nếu trả về task_id (async), poll kết quả
            elif "task_id" in data:
                task_id = data["task_id"]
                poll_url = f"http://localhost:8002/chat/complete/{task_id}"
                import time
                for _ in range(60):  # Poll tối đa 30s (60 lần, mỗi lần 0.5s)
                    poll_resp = requests.get(poll_url)
                    if poll_resp.status_code == 200:
                        poll_data = poll_resp.json()
                        if poll_data.get("task_status") == "SUCCESS":
                            result = poll_data.get("task_result", "Không nhận được phản hồi từ server.")
                            # Xử lý kết quả trả về từ Celery (chứa cả content và sources)
                            answer = extract_content_from_result(result)
                            break
                        elif poll_data.get("task_status") == "FAILURE":
                            answer = f"Lỗi xử lý: {poll_data.get('task_result', '')}"
                            break
                    time.sleep(0.5)
                else:
                    answer = "Hệ thống bận hoặc timeout, vui lòng thử lại."
            else:
                answer = "Không nhận được phản hồi từ server."
        else:
            answer = f"Lỗi server: {response.status_code}"
    except Exception as e:
        answer = f"Lỗi kết nối: {e}"
    return answer

# Hàm xử lý kết quả Celery trả về, lấy nội dung trả lời và định dạng nguồn tài liệu
def extract_content_from_result(result):
    # Nếu là dict có 'content' và 'sources'
    if isinstance(result, dict):
        content = result.get('content', '')
        sources = result.get('sources', [])
        return format_sources_markdown(content, sources)
    # Nếu là tuple/list, lấy phần tử cuối cùng
    if isinstance(result, (tuple, list)):
        if result:
            return extract_content_from_result(result[-1])
        return "Không nhận được phản hồi từ server."
    # Nếu là chuỗi
    if isinstance(result, str):
        return result
    # Nếu là object khác
    return str(result)


with gr.Blocks() as demo:
    gr.Markdown("""
    <h2 style='text-align:center;margin-bottom:8px;'>Legal Chatbot</h2>
    <div style='text-align:center;color:#888;margin-bottom:18px;'>Hỏi đáp pháp lý tiếng Việt</div>
    """)
    chatbot = gr.Chatbot(
        show_label=False,
        height=420,
        render_markdown=True,
        elem_id="chatbot-box",
    )
    msg = gr.Textbox(
        placeholder="Nhập câu hỏi...",
        label="",
        elem_id="chatbox-input",
        autofocus=True,
    )

    def user_chat(message, history):
        if not isinstance(history, list):
            history = []
        answer = chat_fn(message, history)
        # Đảm bảo history là list các dict {'role', 'content'}
        formatted_history = []
        if history and isinstance(history[0], dict):
            formatted_history = history.copy()
        elif history and isinstance(history[0], list):
            # Nếu là list các cặp [user, assistant], chuyển sang dict
            for pair in history:
                if isinstance(pair, list) and len(pair) == 2:
                    formatted_history.append({"role": "user", "content": pair[0]})
                    formatted_history.append({"role": "assistant", "content": pair[1]})
        # Thêm lượt chat mới
        formatted_history.append({"role": "user", "content": message})
        formatted_history.append({"role": "assistant", "content": answer})
        return formatted_history

    msg.submit(user_chat, [msg, chatbot], [chatbot])

    gr.HTML("""
    <style>
    #chatbot-box .avatar, #chatbot-box .bubble .icon, #chatbot-box .bubble .avatar {display:none !important;}
    #chatbot-box .bubble {border-radius: 12px; padding: 10px 16px; font-size: 16px;}
    #chatbox-input textarea {font-size: 16px; border-radius: 8px;}
    .gr-button {background: #2563eb; color: #fff; border-radius: 8px;}
    </style>
    """ + CITATION_CSS)

demo.launch(share=True, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray"))