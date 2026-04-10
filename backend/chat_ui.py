
import gradio as gr
import requests

# Đổi endpoint cho đúng backend FastAPI
API_URL = "http://localhost:8002/chat/complete"  # Đúng port backend

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
            # Nếu trả về trực tiếp (sync), lấy luôn response
            if "response" in data:
                answer = data["response"]
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
                            # Xử lý kết quả trả về từ Celery
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

# Hàm xử lý kết quả Celery trả về, luôn lấy nội dung trả lời cuối cùng
def extract_content_from_result(result):
    # Nếu là dict có 'content'
    if isinstance(result, dict):
        if 'content' in result:
            return result['content']
        if 'assistant' in result:
            return result['assistant']
        return str(result)
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
    """)

demo.launch(share=True, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray"))