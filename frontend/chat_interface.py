import json
import os
import time
import socket
from urllib.parse import urlparse

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8002")

@st.cache_data(ttl=5)
def check_services_health(backend_url):
    # Default statuses
    status_dict = {
        "backend": {"status": "offline", "label": "FastAPI Backend", "details": "Không thể kết nối"},
        "database": {"status": "offline", "label": "Database (SQL)", "details": "Chưa kiểm tra"},
        "qdrant": {"status": "offline", "label": "Qdrant DB", "details": "Chưa kiểm tra"},
        "redis": {"status": "offline", "label": "Redis Cache", "details": "Chưa kiểm tra"},
        "celery": {"status": "offline", "label": "Celery Worker", "details": "Chưa kiểm tra"},
        "ollama": {"status": "not_configured", "label": "Ollama LLM", "details": "Chưa cấu hình"}
    }
    
    def check_port(host, port, timeout=0.5):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False
            
    backend_up = False
    try:
        resp = requests.get(f"{backend_url}/health/detailed", timeout=4.0)
        if resp.status_code == 200:
            backend_up = True
            data = resp.json()
            status_dict["backend"] = {"status": "healthy", "label": "FastAPI Backend", "details": "Hoạt động"}
            
            # Database
            db_data = data.get("database", {})
            status_dict["database"] = {
                "status": db_data.get("status", "unhealthy"),
                "label": "Database (SQL)",
                "details": "Hoạt động" if db_data.get("status") == "healthy" else f"Lỗi: {db_data.get('error')}"
            }
            
            # Redis
            redis_data = data.get("redis", {})
            status_dict["redis"] = {
                "status": redis_data.get("status", "unhealthy"),
                "label": "Redis Cache",
                "details": "Hoạt động" if redis_data.get("status") == "healthy" else f"Lỗi: {redis_data.get('error')}"
            }
            
            # Qdrant
            qdrant_data = data.get("qdrant", {})
            status_dict["qdrant"] = {
                "status": qdrant_data.get("status", "unhealthy"),
                "label": "Qdrant DB",
                "details": "Hoạt động" if qdrant_data.get("status") == "healthy" else f"Lỗi: {qdrant_data.get('error')}"
            }
            
            # Celery
            celery_data = data.get("celery", {})
            celery_status = celery_data.get("status", "unhealthy")
            active_workers = celery_data.get("active_workers", [])
            details_str = "Hoạt động"
            if celery_status == "no_workers":
                details_str = "Không tìm thấy Worker"
            elif celery_status == "unhealthy":
                details_str = f"Lỗi: {celery_data.get('error')}"
            else:
                details_str = f"Hoạt động ({len(active_workers)} workers)"
                
            status_dict["celery"] = {
                "status": celery_status,
                "label": "Celery Worker",
                "details": details_str
            }
            
            # Ollama
            ollama_data = data.get("ollama", {})
            ollama_status = ollama_data.get("status", "not_configured")
            status_dict["ollama"] = {
                "status": ollama_status,
                "label": "Ollama LLM",
                "details": "Hoạt động" if ollama_status == "healthy" else ("Chưa cấu hình" if ollama_status == "not_configured" else f"Lỗi: {ollama_data.get('error')}")
            }
    except Exception as e:
        status_dict["backend"] = {"status": "unhealthy", "label": "FastAPI Backend", "details": f"Ngoại tuyến: {str(e)[:40]}"}

    # If backend is down, do direct TCP checks
    if not backend_up:
        db_url = "postgresql://postgres:cuong1182004@127.0.0.1:5432/legal_chatbot"
        redis_url = "redis://127.0.0.1:6379/0"
        qdrant_url = "http://localhost:6333"
        ollama_url = "http://localhost:11434"
        
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("DATABASE_URL="):
                            db_url = line.split("=", 1)[1]
                        elif line.startswith("REDIS_URL="):
                            redis_url = line.split("=", 1)[1]
                        elif line.startswith("QDRANT_URL="):
                            qdrant_url = line.split("=", 1)[1]
                        elif line.startswith("OLLAMA_BASE_URL="):
                            ollama_url = line.split("=", 1)[1]
            except Exception:
                pass
                
        def parse_host_port(url_str, default_host="127.0.0.1", default_port=80):
            try:
                parsed = urlparse(url_str)
                netloc = parsed.netloc
                if "@" in netloc:
                    netloc = netloc.split("@")[1]
                if ":" in netloc:
                    host, port_str = netloc.split(":")
                    return host, int(port_str)
                else:
                    return netloc or default_host, default_port
            except Exception:
                return default_host, default_port

        # Database TCP Check
        db_host, db_port = parse_host_port(db_url, "127.0.0.1", 5432)
        if check_port(db_host, db_port):
            status_dict["database"] = {"status": "healthy", "label": "Database (SQL)", "details": "Cổng TCP Mở"}
        else:
            status_dict["database"] = {"status": "unhealthy", "label": "Database (SQL)", "details": "Ngoại tuyến"}
            
        # Redis TCP Check
        redis_host, redis_port = parse_host_port(redis_url, "127.0.0.1", 6379)
        if check_port(redis_host, redis_port):
            status_dict["redis"] = {"status": "healthy", "label": "Redis Cache", "details": "Cổng TCP Mở"}
        else:
            status_dict["redis"] = {"status": "unhealthy", "label": "Redis Cache", "details": "Ngoại tuyến"}

        # Qdrant TCP Check
        qdrant_host, qdrant_port = parse_host_port(qdrant_url, "127.0.0.1", 6333)
        if check_port(qdrant_host, qdrant_port):
            status_dict["qdrant"] = {"status": "healthy", "label": "Qdrant DB", "details": "Cổng TCP Mở"}
        else:
            status_dict["qdrant"] = {"status": "unhealthy", "label": "Qdrant DB", "details": "Ngoại tuyến"}

        # Celery Check
        status_dict["celery"] = {"status": "unhealthy", "label": "Celery Worker", "details": "Ngoại tuyến (Backend Down)"}
        
        # Ollama Check
        ollama_host, ollama_port = parse_host_port(ollama_url, "127.0.0.1", 11434)
        if check_port(ollama_host, ollama_port):
            status_dict["ollama"] = {"status": "healthy", "label": "Ollama LLM", "details": "Cổng TCP Mở"}
        else:
            status_dict["ollama"] = {"status": "unhealthy", "label": "Ollama LLM", "details": "Ngoại tuyến"}

    return status_dict


st.set_page_config(page_title="Legal RAG & Agentic", page_icon="⚖️", layout="centered")
st.title("Legal RAG & Agentic Workflow")
st.caption("MVP chat UI with async task polling")

# Per-session identity = the name the user types in the sidebar. Each name
# scopes its own conversation / memory / episodic store, so the agent only
# ever sees that one session. The default is intentionally EMPTY (not a
# shared sentinel like "demo-session") so two different users cannot
# accidentally collapse into the same session and leak facts (e.g. names)
# to each other. The backend rejects an empty id, forcing the user to pick
# a unique name before chatting.
if "session_id" not in st.session_state:
    st.session_state.session_id = ""

# Sidebar settings & history
st.sidebar.title("⚙️ Cấu hình & Lịch sử")

# Service Health Monitor
with st.sidebar.expander("📊 Trạng thái Hệ thống", expanded=True):
    health_data = check_services_health(BACKEND_URL)
    for service_key, info in health_data.items():
        status = info["status"]
        label = info["label"]
        details = info["details"]
        
        if status == "healthy":
            icon = "🟢"
        elif status == "no_workers":
            icon = "🟡"
        elif status == "not_configured":
            icon = "⚪"
        else:
            icon = "🔴"
            
        st.markdown(f"{icon} **{label}**: {details}")
        
    if st.button("🔄 Làm mới trạng thái", key="refresh_health"):
        st.cache_data.clear()
        st.rerun()
st.sidebar.markdown("---")


# User ID input — each name is its own isolated session. Empty by default
# so no two users accidentally share a sentinel id. Only adopt the widget
# value when non-empty, so clearing the field mid-task does NOT reset
# session_id to "" and break the in-flight task poll (which still runs
# against the prior valid id).
_input_id = st.sidebar.text_input(
    "Tên người dùng / Session ID",
    value=st.session_state.session_id,
    placeholder="Nhập tên riêng (vd: snake) để bắt đầu",
    help="Mỗi tên là một phiên riêng. Đừng dùng chung tên nếu không muốn lộ thông tin.",
)
if _input_id.strip():
    st.session_state.session_id = _input_id.strip()

# Clear History button
if st.sidebar.button("🗑️ Xóa lịch sử cuộc trò chuyện"):
    try:
        resp = requests.delete(f"{BACKEND_URL}/history/{st.session_state.session_id}", timeout=10)
        if resp.status_code == 200:
            st.sidebar.success("Đã xóa sạch lịch sử trò chuyện!")
            st.rerun()
        else:
            st.sidebar.error("Không thể xóa lịch sử.")
    except Exception as e:
        st.sidebar.error(f"Lỗi: {e}")

# Fetch and show history in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Nhật ký hội thoại")
if not st.session_state.session_id:
    st.sidebar.caption("Nhập tên ở trên để xem lịch sử.")
else:
    try:
        hist_resp = requests.get(f"{BACKEND_URL}/history/{st.session_state.session_id}", timeout=5)
        if hist_resp.status_code == 200:
            history_data = hist_resp.json().get("history", [])
            if not history_data:
                st.sidebar.caption("Chưa có cuộc trò chuyện nào.")
            else:
                # We display messages in a readable way, latest first
                for idx, msg in enumerate(history_data[:15]):  # limit to last 15 messages
                    role = "👤 Bạn" if msg.get("is_request") else "⚖️ Trợ lý"
                    text = msg.get("message", "")
                    created_at = msg.get("created_at", "")
                    # Format time representation
                    time_str = ""
                    if created_at and "T" in created_at:
                        time_str = created_at.split("T")[1][:5]
                    elif created_at and " " in created_at:
                        time_str = created_at.split(" ")[1][:5]
                    st.sidebar.markdown(f"**{role}** {f'({time_str})' if time_str else ''}: {text[:150]}...")
        else:
            st.sidebar.caption("Không thể tải lịch sử.")
    except Exception as e:
        st.sidebar.caption(f"Không thể kết nối đến Backend: {e}")

def format_event(evt):
    node = evt.get("node", "?")
    payload = evt.get("payload", {})
    
    emoji = "⚙️"
    node_name = node.upper()
    details = ""
    
    if node == "route":
        emoji = "🔍"
        node_name = "Định hướng (Route)"
        route_val = payload.get("route", "")
        if route_val:
            details = f" -> Lựa chọn luồng xử lý: **{route_val}**"
    elif node == "retrieve":
        emoji = "📖"
        node_name = "Truy vấn (Retrieve)"
        doc_count = payload.get("doc_count", 0)
        query_val = payload.get("query", "")
        if query_val:
            details = f" -> Tìm kiếm tài liệu pháp luật cho: *'{query_val[:80]}...'* ({doc_count} tài liệu)"
        else:
            details = f" -> Đang tìm kiếm tài liệu pháp luật... ({doc_count} tài liệu)"
    elif node == "grade_documents":
        emoji = "⚖️"
        node_name = "Thẩm định (Grade)"
        relevant = payload.get("relevant", 0)
        total = payload.get("total", 0)
        details = f" -> Thẩm định tài liệu: phát hiện **{relevant}/{total}** tài liệu liên quan đạt yêu cầu"
    elif node == "rewrite_query":
        emoji = "✍️"
        node_name = "Tối ưu hóa (Rewrite)"
        rewritten = payload.get("rewritten", "")
        if rewritten:
            details = f" -> Tối ưu câu hỏi tìm kiếm: *'{rewritten[:80]}...'*"
    elif node == "web_search":
        emoji = "🌐"
        node_name = "Tìm kiếm Web (Web Search)"
        web_query = payload.get("web_query", "")
        if web_query:
            details = f" -> Tìm kiếm bổ sung trên internet với: *'{web_query[:80]}...'*"
    elif node == "generate":
        emoji = "🤖"
        node_name = "Tổng hợp (Generate)"
        details = " -> Đang tổng hợp và soạn thảo phản hồi pháp lý..."
        
    return f"{emoji} **{node_name}**{details}"

prompt = st.text_area("Nhap cau hoi", placeholder="Vi du: Hay tom tat quy dinh moi ve hop dong lao dong")

if st.button("Gui") and prompt.strip():
    if not st.session_state.session_id.strip():
        st.error("Vui lòng nhập 'Tên người dùng / Session ID' ở thanh bên trước khi gửi.")
        st.stop()
    try:
        submit_resp = requests.post(
            f"{BACKEND_URL}/chat/complete",
            json={"user_id": st.session_state.session_id, "user_message": prompt.strip()},
            timeout=20,
        )
        submit_resp.raise_for_status()
        payload = submit_resp.json()

        # Guardrail Tier1 block (jailbreak/toxic): backend returns the blocked
        # message directly with no task_id. Display it instead of crashing.
        if "task_id" not in payload:
            blocked_msg = payload.get("response") or "Yêu cầu bị chặn bởi bộ bảo vệ."
            with st.chat_message("assistant"):
                st.markdown(blocked_msg)
            st.stop()

        task_id = payload["task_id"]

        st.info(f"Task queued: {task_id}")

        # Realtime trace display
        trace_messages = []
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Đang chuẩn bị chạy Agent...")
        
        trace_expander = st.expander("🔄 Trình tự thực thi Agent (Realtime)", expanded=True)
        trace_placeholder = trace_expander.empty()
        
        final_data = None
        
        try:
            # We connect to the stream endpoint, retrying on 404 in case of Celery worker initialization delays
            stream_resp = None
            for attempt in range(25):  # Retry for up to ~125 seconds
                try:
                    resp = requests.get(
                        f"{BACKEND_URL}/chat/stream/{task_id}",
                        stream=True,
                        timeout=(5, 60)
                    )
                    if resp.status_code == 200:
                        stream_resp = resp
                        break
                    elif resp.status_code == 404:
                        status_placeholder.info(f"⏳ Đang xếp hàng và thẩm định an toàn (Lượt thử {attempt+1}/25)...")
                        time.sleep(5)
                    else:
                        resp.raise_for_status()
                except Exception as stream_conn_err:
                    if attempt == 24:
                        raise stream_conn_err
                    time.sleep(2)
            
            if stream_resp and stream_resp.status_code == 200:
                status_placeholder.info("🚀 Agent đang thực thi workflow...")
                
                for raw in stream_resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("data:"):
                        data_str = raw[len("data:"):].strip()
                        try:
                            evt = json.loads(data_str)
                        except (json.JSONDecodeError, ValueError):
                            continue
                            
                        if "event_type" not in evt:
                            continue
                            
                        # Format and add to trace messages
                        formatted = format_event(evt)
                        if formatted:
                            trace_messages.append(formatted)
                            # Render live markdown bullet list
                            trace_placeholder.markdown("\n".join([f"- {m}" for m in trace_messages]))
                            
                        # Break loop on run end
                        if evt.get("event_type") == "run_end":
                            break
            else:
                trace_placeholder.caption("⚠️ Không thể kết nối với luồng trace thời gian thực. Đang chuyển sang chế độ chờ kết quả...")
                            
        except Exception as e:
            trace_placeholder.caption(f"Không thể đọc kết nối realtime: {e}. Đang chờ tác vụ hoàn thành...")
            
        # Fallback / Final complete poll
        for _ in range(30):
            task_resp = requests.get(f"{BACKEND_URL}/chat/complete/{task_id}", timeout=20)
            task_resp.raise_for_status()
            result_data = task_resp.json()
            if result_data.get("task_status") == "SUCCESS":
                final_data = result_data
                break
            elif result_data.get("task_status") == "FAILURE":
                status_placeholder.error("❌ Tác vụ thất bại trong quá trình thực thi.")
                break
            time.sleep(1)

        if final_data is None:
            status_placeholder.warning("⚠️ Tác vụ chưa hoàn thành. Vui lòng thử lại sau.")
        else:
            status_placeholder.success("✅ Đã hoàn thành!")
            task_result = final_data.get("task_result", {})
            st.markdown("### ⚖️ Trả lời từ Trợ lý Pháp lý:")
            st.write(task_result.get("content", "No result"))
            
            # Display sources if available
            sources = task_result.get("sources", [])
            if sources:
                with st.expander("📚 Nguồn tài liệu tham khảo"):
                    for idx, src in enumerate(sources):
                        content_text = src.get('content') or src.get('text') or ''
                        st.markdown(f"**Tài liệu {idx+1}:** {content_text[:300]}...")

            # Phase 4 — RLHF 👍/👎 feedback. Sent per-user (session_id) so the
            # backend can store it user-scoped and reuse good answers as
            # few-shot / rerank signal. Sentinel/empty ids are rejected server-side.
            _resp_text = task_result.get("content", "")
            fb_col1, fb_col2, _ = st.columns([1, 1, 6])
            with fb_col1:
                if st.button("👍", key=f"fb_good_{task_id}", help="Câu trả lời này tốt"):
                    try:
                        requests.post(
                            f"{BACKEND_URL}/feedback",
                            json={
                                "user_id": st.session_state.session_id,
                                "conversation_id": task_id,
                                "message_id": task_id,
                                "rating": "good",
                                "question": prompt.strip(),
                                "response": _resp_text,
                                "sources": sources,
                            },
                            timeout=10,
                        )
                        st.toast("Cảm ơn bạn đã đánh giá! 👍")
                    except requests.RequestException as fb_exc:
                        st.toast(f"Lỗi gửi đánh giá: {fb_exc}")
            with fb_col2:
                if st.button("👎", key=f"fb_bad_{task_id}", help="Câu trả lời này chưa tốt"):
                    try:
                        requests.post(
                            f"{BACKEND_URL}/feedback",
                            json={
                                "user_id": st.session_state.session_id,
                                "conversation_id": task_id,
                                "message_id": task_id,
                                "rating": "bad",
                                "question": prompt.strip(),
                                "response": _resp_text,
                                "sources": sources,
                            },
                            timeout=10,
                        )
                        st.toast("Cảm ơn phản hồi — chúng tôi sẽ cải thiện. 👎")
                    except requests.RequestException as fb_exc:
                        st.toast(f"Lỗi gửi đánh giá: {fb_exc}")

    except requests.RequestException as exc:
        st.error(f"Yêu cầu thất bại: {exc}")
