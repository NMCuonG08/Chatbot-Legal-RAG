import json
import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8002")

st.set_page_config(page_title="Legal RAG & Agentic", page_icon="⚖️", layout="centered")
st.title("Legal RAG & Agentic Workflow")
st.caption("MVP chat UI with async task polling")

if "session_id" not in st.session_state:
    st.session_state.session_id = "demo-session"

# Sidebar settings & history
st.sidebar.title("⚙️ Cấu hình & Lịch sử")

# User ID input
st.session_state.session_id = st.sidebar.text_input(
    "Tên người dùng / Session ID",
    value=st.session_state.session_id
)

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

prompt = st.text_area("Nhap cau hoi", placeholder="Vi du: Hay tom tat quy dinh moi ve hop dong lao dong")

if st.button("Gui") and prompt.strip():
    try:
        submit_resp = requests.post(
            f"{BACKEND_URL}/chat/complete",
            json={"user_id": st.session_state.session_id, "user_message": prompt.strip()},
            timeout=20,
        )
        submit_resp.raise_for_status()
        payload = submit_resp.json()
        task_id = payload["task_id"]

        st.info(f"Task queued: {task_id}")

        final_data = None
        for _ in range(30):
            task_resp = requests.get(f"{BACKEND_URL}/chat/complete/{task_id}", timeout=20)
            task_resp.raise_for_status()
            result_data = task_resp.json()
            if result_data.get("task_status") == "SUCCESS":
                final_data = result_data
                break
            elif result_data.get("task_status") == "FAILURE":
                st.error("Task failed execution.")
                break
            time.sleep(1)

        if final_data is None:
            st.warning("Task is still running. Try polling again.")
        else:
            st.success("Done")
            task_result = final_data.get("task_result", {})
            st.write(task_result.get("content", "No result"))
            
            # Display sources if available
            sources = task_result.get("sources", [])
            if sources:
                with st.expander("Nguồn tài liệu tham khảo"):
                    for idx, src in enumerate(sources):
                        content_text = src.get('content') or src.get('text') or ''
                        st.markdown(f"**Tài liệu {idx+1}:** {content_text[:300]}...")

            # Agent trace (Phase F): live-stream trace events from /chat/stream/{task_id}.
            # Best-effort + graceful: if the endpoint is unavailable (404 / older
            # backend), the expander simply reports no trace and never breaks chat.
            with st.expander("Agent trace", expanded=False):
                try:
                    stream_resp = requests.get(
                        f"{BACKEND_URL}/chat/stream/{task_id}",
                        stream=True,
                        timeout=15,
                    )
                    if stream_resp.status_code != 200:
                        st.caption(f"Trace không khả dụng (HTTP {stream_resp.status_code}).")
                    else:
                        trace_lines = []
                        for raw in stream_resp.iter_lines(decode_unicode=True):
                            if not raw or not raw.startswith("data:"):
                                continue
                            data_str = raw[len("data:"):].strip()
                            try:
                                evt = json.loads(data_str)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            node = evt.get("node", "?")
                            etype = evt.get("event_type", "step")
                            payload = evt.get("payload", {})
                            trace_lines.append(f"`{node}` · {etype} · {json.dumps(payload, ensure_ascii=False)[:160]}")
                            if etype == "run_end":
                                break
                        if trace_lines:
                            for line in trace_lines:
                                st.markdown(line)
                        else:
                            st.caption("Không có sự kiện trace.")
                except requests.RequestException as trace_exc:
                    st.caption(f"Trace không khả dụng: {trace_exc}")

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
