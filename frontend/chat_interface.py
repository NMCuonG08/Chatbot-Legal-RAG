import time

import requests
import streamlit as st

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="Legal RAG & Agentic", page_icon="⚖️", layout="centered")
st.title("Legal RAG & Agentic Workflow")
st.caption("MVP chat UI with async task polling")

if "session_id" not in st.session_state:
    st.session_state.session_id = "demo-session"

prompt = st.text_area("Nhap cau hoi", placeholder="Vi du: Hay tom tat quy dinh moi ve hop dong lao dong")

if st.button("Gui") and prompt.strip():
    try:
        submit_resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": prompt.strip()},
            timeout=20,
        )
        submit_resp.raise_for_status()
        payload = submit_resp.json()
        task_id = payload["task_id"]
        route = payload["route"]

        st.info(f"Task queued: {task_id} | route: {route}")

        final_data = None
        for _ in range(30):
            task_resp = requests.get(f"{BACKEND_URL}/tasks/{task_id}", timeout=20)
            task_resp.raise_for_status()
            result_data = task_resp.json()
            if result_data["status"] == "SUCCESS":
                final_data = result_data
                break
            time.sleep(1)

        if final_data is None:
            st.warning("Task is still running. Try polling again.")
        else:
            st.success("Done")
            st.write(final_data.get("result", "No result"))

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
