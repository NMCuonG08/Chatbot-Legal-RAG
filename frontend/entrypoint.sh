#!/usr/bin/env sh
set -e
streamlit run chat_interface.py --server.address 0.0.0.0 --server.port 8501
