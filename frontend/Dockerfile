FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY chat_interface.py /app/chat_interface.py

CMD ["streamlit", "run", "chat_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
