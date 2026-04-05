FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

ENV PYTHONPATH=/app

CMD ["sh", "-c", "celery -A src.tasks.celery_app worker --loglevel=${CELERY_LOGLEVEL:-info} --concurrency=${CELERY_WORKER_CONCURRENCY:-2}"]
