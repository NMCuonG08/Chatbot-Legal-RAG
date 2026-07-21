# Worker image — mirrors backend/Dockerfile (same base + deps + PYTHONPATH)
# but runs Celery instead of uvicorn. tasks.py uses bare sibling imports
# (from prompt_loader import, from database import) so PYTHONPATH MUST point
# at /usr/src/app/src, same as backend — not /app.
# Build context: ./backend
#   docker build -f worker.Dockerfile -t legal-worker ./backend
FROM python:3.10.12-slim-bullseye

RUN apt update -y \
    && apt-get install \
        python3-dev \
        default-libmysqlclient-dev \
        build-essential \
        pkg-config -y \
    && apt-get clean

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app/src"

RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

COPY ./entrypoint.sh /usr/src/app/entrypoint.sh
RUN chmod +x /usr/src/app/entrypoint.sh

COPY . /usr/src/app/

# entrypoint.sh execs "$@" when args present -> CMD below becomes the celery cmd.
# -A tasks.celery_app matches scripts/dev.sh (celery app = backend/src/tasks/celery_app).
# -P solo: Windows + low-traffic safe. Override loglevel/concurrency via env
# CELERY_LOGLEVEL / CELERY_WORKER_CONCURRENCY at deploy time (compose sets these).
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
CMD ["celery", "-A", "tasks.celery_app", "worker", "--loglevel=info", "-P", "solo"]