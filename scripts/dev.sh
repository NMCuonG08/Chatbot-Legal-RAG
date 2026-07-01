#!/usr/bin/env bash
# Local dev launcher (Git Bash / WSL). Run from anywhere; cd to repo root.
# Usage: scripts/dev.sh up | setup | app
set -e
cd "$(dirname "$0")/.."

# Auto-activate local virtual environment if present
if [ -d ".venv" ]; then
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
  fi
fi

case "${1:-}" in
  up)
    echo "[1/1] Starting infra services (Redis, MariaDB, Qdrant, Prometheus, Grafana)..."
    docker compose up -d redis mariadb qdrant prometheus grafana
    docker compose ps
    echo
    echo "Next: scripts/dev.sh setup  (first time)  then  scripts/dev.sh app"
    echo "To stop infra: docker compose down"
    ;;
  setup)
    echo "[1/3] Installing backend deps..."
    pip install -r backend/requirements.txt
    echo "[2/3] Installing frontend deps..."
    pip install -r frontend/requirements.txt
    echo "[3/3] Creating DB schema + Qdrant collections..."
    ( cd backend/src && \
      python -c "import models; models.ensure_database_schema()" && \
      python -c "from vectorize import create_collection; [create_collection(c) for c in ['llm','user_episodes','semantic_cache']]" )
    echo
    echo "DONE. Now edit backend/.env:"
    echo "  GROQ_API_KEY / TAVILY_API_KEY / COHERE_API_KEY = real keys"
    echo "  DATABASE_URL=mysql+pymysql://legal:legal@127.0.0.1:3306/legal_db"
    echo "  CUSTOM_EMBEDDING_ENABLED=false  (use Cohere cloud; simplest)"
    echo "  REDIS_URL=redis://localhost:6379/0"
    echo "  TRACE_REDIS_URL=redis://localhost:6379/1"
    echo "Then: scripts/dev.sh app"
    ;;
  app)
    echo "Launching Celery worker, FastAPI, Streamlit (Ctrl-C to stop all)..."
    ( cd backend/src && celery -A tasks.celery_app worker --loglevel=info -P solo ) &
    CELERY_PID=$!
    ( cd backend/src && uvicorn app:app --host 0.0.0.0 --port 8002 ) &
    API_PID=$!
    ( cd frontend && streamlit run chat_interface.py --server.port 8501 ) &
    UI_PID=$!
    
    # Catch exit signals and terminate spawned background processes
    trap "kill $CELERY_PID $API_PID $UI_PID 2>/dev/null" EXIT INT TERM
    echo
    echo "UI:      http://localhost:8501"
    echo "API:     http://localhost:8002/docs"
    echo "Health:  http://localhost:8002/health"
    echo "Grafana: http://localhost:3000 (Monitoring)"
    wait
    ;;
  *)
    echo "Usage: $0 [up | setup | app]"
    echo "  up     - start Redis + MariaDB + Qdrant + Prometheus + Grafana via docker compose"
    echo "  setup  - pip install + create DB schema + Qdrant collections (run once)"
    echo "  app    - launch Celery worker + FastAPI + Streamlit"
    exit 1
    ;;
esac