#!/usr/bin/env bash
# Single ops entrypoint (Git Bash / WSL). Run from anywhere; cd to repo root.
# Keeps every MLOps / opsdev / prod action behind ONE command so production
# runbooks and CI can call the same path as local dev.
#
# Usage: scripts/dev.sh <command> [args...]
#
#   Infra:    up | down | ps
#   Build:    setup            (pip install + DB schema + Qdrant collections, once)
#   App:      app              (Celery + FastAPI + Streamlit, Ctrl-C stops all)
#   Test:     test | gate | phase P1..P7 | slow
#   Eval:     golden | eval [--ollama] | drift <baseline.json> <recent.json>
#             redteam | regression <baseline.json>
#   Prod:     prod | ci-check
#
# Numbered steps inside each command show the MLOps pipeline order so an
# ops/opsdev engineer can read the script top-down as a runbook.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="backend/src:."

# Auto-activate local virtual environment if present
if [ -d ".venv" ]; then
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
  fi
fi

# ---- helpers ---------------------------------------------------------------
require_env() {
  # Verify a required env var exists (value not printed, presence only).
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "[ERROR] env $name missing. Set in backend/.env or export before running." >&2
    exit 1
  fi
}

wait_http() {
  # Poll an HTTP health URL until it responds or timeout (max ~30s).
  local url="$1"; local name="$2"
  for _ in $(seq 1 30); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "  $name up ($url)"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] $name not responding at $url" >&2
  exit 1
}

case "${1:-}" in
  # ===== Infra ==============================================================
  up)
    echo "[1/1] Starting infra (Redis, MariaDB, Qdrant, Prometheus, Grafana)..."
    docker compose up -d redis mariadb qdrant prometheus grafana
    docker compose ps
    echo
    echo "Next: scripts/dev.sh setup  (first time)  then  scripts/dev.sh app"
    echo "Stop:  scripts/dev.sh down"
    ;;
  down)
    echo "[1/1] Stopping infra..."
    docker compose down
    ;;
  ps)
    docker compose ps
    ;;

  # ===== Build / install ====================================================
  setup)
    echo "[1/4] Installing backend deps..."
    pip install -r backend/requirements.txt
    echo "[2/4] Installing frontend deps..."
    pip install -r frontend/requirements.txt
    echo "[3/4] Installing eval/dev deps (pytest, deepeval, ragas, scipy, otel)..."
    pip install -r requirements_dev.txt
    echo "[4/4] Creating DB schema + Qdrant collections..."
    ( cd backend/src && \
      python -c "import models; models.ensure_database_schema()" && \
      python -c "from vectorize import create_collection, list_collections; existing = [c['name'] for c in list_collections()]; [create_collection(c) for c in ['llm','user_episodes','semantic_cache'] if c not in existing]" )
    echo
    echo "DONE. Now edit backend/.env:"
    echo "  GROQ_API_KEY / TAVILY_API_KEY / COHERE_API_KEY = real keys"
    echo "  DATABASE_URL=mysql+pymysql://legal:legal@127.0.0.1:3306/legal_db"
    echo "  CUSTOM_EMBEDDING_ENABLED=false  (use Cohere cloud; simplest)"
    echo "  REDIS_URL=redis://localhost:6379/0"
    echo "  TRACE_REDIS_URL=redis://localhost:6379/1"
    echo "Then: scripts/dev.sh app"
    ;;

  # ===== App (dev) ==========================================================
  app)
    echo "Launching Celery worker, FastAPI, Streamlit (Ctrl-C stops all)..."
    ( cd backend/src && celery -A tasks.celery_app worker --loglevel=info -P solo ) &
    CELERY_PID=$!
    ( cd backend/src && uvicorn app:app --host 0.0.0.0 --port 8002 ) &
    API_PID=$!
    ( cd frontend && streamlit run chat_interface.py --server.port 8501 ) &
    UI_PID=$!
    trap "kill $CELERY_PID $API_PID $UI_PID 2>/dev/null" EXIT INT TERM
    echo
    echo "UI:      http://localhost:8501"
    echo "API:     http://localhost:8002/docs"
    echo "Health:  http://localhost:8002/health"
    echo "Grafana:  http://localhost:3000 (Monitoring)"
    wait
    ;;

  # ===== Test (offline gate, no services/keys needed) =======================
  test)
    echo "[1/1] Offline gate (no services/keys)..."
    pytest -q
    ;;
  gate)
    echo "[1/3] pytest offline gate..."
    pytest -q
    echo "[2/3] mypy (eval harness, non-blocking)..."
    mypy backend/src/evaluation --ignore-missing-imports || echo "  mypy: warnings (non-blocking)"
    echo "[3/3] black --check (non-blocking)..."
    black --check backend/src/evaluation || echo "  black: format drift (non-blocking)"
    ;;
  phase)
    # scripts/dev.sh phase P1   -> run that phase's tests
    phase="${2:-}"
    case "$phase" in
      P1) files="tests/test_run_metadata.py tests/test_stats.py tests/test_parallel.py tests/test_regression.py" ;;
      P2) files="tests/test_judge_panel.py tests/test_pairwise_eval.py" ;;
      P3) files="tests/test_sim_user.py tests/test_scenarios.py" ;;
      P4) files="tests/test_redteam_dataset.py tests/test_redteam_metrics.py tests/test_pii_detector.py tests/test_redteam_deepeval.py" ;;
      P5) files="tests/test_slicing.py tests/test_metrics_extended.py tests/test_golden_unified.py" ;;
      P6) files="tests/test_cost_routing.py tests/test_drift.py tests/test_otel_bridge.py tests/test_canary_shadow.py" ;;
      P7) files="tests/test_ci_markers.py tests/test_deepeval_gate.py tests/test_ragas_calibration.py" ;;
      *) echo "Usage: $0 phase P1|P2|P3|P4|P5|P6|P7"; exit 1 ;;
    esac
    echo "[1/1] pytest $phase: $files"
    pytest -q $files
    ;;
  slow)
    echo "[1/1] Slow / live / redteam_live markers (needs ragas/deepeval installed)..."
    pytest -m "slow or live or redteam_live" -q
    ;;

  # ===== Eval (needs services + Groq key) ====================================
  golden)
    echo "[1/1] Build unified golden set -> data/golden_unified.jsonl..."
    python -c "from evaluation.golden_unified import write_unified_dataset; write_unified_dataset()"
    echo "Done. File: data/golden_unified.jsonl"
    ;;
  eval)
    # scripts/dev.sh eval            -> Groq judge
    # scripts/dev.sh eval --ollama   -> local Ollama judge (no Groq quota)
    shift
    JUDGE_PROVIDER="groq"
    JUDGE_MODEL="${JUDGE_MODEL:-llama-3.1-8b-instant}"
    PARALLEL="${PARALLEL:-4}"
    N="${N:-20}"
    if [ "${1:-}" = "--ollama" ]; then
      JUDGE_PROVIDER="ollama"
      JUDGE_MODEL="${JUDGE_MODEL:-qwen2.5:0.5b}"
      PARALLEL="${PARALLEL:-2}"
      N="${N:-10}"
    fi
    require_env GROQ_API_KEY
    echo "[1/2] Eval suite (judge=$JUDGE_PROVIDER/$JUDGE_MODEL, n=$N, parallel=$PARALLEL)..."
    [ -f data/golden_unified.jsonl ] || { echo "  golden missing -> run: scripts/dev.sh golden"; exit 1; }
    JUDGE_PROVIDER="$JUDGE_PROVIDER" JUDGE_MODEL="$JUDGE_MODEL" \
    python backend/src/evaluation/run_eval.py \
      --mode all --data-file data/golden_unified.jsonl \
      --parallel "$PARALLEL" --n "$N" \
      --output eval_reports/live/${JUDGE_PROVIDER}
    echo "[2/2] Reports under eval_reports/live/${JUDGE_PROVIDER}/"
    ;;
  regression)
    baseline="${2:-}"
    if [ -z "$baseline" ]; then echo "Usage: $0 regression <baseline_run.json>"; exit 1; fi
    echo "[1/1] Regression diff vs baseline $baseline..."
    python backend/src/evaluation/run_eval.py --baseline "$baseline"
    ;;
  drift)
    baseline="${2:-}"; recent="${3:-}"
    if [ -z "$baseline" ] || [ -z "$recent" ]; then
      echo "Usage: $0 drift <baseline_run.json> <recent_run.json>"; exit 1
    fi
    echo "[1/1] Drift (PSI+KL) baseline vs recent..."
    python -c "from evaluation.drift import detect_drift; import json; reps=detect_drift('$baseline','$recent'); print(json.dumps([r.__dict__ for r in reps], indent=2, default=str))"
    ;;
  redteam)
    require_env GROQ_API_KEY
    echo "[1/3] Build golden (if missing)..."
    [ -f data/golden_unified.jsonl ] || python -c "from evaluation.golden_unified import write_unified_dataset; write_unified_dataset()"
    echo "[2/3] Generate promptfoo config..."
    python -c "from evaluation.redteam.promptfoo_config import write_promptfoo_config; from evaluation.redteam.dataset import load_redteam_dataset; write_promptfoo_config(load_redteam_dataset(), 'promptfoo_tests.json', agent_endpoint='http://localhost:8002/chat')"
    echo "[3/3] Run promptfoo eval (needs Node)..."
    npx promptfoo eval -c promptfoo_tests.json
    ;;

  # ===== Prod / CI ===========================================================
  prod)
    # Full orchestrated boot: infra -> setup -> seed -> app
    echo "[1/4] Boot infra..."
    docker compose up -d redis mariadb qdrant prometheus grafana
    echo "[2/4] Wait Qdrant..."
    wait_http "http://localhost:6333/healthz" "Qdrant"
    echo "[3/4] Seed legal data into Qdrant..."
    ( cd backend/src && python import_data.py )
    echo "[4/4] Launch app (Ctrl-C stops all)..."
    exec "$0" app
    ;;
  ci-check)
    # Mirror the offline CI gate exactly (.github/workflows/ci.yml)
    echo "[1/3] Install dev deps..."
    pip install -r requirements_dev.txt
    echo "[2/3] pytest offline gate..."
    pytest -q
    echo "[3/3] mypy + black (non-blocking)..."
    mypy backend/src/evaluation --ignore-missing-imports || true
    black --check backend/src/evaluation || true
    echo "ci-check OK"
    ;;

  *)
    cat <<'USAGE'
Usage: scripts/dev.sh <command> [args]

Infra:
  up              start Redis + MariaDB + Qdrant + Prometheus + Grafana (docker)
  down            stop infra
  ps              show docker compose services

Build (once):
  setup           pip install (backend + frontend + dev) + DB schema + Qdrant collections

App (dev):
  app             Celery worker + FastAPI (:8002) + Streamlit (:8501), Ctrl-C stops all

Test (offline, no services/keys):
  test            offline gate (pytest -q)
  gate            pytest + mypy + black --check (CI mirror)
  phase P1..P7    run one phase's test files
  slow            -m "slow or live or redteam_live" (needs ragas/deepeval installed)

Eval (needs services up + GROQ_API_KEY):
  golden          build data/golden_unified.jsonl
  eval            run_eval --mode all with Groq judge  (env: N, PARALLEL, JUDGE_MODEL)
  eval --ollama   same with local Ollama judge (no Groq quota)
  regression B    diff current run vs baseline run JSON
  drift B R       PSI+KL drift between two run JSONs
  redteam         promptfoo red-team (needs Node + app running on :8002)

Prod / MLOps:
  prod            infra -> wait Qdrant -> seed -> app  (full boot runbook)
  ci-check        exact mirror of .github/workflows/ci.yml offline gate

Env flags:
  N, PARALLEL, JUDGE_MODEL, GROQ_API_KEY, OLLAMA_BASE_URL
  P6 opt-in: OTEL_BRIDGE_ENABLED, COST_ROUTING_ENABLED, SHADOW_MODE_ENABLED
USAGE
    exit 1
    ;;
esac