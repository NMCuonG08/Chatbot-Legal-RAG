#!/usr/bin/env python
import os
import sys
import subprocess
import time
import urllib.request
import urllib.error

# Determine repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

def load_env_file():
    env_path = os.path.join(REPO_ROOT, "backend", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val

load_env_file()

# Locate and configure virtualenv
venv_dir = os.path.join(REPO_ROOT, ".venv")
scripts_dir = ""
if os.path.isdir(venv_dir):
    scripts_dir = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin")
    if os.path.isdir(scripts_dir):
        # Prepend venv paths to PATH
        os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{os.environ.get('PATH', '')}"

# Configure PYTHONPATH
backend_src = os.path.join(REPO_ROOT, "backend", "src")
os.environ["PYTHONPATH"] = f"{backend_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

def get_bin(name):
    """Find command inside virtualenv or fallback to system search path."""
    if scripts_dir:
        exe = name + (".exe" if os.name == "nt" else "")
        path = os.path.join(scripts_dir, exe)
        if os.path.exists(path):
            return path
    return name

def run_cmd(args, cwd=None, env=None, check=True):
    """Run a command synchronously."""
    # Resolve the executable if it's the first argument
    if args and isinstance(args, list):
        args[0] = get_bin(args[0])
    
    # Merge environments
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
        
    return subprocess.run(args, cwd=cwd, env=cmd_env, check=check)

def require_env(name):
    if not os.environ.get(name):
        print(f"[ERROR] env {name} missing. Set in backend/.env or export before running.", file=sys.stderr)
        sys.exit(1)

def wait_http(url, name, timeout=30):
    print(f"Waiting for {name} to be up at {url}...")
    for _ in range(timeout):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print(f"  {name} up ({url})")
                    return True
        except Exception:
            pass
        time.sleep(1)
    print(f"[ERROR] {name} not responding at {url}", file=sys.stderr)
    sys.exit(1)
def kill_process_on_port(port):
    """Find and kill process listening on a given port (Windows and Unix/macOS compatible)."""
    import os
    try:
        if os.name == 'nt':
            # Windows: use netstat and taskkill
            cmd = 'netstat -ano'
            output = subprocess.check_output(cmd, shell=True).decode()
            pids = set()
            for line in output.strip().split('\n'):
                if 'LISTENING' in line and f':{port}' in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        pids.add(parts[-1])
            for pid in pids:
                print(f"Port {port} is occupied by PID {pid}. Terminating process...")
                subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
        else:
            # Unix/macOS: use lsof and kill
            cmd = f'lsof -t -i:{port}'
            try:
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                if output:
                    for pid in output.split('\n'):
                        print(f"Port {port} is occupied by PID {pid}. Terminating process...")
                        subprocess.run(f'kill -9 {pid}', shell=True, capture_output=True)
            except subprocess.CalledProcessError:
                pass
    except Exception:
        pass

def handle_app():
    # Clean up any processes occupying our app ports to avoid bind errors
    kill_process_on_port(8002)
    kill_process_on_port(8501)
    
    print("Launching Celery worker, FastAPI, Streamlit (Ctrl-C stops all)...")
    
    celery_cmd = [get_bin("celery"), "-A", "tasks.celery_app", "worker", "--loglevel=info", "-P", "solo"]
    uvicorn_cmd = [get_bin("uvicorn"), "app:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
    frontend_cmd = ["cmd.exe", "/c", "npm", "run", "dev"] if os.name == "nt" else ["npm", "run", "dev"]
    
    processes = []
    try:
        # Start Celery
        p_celery = subprocess.Popen(
            celery_cmd, 
            cwd=os.path.join("backend", "src"),
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        processes.append(p_celery)
        
        # Start FastAPI
        p_api = subprocess.Popen(
            uvicorn_cmd, 
            cwd=os.path.join("backend", "src"),
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        processes.append(p_api)
        
        # Start React/Vite Frontend
        p_ui = subprocess.Popen(
            frontend_cmd, 
            cwd="frontend",
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        processes.append(p_ui)
        
        print("\nUI (React/Vite): http://localhost:8501")
        print("API:             http://localhost:8002/docs")
        print("Health:          http://localhost:8002/health")
        print("Grafana:         http://localhost:3000 (Monitoring)\n")
        
        # Wait for all processes to complete/interrupt
        while True:
            # Check if any process has stopped
            for p in processes:
                if p.poll() is not None:
                    # One of the processes exited, terminate others
                    raise RuntimeError("One of the services terminated unexpectedly.")
            time.sleep(1)
            
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nCtrl+C detected. Stopping all processes...")
        else:
            print(f"\nError: {e}. Stopping all processes...")
            
        for p in processes:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        # Wait a moment for graceful shutdown
        time.sleep(2)
        for p in processes:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        sys.exit(1 if not isinstance(e, KeyboardInterrupt) else 0)

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
        
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    if cmd == "up":
        print("[1/1] Starting infra (Redis, MariaDB, Qdrant, Prometheus, Grafana)...")
        run_cmd(["docker", "compose", "up", "-d", "redis", "mariadb", "qdrant", "prometheus", "grafana"])
        run_cmd(["docker", "compose", "ps"])
        print("\nNext: python scripts/dev.py setup  (first time)  then  python scripts/dev.py app")
        print("Stop:  python scripts/dev.py down")
        
    elif cmd == "down":
        print("[1/1] Stopping infra...")
        run_cmd(["docker", "compose", "down"])
        
    elif cmd == "ps":
        run_cmd(["docker", "compose", "ps"])
        
    elif cmd == "setup":
        print("[1/4] Installing backend deps...")
        run_cmd(["pip", "install", "-r", "backend/requirements.txt"])
        print("[2/4] Installing frontend deps...")
        run_cmd(["pip", "install", "-r", "frontend/requirements.txt"])
        print("[3/4] Installing eval/dev deps (pytest, deepeval, ragas, scipy, otel)...")
        run_cmd(["pip", "install", "-r", "requirements_dev.txt"])
        print("[4/4] Creating DB schema + Qdrant collections...")
        
        # Runs DB schema and collections setup
        setup_db_cmd = [
            "python", "-c", 
            "import models; models.ensure_database_schema()"
        ]
        run_cmd(setup_db_cmd, cwd=os.path.join("backend", "src"))
        
        setup_qdrant_cmd = [
            "python", "-c",
            "from vectorize import create_collection, list_collections; "
            "existing = [c['name'] for c in list_collections()]; "
            "[create_collection(c) for c in ['llm','user_episodes','semantic_cache'] if c not in existing]"
        ]
        run_cmd(setup_qdrant_cmd, cwd=os.path.join("backend", "src"))
        
        print("\nDONE. Now edit backend/.env:")
        print("  GROQ_API_KEY / TAVILY_API_KEY / COHERE_API_KEY = real keys")
        print("  DATABASE_URL=mysql+pymysql://legal:legal@127.0.0.1:3306/legal_db")
        print("  CUSTOM_EMBEDDING_ENABLED=false  (use Cohere cloud; simplest)")
        print("  REDIS_URL=redis://localhost:6379/0")
        print("  TRACE_REDIS_URL=redis://localhost:6379/1")
        print("Then: python scripts/dev.py app")
        
    elif cmd == "app":
        handle_app()
        
    elif cmd == "test":
        print("[1/1] Offline gate (no services/keys)...")
        run_cmd(["pytest", "-q"])
        
    elif cmd == "gate":
        print("[1/3] pytest offline gate...")
        run_cmd(["pytest", "-q"])
        print("[2/3] mypy (eval harness, non-blocking)...")
        try:
            run_cmd(["mypy", "backend/src/evaluation", "--ignore-missing-imports"])
        except Exception:
            print("  mypy: warnings (non-blocking)")
        print("[3/3] black --check (non-blocking)...")
        try:
            run_cmd(["black", "--check", "backend/src/evaluation"])
        except Exception:
            print("  black: format drift (non-blocking)")
            
    elif cmd == "phase":
        if not args:
            print("Usage: python scripts/dev.py phase P1|P2|P3|P4|P5|P6|P7")
            sys.exit(1)
        phase = args[0]
        phase_map = {
            "P1": ["tests/test_run_metadata.py", "tests/test_stats.py", "tests/test_parallel.py", "tests/test_regression.py"],
            "P2": ["tests/test_judge_panel.py", "tests/test_pairwise_eval.py"],
            "P3": ["tests/test_sim_user.py", "tests/test_scenarios.py"],
            "P4": ["tests/test_redteam_dataset.py", "tests/test_redteam_metrics.py", "tests/test_pii_detector.py", "tests/test_redteam_deepeval.py"],
            "P5": ["tests/test_slicing.py", "tests/test_metrics_extended.py", "tests/test_golden_unified.py"],
            "P6": ["tests/test_cost_routing.py", "tests/test_drift.py", "tests/test_otel_bridge.py", "tests/test_canary_shadow.py"],
            "P7": ["tests/test_ci_markers.py", "tests/test_deepeval_gate.py", "tests/test_ragas_calibration.py"]
        }
        if phase not in phase_map:
            print(f"Unknown phase {phase}. Options: P1-P7")
            sys.exit(1)
        print(f"[1/1] pytest {phase}: {' '.join(phase_map[phase])}")
        run_cmd(["pytest", "-q"] + phase_map[phase])
        
    elif cmd == "slow":
        print("[1/1] Slow / live / redteam_live markers (needs ragas/deepeval installed)...")
        run_cmd(["pytest", "-m", "slow or live or redteam_live", "-q"])
        
    elif cmd == "golden":
        print("[1/1] Build unified golden set -> data/golden_unified.jsonl...")
        run_cmd(["python", "-c", "from evaluation.golden_unified import write_unified_dataset; write_unified_dataset()"])
        print("Done. File: data/golden_unified.jsonl")
        
    elif cmd == "eval":
        # Check command options
        use_ollama = len(args) > 0 and args[0] == "--ollama"
        
        judge_provider = "groq"
        judge_model = os.environ.get("JUDGE_MODEL", "llama-3.1-8b-instant")
        parallel = os.environ.get("PARALLEL", "4")
        n = os.environ.get("N", "20")
        
        if use_ollama:
            judge_provider = "ollama"
            judge_model = os.environ.get("JUDGE_MODEL", "qwen2.5:0.5b")
            parallel = os.environ.get("PARALLEL", "2")
            n = os.environ.get("N", "10")
            
        require_env("GROQ_API_KEY")
        print(f"[1/2] Eval suite (judge={judge_provider}/{judge_model}, n={n}, parallel={parallel})...")
        
        golden_file = os.path.join("data", "golden_unified.jsonl")
        if not os.path.exists(golden_file):
            print("  golden missing -> run: python scripts/dev.py golden")
            sys.exit(1)
            
        eval_env = {
            "JUDGE_PROVIDER": judge_provider,
            "JUDGE_MODEL": judge_model
        }
        
        eval_args = [
            "python", "backend/src/evaluation/run_eval.py",
            "--mode", "all",
            "--data-file", golden_file,
            "--parallel", str(parallel),
            "--n", str(n),
            "--output", f"eval_reports/live/{judge_provider}"
        ]
        run_cmd(eval_args, env=eval_env)
        print(f"[2/2] Reports under eval_reports/live/{judge_provider}/")
        
    elif cmd == "regression":
        if not args:
            print("Usage: python scripts/dev.py regression <baseline_run.json>")
            sys.exit(1)
        baseline = args[0]
        print(f"[1/1] Regression diff vs baseline {baseline}...")
        run_cmd(["python", "backend/src/evaluation/run_eval.py", "--baseline", baseline])
        
    elif cmd == "drift":
        if len(args) < 2:
            print("Usage: python scripts/dev.py drift <baseline_run.json> <recent_run.json>")
            sys.exit(1)
        baseline = args[0]
        recent = args[1]
        print("[1/1] Drift (PSI+KL) baseline vs recent...")
        drift_cmd = [
            "python", "-c",
            f"from evaluation.drift import detect_drift; import json; "
            f"reps=detect_drift('{baseline}','{recent}'); "
            f"print(json.dumps([r.__dict__ for r in reps], indent=2, default=str))"
        ]
        run_cmd(drift_cmd)
        
    elif cmd == "redteam":
        require_env("GROQ_API_KEY")
        print("[1/3] Build golden (if missing)...")
        golden_file = os.path.join("data", "golden_unified.jsonl")
        if not os.path.exists(golden_file):
            run_cmd(["python", "-c", "from evaluation.golden_unified import write_unified_dataset; write_unified_dataset()"])
        
        print("[2/3] Generate promptfoo config...")
        pf_config_cmd = [
            "python", "-c",
            "from evaluation.redteam.promptfoo_config import write_promptfoo_config; "
            "from evaluation.redteam.dataset import load_redteam_dataset; "
            "write_promptfoo_config(load_redteam_dataset(), 'promptfoo_tests.json', agent_endpoint='http://localhost:8002/chat')"
        ]
        run_cmd(pf_config_cmd)
        
        print("[3/3] Run promptfoo eval (needs Node)...")
        run_cmd(["npx", "promptfoo", "eval", "-c", "promptfoo_tests.json"])
        
    elif cmd == "prod":
        print("[1/4] Boot infra...")
        run_cmd(["docker", "compose", "up", "-d", "redis", "mariadb", "qdrant", "prometheus", "grafana"])
        print("[2/4] Wait Qdrant...")
        wait_http("http://127.0.0.1:6333/healthz", "Qdrant")
        print("[3/4] Seed legal data into Qdrant...")
        run_cmd(["python", "import_data.py"], cwd=os.path.join("backend", "src"))
        print("[4/4] Launch app (Ctrl-C stops all)...")
        handle_app()
        
    elif cmd == "ci-check":
        print("[1/3] Install dev deps...")
        run_cmd(["pip", "install", "-r", "requirements_dev.txt"])
        print("[2/3] pytest offline gate...")
        run_cmd(["pytest", "-q"])
        print("[3/3] mypy + black (non-blocking)...")
        try:
            run_cmd(["mypy", "backend/src/evaluation", "--ignore-missing-imports"])
        except Exception:
            pass
        try:
            run_cmd(["black", "--check", "backend/src/evaluation"])
        except Exception:
            pass
        print("ci-check OK")
        
    else:
        print(f"Unknown command: {cmd}")
        print_usage()
        sys.exit(1)

def print_usage():
    print("""Usage: python scripts/dev.py <command> [args]

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
""")

if __name__ == "__main__":
    main()
