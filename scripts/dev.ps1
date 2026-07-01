# Windows PowerShell dev launcher
# Usage: .\scripts\dev.ps1 up | setup | app

param (
    [string]$Action
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ScriptDir -eq $null -or $ScriptDir -eq "") {
    $ScriptDir = "."
}
Set-Location (Join-Path $ScriptDir "..")

# Auto-activate local virtual environment if present
if (Test-Path ".venv") {
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        .venv\Scripts\Activate.ps1
    }
}

switch ($Action) {
    "up" {
        Write-Host "[1/1] Starting infra services (Redis, MariaDB, Qdrant, Prometheus, Grafana)..." -ForegroundColor Green
        docker compose up -d redis mariadb qdrant prometheus grafana
        docker compose ps
        Write-Host "`nNext: .\scripts\dev.ps1 setup (first time) then .\scripts\dev.ps1 app"
        Write-Host "To stop infra: docker compose down"
    }
    "setup" {
        Write-Host "[1/3] Installing backend deps..." -ForegroundColor Green
        pip install -r backend/requirements.txt
        Write-Host "[2/3] Installing frontend deps..." -ForegroundColor Green
        pip install -r frontend/requirements.txt
        Write-Host "[3/3] Creating DB schema + Qdrant collections..." -ForegroundColor Green
        Push-Location backend/src
        python -c "import models; models.ensure_database_schema()"
        python -c "from vectorize import create_collection; [create_collection(c) for c in ['llm','user_episodes','semantic_cache']]"
        Pop-Location
        Write-Host "`nDONE. Now edit backend/.env:"
        Write-Host "  GROQ_API_KEY / TAVILY_API_KEY / COHERE_API_KEY = real keys"
        Write-Host "  DATABASE_URL=mysql+pymysql://legal:legal@127.0.0.1:3306/legal_db"
        Write-Host "  REDIS_URL=redis://localhost:6379/0"
        Write-Host "  TRACE_REDIS_URL=redis://localhost:6379/1"
        Write-Host "Then: .\scripts\dev.ps1 app"
    }
    "app" {
        Write-Host "Launching Celery worker, FastAPI, Streamlit..." -ForegroundColor Green
        Write-Host "UI:      http://localhost:8501" -ForegroundColor Yellow
        Write-Host "API:     http://localhost:8002/docs" -ForegroundColor Yellow
        Write-Host "Health:  http://localhost:8002/health" -ForegroundColor Yellow
        Write-Host "Grafana: http://localhost:3000 (Monitoring)" -ForegroundColor Yellow
        Write-Host "`nSpawning separate windows for logs... Close the terminal windows to shut down individual services." -ForegroundColor Cyan

        # Start Celery, FastAPI, and Streamlit in new PowerShell windows with automatic .venv activation
        Start-Process powershell -ArgumentList '-NoExit', '-Command', '$Host.UI.RawUI.WindowTitle="Celery Worker"; Set-Location backend/src; if (Test-Path "../../.venv") { ..\..\.venv\Scripts\Activate.ps1 }; celery -A tasks.celery_app worker --loglevel=info -P solo'
        Start-Process powershell -ArgumentList '-NoExit', '-Command', '$Host.UI.RawUI.WindowTitle="FastAPI Backend"; Set-Location backend/src; if (Test-Path "../../.venv") { ..\..\.venv\Scripts\Activate.ps1 }; uvicorn app:app --host 0.0.0.0 --port 8002'
        Start-Process powershell -ArgumentList '-NoExit', '-Command', '$Host.UI.RawUI.WindowTitle="Streamlit UI"; Set-Location frontend; if (Test-Path "../.venv") { ..\.venv\Scripts\Activate.ps1 }; streamlit run chat_interface.py --server.port 8501'
    }
    default {
        Write-Host "Usage: .\scripts\dev.ps1 [up | setup | app]"
        Write-Host "  up     - start Redis + MariaDB + Qdrant + Prometheus + Grafana via docker compose"
        Write-Host "  setup  - pip install + create DB schema + Qdrant collections"
        Write-Host "  app    - launch Celery worker + FastAPI + Streamlit"
    }
}
