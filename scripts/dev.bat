@echo off
REM Local dev launcher (Windows). Run from anywhere; cd to repo root automatically.
REM Usage: scripts\dev.bat up | setup | app
setlocal
cd /d "%~dp0\.."

if "%~1"=="" goto usage
if "%~1"=="up" goto up
if "%~1"=="setup" goto setup
if "%~1"=="app" goto app
goto usage

:up
echo [1/1] Starting infra services (Redis, MariaDB, Qdrant)...
docker compose up -d redis mariadb qdrant
docker compose ps
echo.
echo Next: scripts\dev.bat setup  (first time)  then  scripts\dev.bat app
goto end

:setup
echo [1/3] Installing backend deps...
pip install -r backend\requirements.txt
echo [2/3] Installing frontend deps...
pip install -r frontend\requirements.txt
echo [3/3] Creating DB schema + Qdrant collections...
pushd backend\src
python -c "import models; models.ensure_database_schema()"
python -c "from vectorize import create_collection; [create_collection(c) for c in ['llm','user_episodes','semantic_cache']]"
popd
echo.
echo DONE. Now edit backend\.env:
echo   GROQ_API_KEY / TAVILY_API_KEY / COHERE_API_KEY  = real keys
echo   DATABASE_URL=mysql+pymysql://legal:legal@127.0.0.1:3306/legal_db
echo   CUSTOM_EMBEDDING_ENABLED=false  (use Cohere cloud; simplest)
echo   REDIS_URL=redis://localhost:6379/0
echo   TRACE_REDIS_URL=redis://localhost:6379/1
echo Then: scripts\dev.bat app
goto end

:app
echo Launching 3 windows: Celery worker, FastAPI, Streamlit...
start "Celery Worker" cmd /k "cd backend\src && celery -A tasks.celery_app worker --loglevel=info -P solo"
start "FastAPI :8002" cmd /k "cd backend\src && uvicorn app:app --host 0.0.0.0 --port 8002"
start "Streamlit :8501" cmd /k "cd frontend && streamlit run chat_interface.py --server.port 8501"
echo.
echo UI:      http://localhost:8501
echo API:     http://localhost:8002/docs
echo Health:  http://localhost:8002/health
goto end

:usage
echo Usage: scripts\dev.bat [up ^| setup ^| app]
echo   up     - start Redis + MariaDB + Qdrant via docker compose
echo   setup  - pip install + create DB schema + Qdrant collections (run once)
echo   app    - launch Celery worker + FastAPI + Streamlit in 3 windows
exit /b 1

:end
endlocal