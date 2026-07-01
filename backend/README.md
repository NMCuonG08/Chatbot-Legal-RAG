# Backend Service

FastAPI + Celery backend for Legal RAG and agentic routing.

## 🚀 Startup Commands

### Start Celery Worker
From the `backend/src` directory:
```bash
celery -A tasks.celery_app worker --loglevel=info -P solo
```

### Start FastAPI Backend
From the `backend/src` directory:
```bash
uvicorn app:app --host 0.0.0.0 --port 8002
```

## 📥 Ingestion & Data Import
To populate the vector database with Q&A legal pairs, run this command from the `backend` directory:
```bash
python src/import_data.py --data-file ../data_pipeline/data/finetune_data/train_qa_format.jsonl --collection llm
```


