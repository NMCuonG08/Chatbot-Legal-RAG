from celery import Celery
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_FILES = (str(BACKEND_DIR / ".env"), ".env")

class Settings(BaseSettings):
    # Auto-load from backend/.env when running locally.
    model_config = SettingsConfigDict(
        env_file=ENV_FILES, env_file_encoding="utf-8", extra="ignore"
    )

    # Prefer explicit DSN vars when available.
    mariadb_dsn: str | None = None
    database_url: str | None = None

    # Fallback MySQL parts.
    mysql_user: str = "legal"
    mysql_password: str = "legal"
    mysql_root_password: str | None = None
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "legal_db"

    # Redis/Celery
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    redis_url: str = "redis://localhost:6379/0"
    celery_task_always_eager: bool = False

    # LangGraph checkpointing (Phase B). default_ttl is in MINUTES for RedisSaver.
    langgraph_checkpoint_ttl_seconds: int = 86400  # 24h
    # Force in-process MemorySaver (no Redis needed) for local dev / tests.
    use_memory_saver: bool = False

    # Trace pub/sub (Phase C). Separate Redis DB index to avoid broker/checkpointer contention.
    trace_redis_channel: str = "graph_trace_events"
    trace_redis_url: str = "redis://localhost:6379/1"

    def resolve_db_url(self) -> str:
        # Priority: DATABASE_URL(any SQLAlchemy-supported URL) ->
        # MARIADB_DSN -> composed mysql URL from parts.
        if self.database_url:
            return self.database_url

        if self.mariadb_dsn:
            return self.mariadb_dsn

        password = self.mysql_password or self.mysql_root_password or ""
        return (
            f"mysql+pymysql://{self.mysql_user}:{password}@"
            f"{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


settings = Settings()

# Celery settings
CELERY_BROKER_URL = settings.celery_broker_url or settings.redis_url
CELERY_RESULT_BACKEND = settings.celery_result_backend or settings.redis_url

# MySQL database configuration
SQLALCHEMY_DATABASE_URL = settings.resolve_db_url()

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # Audit 6.1 — explicit QueuePool sizing. Defaults (pool_size=5,
    # max_overflow=10, no recycle) overflow under Celery + API concurrency and
    # keep stale MySQL connections past the server's wait_timeout. pool_recycle
    # 1800s < typical MySQL wait_timeout (28800s default, often lowered) so
    # connections are proactively recycled; pool_pre_ping drops dead ones.
    pool_size=20,
    max_overflow=40,
    pool_recycle=1800,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_celery_app(name):
    # Create a Celery app instance
    app = Celery(
        name,
        broker=CELERY_BROKER_URL,  # Redis as the message broker
        backend=CELERY_RESULT_BACKEND,  # Redis as the result backend
    )

    # Optionally, you can configure additional settings here
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Ho_Chi_Minh",  # Set to a city in UTC+7
        enable_utc=True,
        result_backend_transport_options={"redis_client_args": {"protocol": 2}},
        broker_transport_options={"redis_client_args": {"protocol": 2}},
        task_always_eager=settings.celery_task_always_eager,
    )

    # Configure Celery logging
    app.conf.update(
        worker_hijack_root_logger=False,
        worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
        worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    )

    # Audit 8.1 — worker self-recycling. Celery workers load the bge-m3
    # SentenceTransformer (PyTorch) in-process; PyTorch's caching allocator +
    # per-task allocations grow resident memory without bound and never release
    # to the OS. Without recycling a worker eventually OOMs the box. Recycle a
    # worker after 100 tasks OR ~4GB resident (worker_max_memory_per_child is
    # KiB), so a fresh worker replaces it before that happens.
    app.conf.update(
        worker_max_tasks_per_child=100,
        worker_max_memory_per_child=4_000_000,
    )

    return app