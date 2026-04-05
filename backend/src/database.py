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
    SQLALCHEMY_DATABASE_URL, pool_pre_ping=True  # Improve connection resilience
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
    )

    # Configure Celery logging
    app.conf.update(
        worker_hijack_root_logger=False,
        worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
        worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    )

    return app