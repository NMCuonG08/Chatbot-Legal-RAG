import sys
from pathlib import Path

import pytest

# Make backend/src importable so test modules can `import security`, `import
# vectorize`, `from pipeline.orchestrator import run_pipeline`, etc. without a
# manual PYTHONPATH. Resilient to being run from any cwd.
_BACKEND_SRC = Path(__file__).resolve().parent.parent / "backend" / "src"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))


@pytest.fixture
def sample_query() -> str:
    return "sample legal question"


# ---- Phase 2: sqlite in-memory DB fixture ----
# Swaps database.engine + SessionLocal to an in-memory sqlite engine and
# creates all tables on models.Base. Lets audit/approval/user tests run
# without a live MySQL. Scoped per-test (autouse=False): opt in via the
# `sqlite_db` fixture.
@pytest.fixture
def sqlite_db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import database
    import models

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", SessionLocal)

    def _get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(database, "get_db", _get_db)

    models.Base.metadata.create_all(bind=engine)
    yield engine
    models.Base.metadata.drop_all(bind=engine)


@pytest.fixture
def jwt_secret(monkeypatch):
    """Ensure JWT_SECRET is set for auth tests."""
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-for-prod")
    monkeypatch.setenv("JWT_ALG", "HS256")
    monkeypatch.setenv("JWT_EXP_MIN", "60")
    monkeypatch.delenv("ALLOW_UNSAFE_AUTH", raising=False)
    yield "test-secret-not-for-prod"


# Audit 3.2: SANDBOX_ENABLED defaults ON in production so calc tools run in a
# subprocess. The unit suite must not spawn subprocesses (slow + platform-
# fragile), so force it off here. Tests that exercise the sandbox path
# (test_sandboxable) monkeypatch config.SANDBOX_ENABLED=True explicitly, which
# overrides this autouse value (same function-scoped monkeypatch instance).
@pytest.fixture(autouse=True)
def _sandbox_off_in_tests(monkeypatch):
    try:
        import config
        monkeypatch.setattr(config, "SANDBOX_ENABLED", False)
    except Exception:
        pass