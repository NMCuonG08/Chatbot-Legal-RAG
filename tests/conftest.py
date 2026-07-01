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
