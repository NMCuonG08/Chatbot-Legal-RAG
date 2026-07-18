"""Meta-test: the offline gate (pytest.ini addopts) excludes live markers and
registers all custom markers so CI never silently runs live-infra tests."""
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTEST_INI = _REPO_ROOT / "pytest.ini"

_EXPECTED_MARKERS = {
    "unit", "integration", "smoke", "slow",
    "redteam", "redteam_live", "live",
}


def _read_ini() -> str:
    return _PYTEST_INI.read_text(encoding="utf-8")


def test_offline_gate_excludes_live_markers():
    """addopts must filter out integration/slow/redteam_live/live so the offline
    gate (every push) never needs live infra or a Groq key."""
    text = _read_ini()
    assert 'not integration' in text
    assert 'not slow' in text
    assert 'not redteam_live' in text
    assert 'not live' in text


def test_custom_markers_registered(pytestconfig):
    """All expected markers are registered (no PytestUnknownMarkWarning in CI)."""
    raw = pytestconfig.getini("markers") or []
    # ini markers come back as "name: description" strings.
    registered = {str(item).split(":", 1)[0].strip() for item in raw}
    missing = _EXPECTED_MARKERS - registered
    assert not missing, f"unregistered markers: {missing}"


def test_live_tests_deselected_by_default(pytestconfig):
    """The default run's -m filter (addopts) deselects live-markered tests.

    ``option.markexpr`` is the resolved marker expression after addopts are
    applied, so it reflects the pytest.ini gate (not just CLI args).
    """
    markexpr = (pytestconfig.option.markexpr or "").lower()
    assert "not live" in markexpr
    assert "not redteam_live" in markexpr
    assert "not slow" in markexpr


@pytest.mark.live
def test_live_marker_is_recognized():
    """Sanity: a live-marked test exists and is deselected in the offline gate."""
    assert True


@pytest.mark.slow
def test_slow_marker_is_recognized():
    """Sanity: a slow-marked test is deselected in the offline gate."""
    assert True