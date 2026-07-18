"""Unit tests for redteam.dataset: load, filter, generate (deterministic)."""
from evaluation.redteam.dataset import (
    CATEGORIES,
    RedTeamProbe,
    generate_probes,
    load_redteam_dataset,
)


def test_categories_complete():
    assert "jailbreak_legal" in CATEGORIES
    assert "pii_leak" in CATEGORIES
    assert len(CATEGORIES) == 6


def test_load_redteam_dataset_returns_probes():
    probes = load_redteam_dataset()
    assert len(probes) > 0
    assert all(isinstance(p, RedTeamProbe) for p in probes)
    assert all(p.category and p.prompt for p in probes)


def test_load_redteam_dataset_category_filter():
    oos = load_redteam_dataset(category="oos")
    assert len(oos) > 0
    assert all(p.category == "oos" for p in oos)


def test_load_redteam_dataset_unknown_category_empty():
    assert load_redteam_dataset(category="nonexistent") == []


def test_generate_probes_deterministic():
    a = generate_probes(seed=42)
    b = generate_probes(seed=42)
    assert [p.probe_id for p in a] == [p.probe_id for p in b]


def test_generate_probes_all_categories():
    probes = generate_probes()
    cats = {p.category for p in probes}
    assert cats == set(CATEGORIES)


def test_generate_probes_jailbreak_expected_block():
    probes = generate_probes(category="jailbreak_legal")
    assert all(p.expected_block for p in probes)


def test_generate_probes_citation_injection_verdict():
    probes = generate_probes(category="citation_injection")
    assert all(p.expected_verdict == "unsupported" for p in probes)


def test_probe_is_frozen():
    p = RedTeamProbe(probe_id="x", category="oos", prompt="hi")
    try:
        p.prompt = "mutate"
    except Exception:
        return
    assert False, "RedTeamProbe should be frozen"