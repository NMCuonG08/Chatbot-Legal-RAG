"""Tests for legal_effectivity.classify_effectivity — per-document effectivity
status (in_force / not_yet_effective / repealed / amended).

Deterministic, no network/DB needed. Reference dates are pinned via ``as_of``
so the tests do not drift with wall-clock time.
"""
from datetime import date

from legal_effectivity import (
    EFFECTIVITY_AMENDED,
    EFFECTIVITY_IN_FORCE,
    EFFECTIVITY_NOT_YET,
    EFFECTIVITY_REPEALED,
    classify_effectivity,
    effectivity_for_payload,
)


class TestInForce:
    def test_luat_dat_dai_2024_in_force(self):
        # Effective 01/01/2025, not superseded, not amended.
        assert classify_effectivity("Luật Đất đai 2024", as_of=date(2026, 7, 20)) == EFFECTIVITY_IN_FORCE

    def test_blld_2019_in_force(self):
        # Effective 01/01/2021, not superseded, not amended.
        assert classify_effectivity("Bộ luật Lao động 2019", as_of=date(2026, 7, 20)) == EFFECTIVITY_IN_FORCE

    def test_unknown_statute_defaults_in_force(self):
        # Conservative default for statutes absent from the curated table.
        assert classify_effectivity("Luật Something 2018", as_of=date(2026, 7, 20)) == EFFECTIVITY_IN_FORCE


class TestNotYetEffective:
    def test_future_effective_date(self):
        # blld_2019 effective 01/01/2021 -> not yet effective as of 2020.
        assert classify_effectivity("Bộ luật Lao động 2019", as_of=date(2020, 1, 1)) == EFFECTIVITY_NOT_YET

    def test_unknown_statute_future_year(self):
        # Unknown name + a year past the reference year -> not yet effective.
        assert classify_effectivity("Luật Mới", document_year=2099, as_of=date(2026, 7, 20)) == EFFECTIVITY_NOT_YET


class TestRepealed:
    def test_blds_2005_repealed_by_2015(self):
        # blds_2005 replaced_by blds_2015 (effective 01/01/2017) -> repealed today.
        assert classify_effectivity("Bộ luật Dân sự 2005", as_of=date(2026, 7, 20)) == EFFECTIVITY_REPEALED

    def test_luat_dat_dai_2013_repealed_by_2024(self):
        assert classify_effectivity("Luật Đất đai 2013", as_of=date(2026, 7, 20)) == EFFECTIVITY_REPEALED

    def test_blld_2012_repealed_by_2019(self):
        assert classify_effectivity("Bộ luật Lao động 2012", as_of=date(2026, 7, 20)) == EFFECTIVITY_REPEALED

    def test_repealed_only_when_successor_effective(self):
        # blds_2005 replaced_by blds_2015 (eff 01/01/2017). Before that date the
        # old statute was still in force, not yet repealed.
        assert classify_effectivity("Bộ luật Dân sự 2005", as_of=date(2016, 6, 1)) == EFFECTIVITY_IN_FORCE


class TestAmended:
    def test_blds_2015_amended(self):
        # blds_2015 has amended_by non-empty -> amended (still in force).
        assert classify_effectivity("Bộ luật Dân sự 2015", as_of=date(2026, 7, 20)) == EFFECTIVITY_AMENDED


class TestPayloadWrapper:
    def test_never_raises(self):
        # effectivity_for_payload must always return a valid status.
        assert effectivity_for_payload(None, None) in {
            EFFECTIVITY_IN_FORCE, EFFECTIVITY_NOT_YET, EFFECTIVITY_REPEALED, EFFECTIVITY_AMENDED
        }

    def test_payload_matches_classify(self):
        assert effectivity_for_payload("Luật Đất đai 2024", None, as_of=date(2026, 7, 20)) == EFFECTIVITY_IN_FORCE

    def test_empty_name(self):
        assert effectivity_for_payload("", None) == EFFECTIVITY_IN_FORCE