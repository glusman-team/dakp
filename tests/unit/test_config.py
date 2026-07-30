from __future__ import annotations

import pytest

from dakp_pipeline.config import PROFILES, Profile, load_profile


def test_load_mock_profile_has_expected_defaults() -> None:
    profile = load_profile("mock")
    assert isinstance(profile, Profile)
    assert profile.name == "mock"
    assert profile.mock_sources is True
    assert profile.threads == 1
    assert profile.quarter_limit == 1
    # Mock must never attempt the real Tablassert handoff.
    assert profile.run_tablassert is False


def test_load_full_profile_is_conservatively_bounded() -> None:
    profile = load_profile("prod")
    assert profile.mock_sources is False
    assert profile.run_tablassert is True
    # PLAN.md calls for bounded (not unbounded 80-thread) parallelism.
    assert 1 <= profile.threads <= 80
    assert profile.memory_budget_gb >= 1


def test_unknown_profile_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        load_profile("does-not-exist")


def test_profile_overrides_apply() -> None:
    profile = load_profile("mock", threads=4, quarter_limit=7)
    assert profile.threads == 4
    assert profile.quarter_limit == 7
    # Untouched fields keep their base values.
    assert profile.mock_sources is True


def test_all_documented_profiles_exist() -> None:
    assert frozenset({"mock", "sample", "prod"}) == PROFILES
    # No absolute paths leak into a profile.
    for name in PROFILES:
        assert isinstance(load_profile(name), Profile)
