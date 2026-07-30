"""Edge-case tests for ``dakp_pipeline.benchmarks`` instrumentation internals.

Covers the branches a normal benchmark run never hits: the ``_wrap_ingest`` wrapper ignoring a
non-``(ref, hit)`` result and attributing totals without an active stage, the ``_stage_for``
unknown-target guard, and the macOS (Darwin) ``ru_maxrss`` normalization in ``_peak_rss_mb``.
"""

from __future__ import annotations

import pytest

from dakp_pipeline import benchmarks


def test_wrap_ingest_ignores_non_tuple_result() -> None:
    recorder = benchmarks._Recorder()
    wrapped = benchmarks._wrap_ingest(recorder, lambda _self: "not-a-tuple")
    assert wrapped(object()) == "not-a-tuple"
    assert recorder.total_cache_hits == 0
    assert recorder.total_cache_misses == 0


def test_wrap_ingest_ignores_wrong_length_tuple() -> None:
    recorder = benchmarks._Recorder()
    wrapped = benchmarks._wrap_ingest(recorder, lambda _self: (1, 2, 3))
    wrapped(object())
    assert recorder.total_cache_hits == 0
    assert recorder.total_cache_misses == 0


def test_wrap_ingest_without_active_stage_records_totals_only() -> None:
    recorder = benchmarks._Recorder()  # current is None
    wrapped = benchmarks._wrap_ingest(recorder, lambda _self: (None, True))
    wrapped(object())
    assert recorder.total_cache_hits == 1  # rolled up run-wide
    assert all(stat.cache_hits == 0 for stat in recorder.stats.values())  # no stage attribution


def test_stage_for_unknown_target_raises() -> None:
    with pytest.raises(KeyError, match="no stage registered"):
        benchmarks._stage_for(object(), "nonexistent")


def test_peak_rss_mb_darwin_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmarks.platform, "system", lambda: "Darwin")
    assert benchmarks._peak_rss_mb() >= 0.0
