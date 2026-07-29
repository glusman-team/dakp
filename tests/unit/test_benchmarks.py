"""Unit tests for the performance benchmark harness (Milestone 8).

Runs the benchmark on the tiny mock fixtures (fast, no network, no Tablassert) and asserts
the report JSON has the deterministic shape the harness promises: per-stage timings + call
counts + cache outcomes in canonical stage order, run-wide totals (wall-time, peak memory,
rows/throughput, cache hit/miss), and an environment block. Also covers the ``wenceslaus_full``
opt-in guard, argument validation, warm-cache behavior across iterations, and that the
non-invasive stage instrumentation is fully restored after the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline.benchmarks import STAGES, BenchmarkReport, run_benchmark
from dakp_pipeline.io.artifact_store import ArtifactStore

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
ASSERTION_TABLES = ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions")


def _bench(workdir: Path, *, iterations: int = 1) -> BenchmarkReport:
    return run_benchmark(profile="mock", fixture_root=FIXTURE_ROOT, workdir=workdir, iterations=iterations)


# --- report structure -------------------------------------------------------------


def test_run_benchmark_writes_report_json(tmp_path: Path) -> None:
    report = _bench(tmp_path / "work")

    assert report.profile == "mock"
    assert report.path == tmp_path / "work" / "data" / "reports" / "benchmark_mock.json"
    assert report.path.exists()

    on_disk = report.report
    assert on_disk["schema_version"] == "dakp.benchmark.v1"
    assert on_disk["profile"] == "mock"
    assert on_disk["iterations"] == 1
    assert isinstance(on_disk["generated_at"], str)
    assert on_disk["generated_at"]

    # Environment block carries the host/profile provenance.
    env = on_disk["environment"]
    assert env["cpu_count"] >= 1
    assert isinstance(env["platform"], str)
    assert env["platform"]
    assert isinstance(env["python_version"], str)
    assert env["python_version"]
    assert env["dakp_version"]
    assert env["memory_unit"] == "MB"


def test_report_has_per_stage_timings_in_canonical_order(tmp_path: Path) -> None:
    on_disk = _bench(tmp_path / "work").report

    stages = on_disk["stages"]
    assert [stage["name"] for stage in stages] == list(STAGES)
    # Every stage entry has the full, fixed field set.
    for stage in stages:
        assert set(stage) == {"name", "calls", "wall_seconds", "cache_hits", "cache_misses"}
        assert isinstance(stage["calls"], int)
        assert stage["wall_seconds"] >= 0.0
        assert stage["cache_hits"] >= 0
        assert stage["cache_misses"] >= 0

    # A full mock run exercises every stage at least once.
    by_name = {stage["name"]: stage for stage in stages}
    for name in STAGES:
        assert by_name[name]["calls"] > 0


def test_report_totals_and_tables(tmp_path: Path) -> None:
    on_disk = _bench(tmp_path / "work").report

    totals = on_disk["totals"]
    assert set(totals) == {
        "wall_seconds",
        "stages_wall_seconds",
        "overhead_wall_seconds",
        "peak_memory_mb",
        "peak_memory_delta_mb",
        "total_rows",
        "throughput_rows_per_sec",
        "cache_hits",
        "cache_misses",
    }
    assert totals["wall_seconds"] > 0.0
    # Instrumented stages are a subset of the whole run, so overhead is non-negative.
    assert totals["overhead_wall_seconds"] >= 0.0
    assert totals["wall_seconds"] >= totals["stages_wall_seconds"]
    assert totals["peak_memory_mb"] > 0.0
    assert totals["peak_memory_delta_mb"] >= 0.0

    # Rows + throughput reflect the three assertion tables.
    assert totals["total_rows"] > 0
    assert totals["throughput_rows_per_sec"] > 0.0
    tables = {entry["name"]: entry["rows"] for entry in on_disk["tables"]}
    assert set(tables) == set(ASSERTION_TABLES)
    assert all(rows > 0 for rows in tables.values())
    assert sum(tables.values()) == totals["total_rows"]


# --- determinism of report shape --------------------------------------------------


def test_report_shape_is_deterministic_across_runs(tmp_path: Path) -> None:
    first = _bench(tmp_path / "work_a").report
    second = _bench(tmp_path / "work_b").report

    # Same top-level keys, environment keys, totals keys, and stage names/order every run.
    assert list(first) == list(second)
    assert list(first["environment"]) == list(second["environment"])
    assert list(first["totals"]) == list(second["totals"])
    assert [s["name"] for s in first["stages"]] == [s["name"] for s in second["stages"]]
    assert [list(s) for s in first["stages"]] == [list(s) for s in second["stages"]]
    assert [t["name"] for t in first["tables"]] == [t["name"] for t in second["tables"]]


# --- cache behavior across iterations ---------------------------------------------


def test_warm_rerun_produces_cache_hits(tmp_path: Path) -> None:
    # Two runs over the SAME workdir: the first pass ingests fixtures (misses), the second
    # finds them already content-addressed (hits).
    on_disk = _bench(tmp_path / "work", iterations=2).report

    assert on_disk["iterations"] == 2
    assert on_disk["totals"]["cache_hits"] > 0
    assert on_disk["totals"]["cache_misses"] > 0
    # Fixture ingest happens in the acquire stage, so the hits land there.
    acquire = next(stage for stage in on_disk["stages"] if stage["name"] == "acquire")
    assert acquire["cache_hits"] > 0


# --- guards and validation --------------------------------------------------------


def test_wenceslaus_full_requires_opt_in(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="allow_full"):
        run_benchmark(profile="wenceslaus_full", workdir=tmp_path / "work")


def test_unknown_profile_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="nonsense"):
        run_benchmark(profile="nonsense", workdir=tmp_path / "work")


def test_iterations_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iterations"):
        run_benchmark(profile="mock", fixture_root=FIXTURE_ROOT, workdir=tmp_path / "work", iterations=0)


# --- instrumentation is non-invasive (fully restored) -----------------------------


def test_instrumentation_restores_originals(tmp_path: Path) -> None:
    _bench(tmp_path / "work")
    # The wrappers are named "wrapper"; after the run the original methods are back in place.
    assert ArtifactStore.ingest.__name__ == "ingest"
