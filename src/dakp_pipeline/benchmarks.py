"""Performance benchmark harness for the three execution profiles (Milestone 8).

Times :func:`dakp_pipeline.pipeline.run_pipeline` for ``mock`` / ``sample`` /
``prod`` and writes a deterministic-shape report to
``<workdir>/data/reports/benchmark_<profile>.json``.

Measurements (pure stdlib + the existing pipeline; no new dependencies):

* **wall-time per stage** — the stage entry points that ``run_pipeline`` resolves through
  their owning modules at call time (fetchers, extractors, transformers, Tablassert config
  generation, the Tablassert handoff, and the translator contract) are *temporarily* wrapped
  so each call is timed and counted. The wrappers are installed for the duration of the run
  and restored afterwards; the stage modules themselves are never modified. Time spent in
  pipeline-internal helpers that are not module-level stage entry points (e.g. the in-runner
  MEDI normalization) is reported as ``overhead_wall_seconds`` = total - instrumented stages.
* **peak memory** — ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` high-water mark, converted
  to MB (KiB on Linux, bytes on macOS).
* **rows / throughput** — assertion-table rows produced per run and rows-per-second.
* **cache hit/miss** — content-addressed :meth:`ArtifactStore.ingest` outcomes, attributed to
  the active stage. A cold run is all misses; re-running over the same workdir (``iterations``
  > 1) turns the fixture ingests into hits, which the tests assert.

``mock`` / ``sample`` actually run; ``prod`` is runnable but guarded behind an
explicit ``allow_full=True`` opt-in so a benchmark invoked on mock fixtures in tests can never
accidentally kick off the full build (it is expected to run on the ``wenceslaus`` host).
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import resource
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dakp_pipeline import __version__
from dakp_pipeline import tablassert as _tablassert_pkg
from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.config import PROFILES
from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.paths import Workdir
from dakp_pipeline.pipeline import PipelineResult, run_pipeline
from dakp_pipeline.sources import dailymed, drugsfda, faers, medi
from dakp_pipeline.tablassert import configs as _tablassert_configs
from dakp_pipeline.translator import contract as _translator_contract

REPORT_SCHEMA = "dakp.benchmark.v1"

#: Canonical stage order; the report always contains exactly these stages in this order
#: (zero-call stages included) so the report shape is deterministic across runs.
STAGES: tuple[str, ...] = ("acquire", "extract", "shape", "configs", "tablassert", "summary")

# Stage name -> module-level callables that ``run_pipeline`` invokes for that stage. Each is
# wrapped (non-invasively) to record wall-time + call count. ``ArtifactStore.ingest`` is
# wrapped separately to attribute content-addressed cache hits/misses to the active stage.
_STAGE_TARGETS: dict[str, tuple[tuple[Any, str], ...]] = {
    "acquire": ((dailymed, "fetch"), (faers, "fetch"), (drugsfda, "fetch"), (medi, "fetch")),
    "extract": ((spl_xml, "extract"), (faers_ascii, "extract"), (drugsfda_products, "extract")),
    "shape": ((approved_treats, "transform"), (observed_uses, "transform"), (contraindications, "transform")),
    "configs": ((_tablassert_configs, "generate"),),
    "tablassert": ((_tablassert_pkg, "run"),),
    "summary": ((_translator_contract, "validate"),),
}


@dataclass
class _StageStat:
    """Accumulator for one stage's timings and cache outcomes."""

    calls: int = 0
    wall_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class _Recorder:
    """Mutable timing/cache state shared by the instrumentation wrappers (single-threaded)."""

    stats: dict[str, _StageStat] = field(default_factory=lambda: {stage: _StageStat() for stage in STAGES})
    current: str | None = None
    total_cache_hits: int = 0
    total_cache_misses: int = 0


def _wrap_stage(recorder: _Recorder, stage: str, original: Any) -> Any:
    """Wrap a stage entry point so each call is timed and counted under ``stage``."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        previous = recorder.current
        recorder.current = stage
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stat = recorder.stats[stage]
            stat.wall_seconds += time.perf_counter() - start
            stat.calls += 1
            recorder.current = previous

    return wrapper


def _wrap_ingest(recorder: _Recorder, original: Any) -> Any:
    """Wrap :meth:`ArtifactStore.ingest` to count content-addressed cache hits/misses.

    ``ingest`` returns ``(ref, cache_hit)``; the outcome is attributed to the active stage
    (``recorder.current``) and always rolled up into the run-wide totals.
    """

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            cache_hit = bool(result[1])
            if cache_hit:
                recorder.total_cache_hits += 1
            else:
                recorder.total_cache_misses += 1
            if recorder.current is not None:
                stat = recorder.stats[recorder.current]
                if cache_hit:
                    stat.cache_hits += 1
                else:
                    stat.cache_misses += 1
        return result

    return wrapper


@contextlib.contextmanager
def _instrument(recorder: _Recorder) -> Iterator[None]:
    """Temporarily install the timing/cache wrappers; always restore the originals."""
    originals: list[tuple[Any, str, Any]] = []
    for targets in _STAGE_TARGETS.values():
        for module, attr in targets:
            original = getattr(module, attr)
            originals.append((module, attr, original))
            setattr(module, attr, _wrap_stage(recorder, _stage_for(module, attr), original))

    ingest_attr = "ingest"
    original_ingest = getattr(ArtifactStore, ingest_attr)
    originals.append((ArtifactStore, ingest_attr, original_ingest))
    setattr(ArtifactStore, ingest_attr, _wrap_ingest(recorder, original_ingest))
    try:
        yield
    finally:
        for module, attr, original in reversed(originals):
            setattr(module, attr, original)


def _stage_for(module: Any, attr: str) -> str:
    """Resolve the stage name for a (module, attr) target (lookup helper for ``_instrument``)."""
    for stage, targets in _STAGE_TARGETS.items():
        for target_module, target_attr in targets:
            if target_module is module and target_attr == attr:
                return stage
    msg = f"no stage registered for {module}.{attr}"  # pragma: no cover - internal invariant
    raise KeyError(msg)


def _peak_rss_mb() -> float:
    """Process peak resident set size (``ru_maxrss`` high-water mark) in MB.

    ``ru_maxrss`` is KiB on Linux and bytes on macOS; normalize to MB. This is a process
    lifetime high-water mark, so callers also record a before/after delta.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _environment() -> dict[str, Any]:
    return {
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "dakp_version": __version__,
        "memory_unit": "MB",
    }


@dataclass(frozen=True)
class BenchmarkReport:
    """Handle to a written benchmark report plus the last pipeline result."""

    profile: str
    path: Path
    report: dict[str, Any]
    result: PipelineResult | None


def run_benchmark(
    profile: str = "mock",
    *,
    fixture_root: Path | str | None = None,
    workdir: Path | str = "data",
    iterations: int = 1,
    allow_full: bool = False,
    params: Mapping[str, Any] | None = None,
) -> BenchmarkReport:
    """Benchmark ``run_pipeline`` for ``profile`` and write the report JSON.

    Args:
        profile: One of ``mock`` / ``sample`` / ``prod``.
        fixture_root: Mock fixture directory (forwarded to ``run_pipeline``; required for mock).
        workdir: Pipeline workdir root; the report lands at ``<workdir>/data/reports/``.
        iterations: Number of full pipeline runs to time (>= 1). Running more than once over
            the same workdir exercises the content-addressed cache (warm pass = cache hits).
        allow_full: Explicit opt-in required to actually run ``prod`` (guard so a
            benchmark invoked on mock fixtures in tests never starts the full build).
        params: Extra pipeline params (forwarded to ``run_pipeline``).

    Returns:
        A :class:`BenchmarkReport` with the report path, the report dict, and the final
        :class:`PipelineResult`.
    """
    if profile not in PROFILES:
        msg = f"Unknown profile {profile!r}; expected one of: {', '.join(sorted(PROFILES))}"
        raise KeyError(msg)
    if iterations < 1:
        msg = f"iterations must be >= 1, got {iterations}"
        raise ValueError(msg)
    if profile == "prod" and not allow_full:
        msg = (
            "profile 'prod' runs the full real build and requires explicit opt-in "
            "(pass allow_full=True); it is intended for the wenceslaus host, not tests."
        )
        raise RuntimeError(msg)

    wd = Workdir(Path(workdir))
    wd.create()
    log = bind(task_id="benchmark", profile=profile, iterations=iterations)
    log.info("benchmark start")

    recorder = _Recorder()
    mem_before = _peak_rss_mb()
    start = time.perf_counter()
    result: PipelineResult | None = None
    with _instrument(recorder):
        for _ in range(iterations):
            result = run_pipeline(profile=profile, fixture_root=fixture_root, workdir=wd.root, run_airflow=False, params=params)
    total_wall = time.perf_counter() - start
    mem_after = _peak_rss_mb()

    report = _build_report(profile, iterations, recorder, total_wall, mem_before, mem_after, result)
    out = wd.reports / f"benchmark_{profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("benchmark complete", wall_seconds=report["totals"]["wall_seconds"], report=str(out))
    return BenchmarkReport(profile=profile, path=out, report=report, result=result)


def _build_report(
    profile: str, iterations: int, recorder: _Recorder, total_wall: float, mem_before: float, mem_after: float, result: PipelineResult | None
) -> dict[str, Any]:
    """Assemble the deterministic-shape benchmark report dict."""
    stages = [
        {
            "name": stage,
            "calls": recorder.stats[stage].calls,
            "wall_seconds": round(recorder.stats[stage].wall_seconds, 6),
            "cache_hits": recorder.stats[stage].cache_hits,
            "cache_misses": recorder.stats[stage].cache_misses,
        }
        for stage in STAGES
    ]
    stages_wall = sum(recorder.stats[stage].wall_seconds for stage in STAGES)
    total_rows = sum(table.rows for table in result.tables.values()) if result is not None else 0
    throughput = total_rows / total_wall if total_wall > 0 else 0.0
    tables = [{"name": name, "rows": result.tables[name].rows} for name in sorted(result.tables)] if result is not None else []
    return {
        "schema_version": REPORT_SCHEMA,
        "profile": profile,
        "generated_at": datetime.now(UTC).isoformat(),
        "iterations": iterations,
        "environment": _environment(),
        "stages": stages,
        "totals": {
            "wall_seconds": round(total_wall, 6),
            "stages_wall_seconds": round(stages_wall, 6),
            "overhead_wall_seconds": round(max(total_wall - stages_wall, 0.0), 6),
            "peak_memory_mb": round(mem_after, 3),
            "peak_memory_delta_mb": round(max(mem_after - mem_before, 0.0), 3),
            "total_rows": total_rows,
            "throughput_rows_per_sec": round(throughput, 3),
            "cache_hits": recorder.total_cache_hits,
            "cache_misses": recorder.total_cache_misses,
        },
        "tables": tables,
    }


__all__ = ["REPORT_SCHEMA", "STAGES", "BenchmarkReport", "run_benchmark"]
