"""Pure-Python pipeline runner.

``run_pipeline`` is the single orchestration entry point shared by the CLI and the
(optional) Airflow DAG. It wires the staged stubs end-to-end and is fully
monkeypatchable: fetchers and :mod:`dakp_pipeline.tablassert.run` are always resolved
through their owning module at call time, so ``monkeypatch.setattr(module, "fetch", ...)``
and ``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` take effect (see the
integration test in ``PLAN.md``).

Milestone 1 ships a *minimal* runner that exercises config/paths/logging and returns an
empty :class:`PipelineResult`. The full wiring (acquire -> extract -> shape -> handoff)
is filled in by the pipeline-runner commit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dakp_pipeline.config import Profile, load_profile
from dakp_pipeline.logging_setup import bind, configure_logging
from dakp_pipeline.paths import Workdir


@dataclass
class TableResult:
    """Summary of one produced tabular output."""

    name: str
    path: Path
    rows: int

    def exists(self) -> bool:
        return self.path.exists()


@dataclass
class PipelineResult:
    """Return value of :func:`run_pipeline`. Exposes table summaries for assertions."""

    workdir: Workdir
    profile: Profile
    tables: dict[str, TableResult] = field(default_factory=dict)
    build_summary: Path | None = None

    def table(self, name: str) -> TableResult:
        if name not in self.tables:
            available = ", ".join(sorted(self.tables)) or "<none>"
            msg = f"No table named {name!r} in result; available: {available}"
            raise KeyError(msg)
        return self.tables[name]


def run_pipeline(
    profile: str = "mock",
    fixture_root: Path | str | None = None,
    workdir: Path | str = "data",
    run_airflow: bool = False,
    params: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Run the (mocked) DAKP pipeline end-to-end.

    Milestone-1 minimal implementation: resolve the profile, materialize the workdir,
    configure logging, and return an empty result. Real stage wiring lands in the
    pipeline-runner commit; the signature and monkeypatchable boundaries are stable now.
    """
    resolved_profile = load_profile(profile)
    wd = Workdir(Path(workdir))
    wd.create()
    configure_logging(wd.logs.parent, level="INFO")

    fixture = Path(fixture_root) if fixture_root is not None else None
    log = bind(task_id="run_pipeline", profile=profile, workdir=str(wd.root))
    log.info("pipeline start (milestone-1 minimal runner)", fixture_root=str(fixture))

    return PipelineResult(workdir=wd, profile=resolved_profile)


__all__ = ["PipelineResult", "TableResult", "run_pipeline"]
