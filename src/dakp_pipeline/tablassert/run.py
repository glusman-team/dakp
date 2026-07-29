"""Tablassert runner — shell out to ``../Tablassert`` (Milestone 7).

DAKP ships NO local KGX compiler and adds NO ``tablassert`` import / hard dependency
(PLAN.md "Tablassert modeling layer"). The real runner (:class:`TablassertRunner`) shells
out to a local ``../Tablassert`` editable checkout via ``uv run --with-editable`` and
captures stdout / exit code into a handoff report; the mock runner
(:class:`MockTablassertRunner`) writes a deferred-handoff report without ever touching
Tablassert (default in the ``mock`` profile and in tests).

The module-level :func:`run` is the package entry point used by ``pipeline.py`` and
``dags.dakp_build`` (``from dakp_pipeline.tablassert import run``); it dispatches to the
real or mock runner based on ``ctx``. Tests monkeypatch either :func:`run` itself or the
runner's subprocess hook (:func:`run_subprocess`) — no real Tablassert required.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.paths import Workdir

DEFAULT_TABLASERT_DIR = "../Tablassert"
DEFAULT_FULLMAP = ".fullmap"
REPORT_NAME = "tablassert_handoff.json"
_REPORT_SCHEMA = "dakp.tablassert_handoff.v1"
_OPERATION = "run_tablassert"


def run_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Execute ``command`` capturing stdout/stderr, never raising on non-zero exit.

    This is the monkeypatch point for tests: patch the ``run_subprocess`` attribute on THIS
    module and no real process is spawned. (The package ``__init__`` re-exports the ``run``
    function, shadowing the ``run`` submodule attribute, so resolve the module via
    ``importlib.import_module("dakp_pipeline.tablassert.run")`` rather than the package attr.)
    """
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _base_report(mode: str, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef]) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "stage": "tablassert_handoff",
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "assertion_inputs": [{"table": ref.uri.stem, "artifact_id": ref.blake3, "rows": ref.rows} for ref in assertion_refs],
        "config_inputs": [str(ref.uri) for ref in config_refs],
    }


def _write_report(report: dict[str, Any], assertion_refs: list[ArtifactRef], ctx: TaskContext) -> ArtifactRef:
    workdir = Workdir(ctx.workdir)
    path = workdir.reports / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    store = ArtifactStore(workdir)
    return store.register(
        path, media_type="application/json", inputs=[ref.blake3 for ref in assertion_refs], operation=OperationBlock(name=_OPERATION)
    )


def _find_graph(config_refs: list[ArtifactRef], ctx: TaskContext) -> Path:
    """Locate ``graph.yaml`` among the generated config refs (fall back to the conventional path)."""
    for ref in config_refs:
        if ref.uri.name == "graph.yaml":
            return ref.uri
    return Workdir(ctx.workdir).root / "tables" / "graph.yaml"


@dataclass(frozen=True)
class TablassertRunner:
    """Run ``../Tablassert`` as a subprocess; no ``tablassert`` import, no hard dependency.

    Builds ``uv run --with-editable <tablassert_dir> tablassert build-kg <graph.yaml>
    --fullmap <path>`` and records stdout / stderr / exit code in the handoff report. A
    non-zero exit is captured as ``status: failed`` (logged loudly), not raised — the
    report is the artifact the pipeline surfaces.
    """

    tablassert_dir: str = DEFAULT_TABLASERT_DIR

    def build_command(self, graph_yaml: Path, fullmap: str, tablassert_dir: str) -> list[str]:
        """The exact Tablassert invocation (pure; testable without spawning a process)."""
        return ["uv", "run", "--with-editable", tablassert_dir, "tablassert", "build-kg", str(graph_yaml), "--fullmap", fullmap]

    def run(self, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        graph_yaml = _find_graph(config_refs, ctx)
        fullmap = str(ctx.params.get("fullmap") or DEFAULT_FULLMAP)
        tablassert_dir = str(ctx.params.get("tablassert_dir") or self.tablassert_dir)
        command = self.build_command(graph_yaml, fullmap, tablassert_dir)
        cwd = Workdir(ctx.workdir).root

        logger.info("running Tablassert: {}", " ".join(command))
        completed = run_subprocess(command, cwd=cwd)
        status = "ok" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            logger.error("Tablassert exited {}: {}", completed.returncode, (completed.stderr or "").strip()[:2000])

        report = _base_report("real", assertion_refs, config_refs)
        report.update(
            {
                "status": status,
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "graph_config": str(graph_yaml),
                "fullmap": fullmap,
                "tablassert_dir": tablassert_dir,
            }
        )
        return [_write_report(report, assertion_refs, ctx)]


@dataclass(frozen=True)
class MockTablassertRunner:
    """Write a deferred-handoff report; never touch ``../Tablassert`` (mock profile + tests)."""

    def run(self, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        report = _base_report("mock", assertion_refs, config_refs)
        report.update(
            {
                "status": "deferred",
                "reason": "mock profile / run_tablassert disabled; canonical resolution + KGX compilation delegated to ../Tablassert",
            }
        )
        return [_write_report(report, assertion_refs, ctx)]


def run(assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Package entry point (``pipeline.py`` / ``dags.dakp_build``): dispatch to a runner.

    Real execution requires ``run_tablassert`` truthy in ``ctx.params`` AND a non-``mock``
    profile; otherwise the mock runner writes a deferred-handoff report. Returns a list with
    one ArtifactRef to the handoff report.
    """
    run_real = bool(ctx.params.get("run_tablassert")) and ctx.profile != "mock"
    runner: TablassertRunner | MockTablassertRunner = TablassertRunner() if run_real else MockTablassertRunner()
    return runner.run(assertion_refs, config_refs, ctx)


__all__ = ["DEFAULT_FULLMAP", "DEFAULT_TABLASERT_DIR", "REPORT_NAME", "MockTablassertRunner", "TablassertRunner", "run", "run_subprocess"]
