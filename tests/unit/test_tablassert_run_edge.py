"""Edge-case tests for the ``dakp_pipeline.tablassert`` runner.

Covers the real ``run_subprocess`` body (a genuine subprocess, including a non-zero exit that
must NOT raise) and the ``_find_graph`` conventional-path fallback when no ``graph.yaml`` ref is
present.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import _find_graph, run_subprocess


def _ctx(workdir: Path) -> TaskContext:
    return TaskContext(workdir=workdir, fixture_root=None, params={})


def test_run_subprocess_captures_stdout() -> None:
    completed = run_subprocess([sys.executable, "-c", "print('hello')"])
    assert completed.returncode == 0
    assert completed.stdout.strip() == "hello"


def test_run_subprocess_captures_nonzero_without_raising() -> None:
    completed = run_subprocess([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert completed.returncode == 3


def test_find_graph_falls_back_to_conventional_path(tmp_path: Path) -> None:
    other = ArtifactRef(uri=tmp_path / "approved_treats.yaml", blake3="b3:" + "0" * 64, media_type="application/yaml")
    assert _find_graph([other], _ctx(tmp_path)) == Workdir(tmp_path).root / "tables" / "graph.yaml"


def test_find_graph_prefers_the_graph_ref(tmp_path: Path) -> None:
    graph = ArtifactRef(uri=tmp_path / "graph.yaml", blake3="b3:" + "1" * 64, media_type="application/yaml")
    assert _find_graph([graph], _ctx(tmp_path)) == tmp_path / "graph.yaml"
