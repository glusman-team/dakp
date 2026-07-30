"""Core source-shaping interfaces — exactly as sketched in ``PLAN.md`` ("Core
source-shaping interfaces").

Every stage of the pipeline communicates through :class:`ArtifactRef` handles (paths +
BLAKE3 ids + optional manifest/schema metadata) rather than in-memory dataframes, so
tasks are restartable, cacheable by hash, and easy to monkeypatch in tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.downloads import infer_media_type


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable handle to one pipeline artifact (raw, interim, or tabular).

    ``uri`` is the concrete path to read from; ``blake3`` is its canonical
    ``b3:<hex>`` content id; ``manifest`` points at the JSON manifest describing
    provenance/inputs (written by :class:`~dakp_pipeline.io.artifact_store.ArtifactStore`).
    """

    uri: Path
    blake3: str
    media_type: str
    rows: int | None = None
    schema_fingerprint: str | None = None
    manifest: Path | None = None


@dataclass(frozen=True)
class TaskContext:
    """Per-task execution context passed to every fetcher/extractor/transformer.

    Fields match the PLAN.md sketch exactly. Fetchers/extractors build their own
    :class:`~dakp_pipeline.io.artifact_store.ArtifactStore` from ``workdir`` on demand,
    keeping the context itself lightweight and serializable.
    """

    profile: str  # mock | sample | prod
    workdir: Path
    fixture_root: Path | None
    threads: int
    memory_budget_gb: int
    params: Mapping[str, Any]

    def fixture(self, name: str) -> ArtifactRef:
        """Return an :class:`ArtifactRef` for a fixture file under ``fixture_root``.

        Used by mocked fetchers (and by the PLAN.md integration-test sketch, e.g.
        ``ctx.fixture("dailymed_release.zip")``). The ref points directly at the fixture
        file; it is not copied into the content-addressed store.
        """
        if self.fixture_root is None:
            msg = "TaskContext.fixture_root is None; cannot resolve fixture"
            raise ValueError(msg)
        path = self.fixture_root / name
        if not path.exists():
            msg = f"fixture not found: {path}"
            raise FileNotFoundError(msg)
        return ArtifactRef(uri=path, blake3=hash_file(path), media_type=infer_media_type(path))


@runtime_checkable
class Fetcher(Protocol):
    """Acquire raw source artifacts (network or fixtures)."""

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]: ...


@runtime_checkable
class Extractor(Protocol):
    """Parse raw artifacts into normalized interim tables."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]: ...


@runtime_checkable
class Transformer(Protocol):
    """Shape interim tables into assertion-ready outputs."""

    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]: ...


__all__ = ["ArtifactRef", "Extractor", "Fetcher", "TaskContext", "Transformer"]
