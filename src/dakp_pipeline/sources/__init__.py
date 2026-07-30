"""Source fetchers for DailyMed, FAERS, and Drugs@FDA.

Each source module exposes a module-level :func:`fetch` (the default instance method) so tests
can ``monkeypatch.setattr(dailymed, "fetch", ...)`` and the pure-Python runner can call
``dailymed.fetch(ctx)``. Mock profiles ingest tiny fixtures into the content-addressed store;
real profiles use source-specific stdlib downloaders.
"""

from __future__ import annotations

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir


def ingest_fixtures(ctx: TaskContext, names: tuple[str, ...], *, namespace: str) -> list[ArtifactRef]:
    """Content-address each named fixture file under ``fixture_root`` and return refs."""
    if ctx.fixture_root is None:
        msg = "TaskContext.fixture_root is None; cannot ingest fixtures"
        raise ValueError(msg)
    store = ArtifactStore(Workdir(ctx.workdir))
    refs: list[ArtifactRef] = []
    for name in names:
        path = ctx.fixture_root / name
        ref, _ = store.ingest(path, alias=f"{namespace}/{name}")
        refs.append(ref)
    return refs


__all__ = ["ingest_fixtures"]
