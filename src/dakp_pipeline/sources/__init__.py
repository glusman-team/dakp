"""Source fetchers (stubs). Real network acquisition lands in **Milestone 2**.

Each module exposes a module-level :func:`fetch` (the default instance method) so tests
can ``monkeypatch.setattr(dailymed, "fetch", ...)`` and the pure-Python runner can call
``dailymed.fetch(ctx)``. In the ``mock`` profile, fetchers ingest tiny fixtures into the
content-addressed store; any other profile fails loudly (no silent network fallback).
"""

from __future__ import annotations

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir


def require_mock(ctx: TaskContext, source_name: str) -> None:
    """Raise ``NotImplementedError`` unless the profile is mock.

    The mock profile loads fixtures; real DailyMed/FAERS/Drugs@FDA acquisition is
    implemented in Milestone 2.
    """
    if ctx.profile != "mock":
        msg = f"real acquisition for {source_name!r} lands in Milestone 2; only the mock profile is implemented (got profile={ctx.profile!r})"
        raise NotImplementedError(msg)


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


__all__ = ["ingest_fixtures", "require_mock"]
