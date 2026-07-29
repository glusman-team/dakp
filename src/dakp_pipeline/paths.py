"""Configurable workdir layout. No absolute paths anywhere in the pipeline.

All paths are derived from a single workdir root, which is supplied by the caller
(CLI ``--workdir``, Airflow DAG param ``workdir``, or ``tmp_path`` in tests). The
layout mirrors the directory tree sketched in ``PLAN.md`` ("Proposed repository layout").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workdir:
    """Immutable handle to the pipeline workdir tree.

    Constructing an instance does not create directories; call :meth:`create` to
    materialize the full tree (idempotent).
    """

    root: Path

    def __post_init__(self) -> None:
        # Normalize without resolving symlinks: we never want to quietly rewrite a
        # user-supplied path that contains intentional symlink components.
        object.__setattr__(self, "root", Path(self.root))

    # -- raw acquisition layer -------------------------------------------------
    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def by_hash(self) -> Path:
        """Content-addressed store: ``data/raw/by-hash/<hash>/``."""
        return self.raw / "by-hash"

    @property
    def aliases(self) -> Path:
        """Human-readable aliases into the content-addressed store."""
        return self.raw / "aliases"

    # -- processed layers ------------------------------------------------------
    @property
    def interim(self) -> Path:
        """Partitioned parquet interim tables (``data/interim/``)."""
        return self.root / "data" / "interim"

    @property
    def tabular(self) -> Path:
        """Uncompressed TSV assertion tables consumed by Tablassert (``data/tabular/``)."""
        return self.root / "data" / "tabular"

    @property
    def kgx(self) -> Path:
        """KGX NDJSON outputs produced by Tablassert (``data/kgx/``)."""
        return self.root / "data" / "kgx"

    @property
    def manifests(self) -> Path:
        """Per-artifact JSON manifests (``data/manifests/``)."""
        return self.root / "data" / "manifests"

    @property
    def store(self) -> Path:
        """Typed/derived store artifacts (``data/store/``)."""
        return self.root / "data" / "store"

    @property
    def reports(self) -> Path:
        """Per-task ``task_report.json`` and build summaries."""
        return self.root / "data" / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def create(self) -> None:
        """Create the full directory tree (idempotent)."""
        for path in (self.by_hash, self.aliases, self.interim, self.tabular, self.kgx, self.manifests, self.store, self.reports, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def default_workdir() -> Workdir:
    """Workdir used when the caller does not pass one (repo-local ``data/``).

    Production/full builds always pass an explicit workdir; this default exists only
    for ad-hoc local invocation and tests.
    """
    return Workdir(Path("data"))
