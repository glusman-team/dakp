"""Drugs@FDA data-files fetcher.

Acquires the FDA "Drugs@FDA Data Files" ZIP — the same artifact the legacy
``ref/legacy/DrugsFDA/bin/download.pl`` fetched from ``https://www.fda.gov/media/89850/download`` —
content-addresses it with BLAKE3, and returns one :class:`ArtifactRef`.

A cached copy younger than the default seven-day freshness window is reused without network I/O;
``force`` bypasses the gate. This bounds staleness for FDA's fixed-name, replace-in-place ZIP.

Downloads the ZIP (aria2c-accelerated via the bundled :mod:`aria2` wheel, with a stdlib
:mod:`urllib` fallback) into a temp path, then
:meth:`ArtifactStore.ingest` copies it into the content-addressed store with a provenance
manifest. Identical bytes hash to the same store path, so re-runs are a cache hit (no copy,
manifest reused). Offline tests monkeypatch :func:`download_drugsfda_zip` to serve a local
fixture ZIP.

Idempotent and non-destructive: the only writes are the content-addressed store copy, its
alias, and its manifest. The download target is monkeypatchable via
:func:`download_drugsfda_zip` / ``ctx.params["drugsfda_url"]`` / :pyattr:`DrugsFDAFetcher.url`.
"""

from __future__ import annotations

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloader import download
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir

# The FDA "Drugs@FDA Data Files" ZIP (legacy ref/legacy/DrugsFDA/bin/download.pl target).
DRUGSFDA_DATA_FILES_URL = "https://www.fda.gov/media/89850/download"
_DEFAULT_MAX_AGE_DAYS = 7.0
#: Narration prefix for every log line this fetcher emits (one stat per line).
_EVENT = "acquire_drugsfda"


class DrugsFDAFetcher:
    """Acquire Drugs@FDA product/application/submission tables over the network."""

    url: str = DRUGSFDA_DATA_FILES_URL

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, _EVENT):
            store = ArtifactStore(Workdir(ctx.workdir))
            url = str(ctx.params.get("drugsfda_url", self.url))
            force = bool(ctx.params.get("force", False))
            stats(logger, _EVENT, url=url, force=force)
            alias = "drugsfda/drugsfda_data_files.zip"
            cached = store.cached_ref(alias)
            manifest = store.read_manifest(cached.blake3) if cached is not None else None
            age = _cache_age_days(manifest)
            max_age = _max_age_days(ctx)
            if (
                not force
                and max_age is not None
                and cached is not None
                and cached.uri.exists()
                and manifest is not None
                and manifest.source.url == url
                and age is not None
                and age < max_age
            ):
                stats(logger, _EVENT, cache_hit=True, age_days=round(age, 2), max_age_days=max_age, blake3=cached.blake3)
                return [cached]

            # Stage into a temp path so the only persistent write is the content-addressed
            # store copy. No destructive stashing of prior downloads.
            with tempfile.NamedTemporaryFile(prefix="drugsfda-", suffix=".zip", delete=False) as handle:
                dest = Path(handle.name)
            try:
                started = time.monotonic()
                download_drugsfda_zip(url, dest)
                stats(logger, _EVENT, bytes=dest.stat().st_size, elapsed_s=round(time.monotonic() - started, 3))
                ref, cache_hit = store.ingest(
                    dest, media_type="application/zip", alias=alias, source=SourceBlock(url=url, retrieved_at=datetime.now(UTC).isoformat())
                )
            finally:
                dest.unlink(missing_ok=True)

            stats(logger, _EVENT, blake3=ref.blake3, cache_hit=cache_hit)
            return [ref]


def _max_age_days(ctx: TaskContext) -> float | None:
    """Resolve the Drugs@FDA cache window; ``None``/non-positive disables the gate."""
    value = ctx.params.get("drugsfda_max_age_days")
    if value is None:
        return _DEFAULT_MAX_AGE_DAYS
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None
    return _DEFAULT_MAX_AGE_DAYS


def _cache_age_days(manifest: object | None) -> float | None:
    """Return the age of a cached manifest, or ``None`` when provenance is unusable."""
    retrieved_at = getattr(getattr(manifest, "source", None), "retrieved_at", None)
    if not retrieved_at:
        return None
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(retrieved_at)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return None


def download_drugsfda_zip(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` (aria2c-accelerated, stdlib fallback). Monkeypatchable.

    Tests replace this to serve a local fixture ZIP without network; the real path is covered by
    the offline prod-smoke test and the downloader unit tests.
    """
    return download(url, dest, timeout=timeout, headers={"User-Agent": "dakp-pipeline/0.1"})


fetch = DrugsFDAFetcher().fetch

__all__ = ["DRUGSFDA_DATA_FILES_URL", "DrugsFDAFetcher", "download_drugsfda_zip", "fetch"]
