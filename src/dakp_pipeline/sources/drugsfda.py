"""Drugs@FDA data-files fetcher.

Acquires the FDA "Drugs@FDA Data Files" ZIP — the same artifact the legacy
``ref/legacy/DrugsFDA/bin/download.pl`` fetched from ``https://www.fda.gov/media/89850/download`` —
content-addresses it with BLAKE3, and returns one :class:`ArtifactRef`.

Streams the ZIP with stdlib :mod:`urllib` (no new dependency) into a temp path, then
:meth:`ArtifactStore.ingest` copies it into the content-addressed store with a provenance
manifest. Identical bytes hash to the same store path, so re-runs are a cache hit (no copy,
manifest reused). Offline tests monkeypatch :func:`download_drugsfda_zip` to serve a local
fixture ZIP.

Idempotent and non-destructive: the only writes are the content-addressed store copy, its
alias, and its manifest. The download target is monkeypatchable via
:func:`download_drugsfda_zip` / ``ctx.params["drugsfda_url"]`` / :pyattr:`DrugsFDAFetcher.url`.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir

# The FDA "Drugs@FDA Data Files" ZIP (legacy ref/legacy/DrugsFDA/bin/download.pl target).
DRUGSFDA_DATA_FILES_URL = "https://www.fda.gov/media/89850/download"
#: Narration prefix for every log line this fetcher emits (one stat per line).
_EVENT = "acquire_drugsfda"


class DrugsFDAFetcher:
    """Acquire Drugs@FDA product/application/submission tables over the network."""

    url: str = DRUGSFDA_DATA_FILES_URL

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, _EVENT):
            store = ArtifactStore(Workdir(ctx.workdir))
            url = str(ctx.params.get("drugsfda_url", self.url))
            stats(logger, _EVENT, url=url)

            # Stage into a temp path so the only persistent write is the content-addressed
            # store copy. No destructive stashing of prior downloads.
            with tempfile.NamedTemporaryFile(prefix="drugsfda-", suffix=".zip", delete=False) as handle:
                dest = Path(handle.name)
            try:
                started = time.monotonic()
                download_drugsfda_zip(url, dest)
                stats(logger, _EVENT, bytes=dest.stat().st_size, elapsed_s=round(time.monotonic() - started, 3))
                ref, cache_hit = store.ingest(
                    dest, media_type="application/zip", alias="drugsfda/drugsfda_data_files.zip", source=SourceBlock(url=url)
                )
            finally:
                dest.unlink(missing_ok=True)

            stats(logger, _EVENT, blake3=ref.blake3, cache_hit=cache_hit)
            return [ref]


def download_drugsfda_zip(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
    """Stream ``url`` to ``dest`` using stdlib (no extra dependency).

    Monkeypatchable: tests replace this to serve a local fixture ZIP without network.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "dakp-pipeline/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    return dest


fetch = DrugsFDAFetcher().fetch

__all__ = ["DRUGSFDA_DATA_FILES_URL", "DrugsFDAFetcher", "download_drugsfda_zip", "fetch"]
