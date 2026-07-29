"""Drugs@FDA data-files fetcher (Milestone 2).

Acquires the FDA "Drugs@FDA Data Files" ZIP — the same artifact the legacy
``DrugsFDA/bin/download.pl`` fetched from ``https://www.fda.gov/media/89850/download`` —
content-addresses it with BLAKE3, and returns one :class:`ArtifactRef`.

* ``mock`` profile: ingests tiny Products/Applications/Submissions fixtures so the
  pipeline (and tests) never touch the network.
* any other profile: streams the ZIP with stdlib :mod:`urllib` (no new dependency) into a
  temp path, then :meth:`ArtifactStore.ingest` copies it into the content-addressed store
  with a provenance manifest. Identical bytes hash to the same store path, so re-runs are
  a cache hit (no copy, manifest reused).

Idempotent and non-destructive: the only writes are the content-addressed store copy, its
alias, and its manifest. The download target is monkeypatchable via
:func:`download_drugsfda_zip` / ``ctx.params["drugsfda_url"]`` / :pyattr:`DrugsFDAFetcher.url`.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import ingest_fixtures

# The FDA "Drugs@FDA Data Files" ZIP (legacy DrugsFDA/bin/download.pl target).
DRUGSFDA_DATA_FILES_URL = "https://www.fda.gov/media/89850/download"

# Fixture refs used by the mock profile; they mirror the real Products/Applications/
# Submissions tab-delimited tables so the extractor treats mock and real inputs alike.
_DRUGSFDA_FIXTURES = ("drugsfda/drugsfda_products.tsv", "drugsfda/drugsfda_applications.tsv", "drugsfda/drugsfda_submissions.tsv")


class DrugsFDAFetcher:
    """Acquire Drugs@FDA product/application/submission tables."""

    url: str = DRUGSFDA_DATA_FILES_URL

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        if ctx.profile == "mock":
            return ingest_fixtures(ctx, _DRUGSFDA_FIXTURES, namespace="drugsfda")
        return self._fetch_real(ctx)

    def _fetch_real(self, ctx: TaskContext) -> list[ArtifactRef]:
        store = ArtifactStore(Workdir(ctx.workdir))
        url = str(ctx.params.get("drugsfda_url", self.url))
        log = bind(task_id="acquire_drugsfda", url=url)
        log.info("downloading Drugs@FDA data-files zip")

        # Stage into a temp path so the only persistent write is the content-addressed
        # store copy. No destructive stashing of prior downloads.
        with tempfile.NamedTemporaryFile(prefix="drugsfda-", suffix=".zip", delete=False) as handle:
            dest = Path(handle.name)
        try:
            download_drugsfda_zip(url, dest)
            ref, cache_hit = store.ingest(dest, media_type="application/zip", alias="drugsfda/drugsfda_data_files.zip", source=SourceBlock(url=url))
        finally:
            if dest.exists():
                dest.unlink()

        log.info("acquired Drugs@FDA zip", artifact_id=ref.blake3, cache_hit=cache_hit)
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
