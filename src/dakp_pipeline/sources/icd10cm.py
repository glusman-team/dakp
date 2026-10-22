"""ICD-10-CM Table of Drugs and Chemicals fetcher.

Acquires the CDC/NCHS fiscal-year release of the *Table of Drugs and Chemicals* — the official
ICD-10-CM indexing reference that names every drug, chemical and noxious substance with its
poisoning / adverse-effect / underdosing codes — and content-addresses the release ZIP. The
``icd10cm-drug-<FY>.xml`` member is what :mod:`dakp_pipeline.extract.icd10cm` mines for the
non-therapeutic subject screen.

Why this source, and not a hand-kept list
-----------------------------------------
The Table is the curated authority the U.S. mortality/morbidity coding system already maintains
for "substance with no accepted therapeutic use". Its structure carries the distinction directly:
an entry whose **Adverse effect** column is empty (``--``) is an assertion that the substance has
no use under proper administration, which is exactly the claim DAKP must not attach a
``biolink:applied_to_treat`` edge to. Section I.C.19.e of the FY Official Guidelines defines the
columns (adverse effect = "correct prescription or proper administration"; poisoning = "improper
use"; toxic effect = "harmful substance ... not being used as a drug"), so the empty column is a
coded judgment, not an omission. See :mod:`dakp_pipeline.extract.icd10cm` for the tier rules.

The archive is release-pinned (the fiscal year is in both the directory and the file name), so a
cached copy of a pinned URL is canonical: the freshness gate defaults to none. Publishing a new
fiscal year means bumping :data:`ICD10CM_DRUG_TABLE_ZIP_URL`; the new URL re-acquires under a new
alias and the old release stays addressable.

The download is factored into a module-level :func:`download_icd10cm_zip` seam so tests can
serve a local fixture ZIP without the DAG touching the network.
"""

from __future__ import annotations

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloader import download
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir

#: The release archive DAKP acquires. Pinning the fiscal year keeps the artifact immutable.
ICD10CM_DRUG_TABLE_ZIP_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-table-and-index-2027.zip"
#: The NCHS ICD-10-CM page that links the annual Table and Index files (canonical dataset page).
ICD10CM_TABLES_URL = "https://www.cdc.gov/nchs/icd/icd10cm.htm"
#: Fiscal-year label used in provenance text so a bump is visible in the generated config.
ICD10CM_RELEASE = "FY2027"
#: Prefix of the drug-table XML member inside the release ZIP (matched, not hardcoded, so a
#: fiscal-year rename of the member cannot silently produce an empty screen).
ICD10CM_DRUG_XML_PREFIX = "icd10cm-drug-"
_DEFAULT_TIMEOUT_SECONDS = 300.0
#: Narration prefix for every log line this fetcher emits (one stat per line).
_EVENT = "acquire_icd10cm"


class ICD10CMFetcher:
    """Acquire the ICD-10-CM Table of Drugs and Chemicals release ZIP."""

    url: str = ICD10CM_DRUG_TABLE_ZIP_URL

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, _EVENT):
            store = ArtifactStore(Workdir(ctx.workdir))
            url = str(ctx.params.get("icd10cm_url", self.url))
            force = bool(ctx.params.get("force", False))
            stats(logger, _EVENT, url=url, force=force)
            alias = f"icd10cm/{PurePosixPath(url).name}"
            cached = store.cached_ref(alias)
            manifest = store.read_manifest(cached.blake3) if cached is not None else None
            age = _cache_age_days(manifest)
            max_age = _max_age_days(ctx)
            if (
                not force
                and cached is not None
                and cached.uri.exists()
                and manifest is not None
                and manifest.source.url == url
                and age is not None
                and (max_age is None or age < max_age)
            ):
                stats(logger, _EVENT, cache_hit=True, age_days=round(age, 2), max_age_days=max_age, blake3=cached.blake3)
                return [cached]

            # Stage into a temp path so the only persistent write is the content-addressed
            # store copy. No destructive stashing of prior downloads.
            with tempfile.NamedTemporaryFile(prefix="icd10cm-", suffix=".zip", delete=False) as handle:
                dest = Path(handle.name)
            try:
                started = time.monotonic()
                download_icd10cm_zip(url, dest)
                stats(logger, _EVENT, bytes=dest.stat().st_size, elapsed_s=round(time.monotonic() - started, 3))
                ref, cache_hit = store.ingest(
                    dest, media_type="application/zip", alias=alias, source=SourceBlock(url=url, retrieved_at=datetime.now(UTC).isoformat())
                )
            finally:
                dest.unlink(missing_ok=True)

            stats(logger, _EVENT, blake3=ref.blake3, cache_hit=cache_hit)
            return [ref]


def _max_age_days(ctx: TaskContext) -> float | None:
    """Resolve the cache window. ``None``/non-positive disables the gate (the release is pinned)."""
    value = ctx.params.get("icd10cm_max_age_days")
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None
    return None


def _cache_age_days(manifest: object | None) -> float | None:
    """Age of the cached artifact in days, or ``None`` when it cannot be established."""
    retrieved_at = getattr(getattr(manifest, "source", None), "retrieved_at", None)
    if not retrieved_at:
        return None
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(retrieved_at)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return None


def download_icd10cm_zip(url: str, dest: Path, *, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> Path:
    """Download ``url`` to ``dest`` (aria2c-accelerated, stdlib fallback). Monkeypatchable.

    Tests replace this to serve a local fixture ZIP without network; the real path is covered by
    the downloader unit tests and a live acquire on the verification host.
    """
    return download(url, dest, timeout=timeout)


fetch = ICD10CMFetcher().fetch

__all__ = [
    "ICD10CM_DRUG_TABLE_ZIP_URL",
    "ICD10CM_DRUG_XML_PREFIX",
    "ICD10CM_RELEASE",
    "ICD10CM_TABLES_URL",
    "ICD10CMFetcher",
    "download_icd10cm_zip",
    "fetch",
]
