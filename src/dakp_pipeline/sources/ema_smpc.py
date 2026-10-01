"""EMA product-information (SmPC) PDF crawl.

Acquires the English, human, centrally-authorised **product information** documents — the ones
carrying the SmPC sections DAKP mines (``4.1 Therapeutic indications``, ``4.3 Contraindications``,
``4.4 Special warnings``) — and content-addresses each PDF as its own artifact. The document list
comes from the EPAR documents report (:mod:`dakp_pipeline.sources.ema_documents`), which is itself
a fixed-name nightly export, so the crawl is *bounded by a manifest* rather than discovered by
scraping: 1,945 documents on the 2026-10-01 report.

Two cache layers keep a re-run cheap, because a SmPC changes rarely while the report is
regenerated nightly:

1. **Fan-out completion record** keyed on :func:`crawl_fingerprint` — the BLAKE3 of the selection's
   ``(stem, url, last_updated_date)`` triples. Unchanged selection => the whole crawl is a cache
   hit with zero HTTP traffic. (Keying on the manifest's artifact id instead would invalidate the
   crawl every night, because the report embeds a regeneration timestamp.)
2. **Per-document currency** — a stored PDF is reused when its manifest ``retrieved_at`` is at or
   after that document's ``last_updated_date``, i.e. we already fetched this version. Documents
   whose ``last_updated_date`` is missing or unparsable fall back to the crawl-level
   ``ema_smpc_max_age_days`` window (default 30 days, the SmPC revision cadence is months).

Downloads run on a bounded thread pool (``ema_smpc_concurrency``, default 8); ingest runs serially
in the calling thread so the returned refs — and therefore every downstream artifact hash — are in
selection order regardless of network timing.

A per-document failure never aborts the crawl: it is logged, counted, and left out of the fan-out
record so the next run retries exactly the missing documents. A crawl that yields no documents at
all raises, because an empty SmPC corpus would silently empty the EU contraindication family.

``ema_smpc_limit`` bounds the crawl to the first N documents (smoke tests, the offline prod run).
The download target is monkeypatchable via :func:`download_ema_smpc_pdf`. There is deliberately no
URL-override param: document URLs come from the manifest, which is the authoritative record of what
was fetched.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_bytes
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloader import download
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import logger, progress, stats, step
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import ema_documents
from dakp_pipeline.sources.ema_documents import EparDocument, load_documents, product_information_documents

#: Fan-out source alias. Member aliases are ``<this>::<document stem>.pdf`` (the store's
#: fan-out convention, which also lets it prune members that left the selection).
SMPD_FANOUT_ALIAS = "ema/smpc/product_information_en"
PDF_MEDIA_TYPE = "application/pdf"
_DEFAULT_CONCURRENCY = 8
_DEFAULT_MAX_AGE_DAYS = 30.0
_CRAWL_PROGRESS_EVERY = 100
_DOWNLOAD_ATTEMPTS = 2
_RETRY_BACKOFF_S = 1.0
#: Narration prefix for every log line this fetcher emits (one stat per line).
_EVENT = "acquire_ema_smpc"


@dataclass(frozen=True)
class _CrawlFailure:
    """One document that could not be fetched (structured, so failures are attributable)."""

    document: EparDocument
    error: str


def member_alias(doc: EparDocument) -> str:
    """The content-address alias for one product-information PDF."""
    return f"{SMPD_FANOUT_ALIAS}::{doc.stem}.pdf"


def crawl_fingerprint(documents: list[EparDocument]) -> str:
    """Stable key for the fan-out cache: what is in the selection, at which revision.

    Sorted so the key depends only on the selection's CONTENT, not on report order, and
    deliberately excludes the manifest's own regeneration timestamp so a nightly report that
    changes nothing does not invalidate 1,945 cached PDFs.
    """
    payload = json.dumps(sorted([doc.stem, doc.document_url, doc.last_updated_date] for doc in documents), ensure_ascii=True, separators=(",", ":"))
    return hash_bytes(payload.encode("utf-8"))


class EmaSmpcFetcher:
    """Acquire every English product-information PDF the documents report lists."""

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, _EVENT):
            store = ArtifactStore(Workdir(ctx.workdir))
            force = bool(ctx.params.get("force", False))
            concurrency = _concurrency(ctx)
            max_age = _max_age_days(ctx)

            # The manifest fetcher has its own freshness gate, so this is a store hit on a re-run.
            manifest_ref = ema_documents.fetch(ctx)[0]
            documents = _apply_limit(product_information_documents(load_documents(manifest_ref.uri)), ctx)
            if not documents:
                msg = f"EMA product-information crawl: the documents report selected no PDFs (manifest {manifest_ref.blake3})"
                raise RuntimeError(msg)

            fingerprint = crawl_fingerprint(documents)
            stats(logger, _EVENT, documents=len(documents), force=force, concurrency=concurrency, max_age_days=max_age, fingerprint=fingerprint)
            if not force:
                cached_all = store.cached_refs(SMPD_FANOUT_ALIAS, fingerprint)
                if cached_all is not None:
                    stats(logger, _EVENT, cache_hit=True, documents_ingested=len(cached_all))
                    return cached_all
                store.invalidate_cached_refs(SMPD_FANOUT_ALIAS)

            plan = [(doc, None if force else _current_ref(store, doc, max_age)) for doc in documents]
            missing = [doc for doc, ref in plan if ref is None]
            stats(logger, _EVENT, reused=len(documents) - len(missing), to_fetch=len(missing))

            staging = Workdir(ctx.workdir).root / ".staging" / "ema_smpc"
            staging.mkdir(parents=True, exist_ok=True)
            downloaded = _download_all(missing, staging, concurrency)

            refs: list[ArtifactRef] = []
            failures: list[_CrawlFailure] = []
            by_stem = {doc.stem: (path, error) for doc, path, error in downloaded}
            for doc, cached_ref in plan:
                if cached_ref is not None:
                    refs.append(cached_ref)
                    continue
                path, error = by_stem[doc.stem]
                if path is None:
                    failures.append(_CrawlFailure(document=doc, error=error or "unknown download failure"))
                    continue
                try:
                    ref, cache_hit = store.ingest(
                        path, media_type=PDF_MEDIA_TYPE, alias=member_alias(doc), source=SourceBlock(url=doc.document_url, retrieved_at=_now_iso())
                    )
                finally:
                    path.unlink(missing_ok=True)
                refs.append(ref)
                if not cache_hit:
                    stats(logger, _EVENT, level="DEBUG", ingested=doc.stem, blake3=ref.blake3)

            for failure in failures:
                logger.warning(
                    "{}: product-information fetch failed for {} ({}): {}",
                    _EVENT,
                    failure.document.medicine_name,
                    failure.document.ema_product_number,
                    failure.error,
                )
            stats(logger, _EVENT, documents_ingested=len(refs), fetched=len(refs) - (len(documents) - len(missing)), failed=len(failures))
            if not refs:
                msg = f"EMA product-information crawl: every one of the {len(documents)} documents failed to download"
                raise RuntimeError(msg)
            if not failures:
                # Published only for a complete crawl, so a partial one retries next run.
                store.write_cached_refs(SMPD_FANOUT_ALIAS, fingerprint, [member_alias(doc) for doc, _ in plan])
            return refs


def _current_ref(store: ArtifactStore, doc: EparDocument, max_age: float | None) -> ArtifactRef | None:
    """The stored PDF for ``doc`` when we already hold the version the report points at.

    Currency is decided by the report's own ``last_updated_date`` against the stored artifact's
    ``retrieved_at``: a document published after our fetch is stale and must be re-downloaded. A
    document with no usable date falls back to the crawl-level age window.
    """
    alias = member_alias(doc)
    cached = store.cached_ref(alias)
    if cached is None or not cached.uri.exists():
        return None
    manifest = store.read_manifest(cached.blake3)
    retrieved_at = _parse_datetime(getattr(getattr(manifest, "source", None), "retrieved_at", None))
    updated_at = _parse_datetime(doc.last_updated_date)
    if retrieved_at is not None and updated_at is not None:
        return cached if retrieved_at >= updated_at else None
    age = _age_days(retrieved_at)
    if max_age is None or age is None or age >= max_age:
        return None
    return cached


def _download_all(documents: list[EparDocument], staging: Path, concurrency: int) -> list[tuple[EparDocument, Path | None, str | None]]:
    """Fetch every missing PDF on a bounded pool; returns one entry per document, in order."""
    if not documents:
        return []
    total = len(documents)
    results: list[tuple[EparDocument, Path | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index, result in enumerate(pool.map(lambda doc: _download_one(doc, staging), documents), start=1):
            results.append(result)
            progress(logger, f"{_EVENT} crawl", index, total, every=_CRAWL_PROGRESS_EVERY)
    return results


def _download_one(doc: EparDocument, staging: Path) -> tuple[EparDocument, Path | None, str | None]:
    """Download one PDF with a single retry; never raises (the caller records the failure)."""
    dest = staging / f"{doc.stem}.pdf"
    last_error = "not attempted"
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            download_ema_smpc_pdf(doc.document_url, dest)
            if not dest.exists() or dest.stat().st_size == 0:
                msg = f"download produced an empty file: {dest}"
                raise OSError(msg)
            return (doc, dest, None)
        except Exception as exc:  # one bad document must not abort a 1,945-document crawl
            last_error = f"{type(exc).__name__}: {exc}"
            dest.unlink(missing_ok=True)
            if attempt < _DOWNLOAD_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S)
    return (doc, None, last_error)


def download_ema_smpc_pdf(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
    """Download ``url`` to ``dest`` (aria2c-accelerated, stdlib fallback). Monkeypatchable.

    Tests replace this to serve committed fixture PDFs without network; the real path is covered
    by the downloader unit tests and the offline prod-smoke crawl (``ema_smpc_limit``).
    """
    return download(url, dest, timeout=timeout, headers={"User-Agent": "dakp-pipeline/0.1"})


def _apply_limit(documents: list[EparDocument], ctx: TaskContext) -> list[EparDocument]:
    """Slice to the first N documents when ``ema_smpc_limit`` is set (<=0 / None = all)."""
    limit = ctx.params.get("ema_smpc_limit")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return documents
    return documents[:limit]


def _concurrency(ctx: TaskContext) -> int:
    """Resolve the crawl pool size; anything that is not a positive int falls back to the default."""
    value = ctx.params.get("ema_smpc_concurrency")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return _DEFAULT_CONCURRENCY


def _max_age_days(ctx: TaskContext) -> float | None:
    """Resolve the per-document age fallback; ``None``/non-positive disables the age gate."""
    value = ctx.params.get("ema_smpc_max_age_days")
    if value is None:
        return _DEFAULT_MAX_AGE_DAYS
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None
    return _DEFAULT_MAX_AGE_DAYS


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _age_days(retrieved_at: datetime | None) -> float | None:
    if retrieved_at is None:
        return None
    return (datetime.now(UTC) - retrieved_at).total_seconds() / 86400.0


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


fetch = EmaSmpcFetcher().fetch

__all__ = ["SMPD_FANOUT_ALIAS", "EmaSmpcFetcher", "crawl_fingerprint", "download_ema_smpc_pdf", "fetch", "member_alias"]
