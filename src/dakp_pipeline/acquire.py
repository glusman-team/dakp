"""Acquisition orchestration — shared download-to-store logic for the DAG + ``run_pipeline``.

Thin, decoupled helpers that wrap the source fetchers (:mod:`dakp_pipeline.sources`) and the
NER model cache (:mod:`dakp_pipeline.ner.model_cache`) so both the Airflow acquisition tasks
(:mod:`dakp_pipeline.dags.dakp_build`) and the pure-Python runner can call one function per
source. Every helper takes a :class:`~dakp_pipeline.io.contracts.TaskContext` and returns
:class:`~dakp_pipeline.io.contracts.ArtifactRef` manifests (paths + BLAKE3 ids) — never
dataframes.

Idempotent and non-destructive: raw downloads land in the BLAKE3 content-addressed store
(re-ingesting identical bytes is a cache hit) and NER weights land in the model cache
(re-used by tree hash). Nothing is renamed or deleted except per-run staging files.

The mock profile never touches the network: the source fetchers ingest fixtures, the NER
model acquisition is a no-op (the deterministic dictionary/mock backends need no weights),
and ontology acquisition ingests the bundled ontology fixture. Real profiles download; the
fullmap redb source is config-driven (:attr:`DownloadConfig.fullmap_source`) and defaults to
a stub URL until a canonical source is published.
"""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dakp_pipeline.config import load_profile
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.ner import model_cache
from dakp_pipeline.ner.ner import DEFAULT_MODEL
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, drugsfda, faers, ingest_fixtures

#: Documented stub source for the fullmap redb. No canonical public fullmap is published yet,
#: so the ontology acquisition task is real/structured but points at an ``example.invalid``
#: URL (PLAN.md source-url convention) until :attr:`DownloadConfig.fullmap_source` overrides
#: it. The mock profile never reaches this (it ingests the ontology fixture instead).
DEFAULT_FULLMAP_SOURCE = "https://example.invalid/dakp/fullmap.redb"

#: Media type for a cached NER model directory (a tree artifact, not a single file).
_MODEL_DIR_MEDIA_TYPE = "application/x-directory"

#: Ontology fixtures ingested by the mock profile, relative to ``fixture_root``.
_ONTOLOGY_FIXTURE_GLOB = "ontology/*.tsv"

_DOWNLOAD_TIMEOUT = 120.0
_CHUNK = 1 << 20  # 1 MiB streaming window.

# A downloader writes the resource at ``url`` to ``dest``. Dependency-injected so tests serve
# local bytes with no network (mirrors ner.model_cache.Downloader's ``(id, dest)`` shape).
Downloader = Callable[[str, Path], None]


# --- source acquisition (delegate to the real fetchers) -------------------------


def acquire_dailymed(ctx: TaskContext) -> list[ArtifactRef]:
    """Acquire DailyMed SPL full-release artifacts (fixtures in mock, network otherwise)."""
    return dailymed.fetch(ctx)


def acquire_faers(ctx: TaskContext) -> list[ArtifactRef]:
    """Acquire FAERS quarterly ASCII artifacts (fixtures in mock, network otherwise)."""
    return faers.fetch(ctx)


def acquire_drugsfda(ctx: TaskContext) -> list[ArtifactRef]:
    """Acquire the Drugs@FDA data-files ZIP (fixtures in mock, network otherwise)."""
    return drugsfda.fetch(ctx)


# --- NER model acquisition ------------------------------------------------------


def default_ner_models(ctx: TaskContext) -> list[str]:
    """NER model ids to cache for ``ctx`` (mock = none; else config override or backend default).

    The mock profile uses deterministic dictionary/mock backends that need no weights, so it
    acquires nothing. Other profiles use :attr:`DownloadConfig.ner_model_ids` when set, else
    the default GLiNER checkpoint the real NER backend loads.
    """
    if ctx.profile == "mock":
        return []
    configured = list(load_profile(ctx.profile).download.ner_model_ids)
    return configured or [DEFAULT_MODEL]


def model_ref_to_artifact(ref: model_cache.ModelRef) -> ArtifactRef:
    """Project a cached-model handle into an :class:`ArtifactRef` (path + tree hash + manifest)."""
    return ArtifactRef(uri=ref.path, blake3=ref.b3, media_type=_MODEL_DIR_MEDIA_TYPE, manifest=ref.manifest)


def acquire_ner_models(
    ctx: TaskContext, *, models: Iterable[str] | None = None, downloader: model_cache.Downloader | None = None, cache_dir: Path | str | None = None
) -> list[ArtifactRef]:
    """Download/cache NER model weights via :func:`model_cache.ensure_model`; return manifests.

    Idempotent: a cached model whose content tree hash still matches is a hit (no download).
    ``force`` (from ``ctx.params``) re-downloads unconditionally. ``models`` overrides the
    profile-derived default (:func:`default_ner_models`); ``downloader`` is injectable for
    offline tests (defaults to the Hugging Face Hub downloader, which needs the ``[ner]``
    extra). The mock profile with no explicit ``models`` acquires nothing.
    """
    log = bind(task_id="acquire_ner_models", profile=ctx.profile)
    model_ids = list(models) if models is not None else default_ner_models(ctx)
    if not model_ids:
        log.info("no NER models to acquire (deterministic backend; no weights needed)")
        return []
    force = bool(ctx.params.get("force", False))
    resolved_cache = Path(cache_dir) if cache_dir is not None else model_cache.default_model_cache_dir(ctx.workdir)
    refs: list[ArtifactRef] = []
    for model_id in model_ids:
        cached = model_cache.ensure_model(model_id, cache_dir=resolved_cache, downloader=downloader, force=force)
        refs.append(model_ref_to_artifact(cached))
        log.info("ner model cached", model_id=model_id, artifact_id=cached.b3, source=cached.source)
    return refs


# --- ontology / fullmap acquisition ---------------------------------------------


def default_ontology_sources(ctx: TaskContext) -> dict[str, str]:
    """Ontology/fullmap sources for ``ctx``: config extras plus the (stubbed) fullmap redb.

    Merges :attr:`DownloadConfig.ontology_sources` with the fullmap redb source
    (:attr:`DownloadConfig.fullmap_source`, defaulting to :data:`DEFAULT_FULLMAP_SOURCE`) under
    the ``fullmap.redb`` name. The mock profile does not call this (it ingests fixtures).
    """
    download = load_profile(ctx.profile).download
    sources = dict(download.ontology_sources)
    sources.setdefault("fullmap.redb", download.fullmap_source or DEFAULT_FULLMAP_SOURCE)
    return sources


def acquire_ontologies(ctx: TaskContext, *, sources: Mapping[str, str] | None = None, downloader: Downloader | None = None) -> list[ArtifactRef]:
    """Acquire ontology/fullmap resolution resources into the content-addressed store.

    * mock profile: ingest the bundled ontology fixture(s) under ``fixture_root/ontology``
      (no network); a mock context with no ``fixture_root`` yields ``[]``.
    * other profiles: download each ``name -> url/path`` source (``sources`` overrides
      :func:`default_ontology_sources`) into the store with URL provenance. The fullmap redb
      source is config-driven and stubbed until a canonical source is published, so point
      :attr:`DownloadConfig.fullmap_source` at a real source for production.

    Idempotent by BLAKE3; only per-run staging files are removed. ``downloader`` is injectable
    for offline tests (defaults to stdlib :mod:`urllib`, which also handles ``file://`` URLs).
    """
    log = bind(task_id="acquire_ontologies", profile=ctx.profile)
    if ctx.profile == "mock":
        refs = _ingest_ontology_fixtures(ctx)
        log.info("ontology fixtures ingested", artifacts=len(refs))
        return refs

    resolved = dict(sources) if sources is not None else default_ontology_sources(ctx)
    if not resolved:
        log.info("no ontology/fullmap sources configured; nothing to acquire")
        return []

    workdir = Workdir(ctx.workdir)
    store = ArtifactStore(workdir)
    staging = workdir.root / ".staging" / "ontologies"
    staging.mkdir(parents=True, exist_ok=True)
    refs = []
    for name, url in sorted(resolved.items()):
        dest = staging / name
        _download_to(url, dest, downloader)
        ref, cache_hit = store.ingest(dest, alias=f"ontology/{name}", source=SourceBlock(url=url, retrieved_at=_now_iso()))
        refs.append(ref)
        dest.unlink(missing_ok=True)
        log.info("ontology acquired", name=name, artifact_id=ref.blake3, cache_hit=cache_hit)
    return refs


def _ingest_ontology_fixtures(ctx: TaskContext) -> list[ArtifactRef]:
    """Content-address the bundled ontology fixture(s) under ``fixture_root/ontology``."""
    if ctx.fixture_root is None:
        return []
    names = tuple(sorted(path.relative_to(ctx.fixture_root).as_posix() for path in ctx.fixture_root.glob(_ONTOLOGY_FIXTURE_GLOB)))
    if not names:
        return []
    return ingest_fixtures(ctx, names, namespace="ontology")


def _download_to(url: str, dest: Path, downloader: Downloader | None) -> None:
    """Write ``url`` to ``dest`` via the injected downloader or stdlib :mod:`urllib`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if downloader is not None:
        downloader(url, dest)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "dakp-pipeline/0.1"})
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out, length=_CHUNK)


# --- aggregate acquisition ------------------------------------------------------


def acquire_all(ctx: TaskContext, *, downloader: Downloader | None = None) -> dict[str, list[ArtifactRef]]:
    """Run every acquisition, bounded by :attr:`DownloadConfig.concurrency`; return keyed manifests.

    The five acquisitions are independent and content-addressed (order-independent hashes), so
    running them on a bounded thread pool is deterministic. The ``downloader`` is forwarded to
    the NER-model and ontology acquisitions (the source fetchers own their own monkeypatchable
    network boundaries). Useful as the single acquisition entry point for ``run_pipeline``.
    """
    concurrency = max(1, load_profile(ctx.profile).download.concurrency)
    jobs: dict[str, Callable[[], list[ArtifactRef]]] = {
        "dailymed": lambda: acquire_dailymed(ctx),
        "drugsfda": lambda: acquire_drugsfda(ctx),
        "faers": lambda: acquire_faers(ctx),
        "ner_models": lambda: acquire_ner_models(ctx, downloader=downloader),
        "ontologies": lambda: acquire_ontologies(ctx, downloader=downloader),
    }
    results: dict[str, list[ArtifactRef]] = {}
    with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        for name, future in futures.items():
            results[name] = future.result()
    return results


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DEFAULT_FULLMAP_SOURCE",
    "Downloader",
    "acquire_all",
    "acquire_dailymed",
    "acquire_drugsfda",
    "acquire_faers",
    "acquire_ner_models",
    "acquire_ontologies",
    "default_ner_models",
    "default_ontology_sources",
    "model_ref_to_artifact",
]
