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

The mock profile never touches the network: the source fetchers ingest fixtures and the NER
model acquisition is a no-op (the deterministic offline NER backend needs no weights). The
fullmap redb is NOT acquired here — it is an external artifact the caller supplies (the CLI
``--fullmap`` path, threaded into ``ctx.params["fullmap"]`` for the Tablassert handoff); DAKP
never downloads a fullmap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dakp_pipeline.config import load_profile
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.ner import model_cache
from dakp_pipeline.ner.ner import DEFAULT_MODEL
from dakp_pipeline.sources import dailymed, drugsfda, faers

#: Media type for a cached NER model directory (a tree artifact, not a single file).
_MODEL_DIR_MEDIA_TYPE = "application/x-directory"


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

    The mock profile uses the deterministic offline NER backend, which needs no weights, so it
    acquires nothing. Other profiles use :attr:`DownloadConfig.ner_model_ids` when set, else
    the default GLiNER checkpoint the production NER backend loads.
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
    offline tests (defaults to the Hugging Face Hub downloader, which needs the NER
    dependencies). The mock profile with no explicit ``models`` acquires nothing.
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


# --- aggregate acquisition ------------------------------------------------------


def acquire_all(ctx: TaskContext, *, downloader: model_cache.Downloader | None = None) -> dict[str, list[ArtifactRef]]:
    """Run every acquisition, bounded by :attr:`DownloadConfig.concurrency`; return keyed manifests.

    The four acquisitions are independent and content-addressed (order-independent hashes), so
    running them on a bounded thread pool is deterministic. The ``downloader`` is forwarded to
    the NER-model acquisition (the source fetchers own their own monkeypatchable network
    boundaries). Useful as the single acquisition entry point for ``run_pipeline``.
    """
    concurrency = max(1, load_profile(ctx.profile).download.concurrency)
    jobs: dict[str, Callable[[], list[ArtifactRef]]] = {
        "dailymed": lambda: acquire_dailymed(ctx),
        "drugsfda": lambda: acquire_drugsfda(ctx),
        "faers": lambda: acquire_faers(ctx),
        "ner_models": lambda: acquire_ner_models(ctx, downloader=downloader),
    }
    results: dict[str, list[ArtifactRef]] = {}
    with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        for name, future in futures.items():
            results[name] = future.result()
    return results


__all__ = [
    "acquire_all",
    "acquire_dailymed",
    "acquire_drugsfda",
    "acquire_faers",
    "acquire_ner_models",
    "default_ner_models",
    "model_ref_to_artifact",
]
