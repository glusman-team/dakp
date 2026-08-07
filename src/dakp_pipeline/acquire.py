"""Acquisition orchestration — shared download-to-store logic for the DAG + the test harness.

Thin, decoupled helpers that wrap the source fetchers (:mod:`dakp_pipeline.sources`) and the
NER model cache (:mod:`dakp_pipeline.ner.model_cache`) so both the Airflow acquisition tasks
(:mod:`dakp_pipeline.dags.dakp_build`) and the pure-Python runner can call one function per
source. Every helper takes a :class:`~dakp_pipeline.io.contracts.TaskContext` and returns
:class:`~dakp_pipeline.io.contracts.ArtifactRef` manifests (paths + BLAKE3 ids) — never
dataframes.

Idempotent and non-destructive: raw downloads land in the BLAKE3 content-addressed store
(re-ingesting identical bytes is a cache hit) and NER weights land in the model cache
(re-used by tree hash). Nothing is renamed or deleted except per-run staging files. Offline
tests monkeypatch the source fetchers' module-level ``fetch`` and inject a fake NER
``downloader``. The fullmap redb is NOT acquired here — it is an external artifact the caller
supplies (the CLI ``--fullmap`` path, threaded into ``ctx.params["fullmap"]`` for the Tablassert
handoff); DAKP never downloads a fullmap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.ner import model_cache
from dakp_pipeline.ner.ner import DEFAULT_MODEL
from dakp_pipeline.sources import dailymed, drugsfda, faers

#: Media type for a cached NER model directory (a tree artifact, not a single file).
_MODEL_DIR_MEDIA_TYPE = "application/x-directory"

#: Max concurrent source downloads (sizes the thread pool / the Airflow download pool).
_DOWNLOAD_CONCURRENCY = 4


# --- source acquisition (delegate to the real fetchers) -------------------------


def acquire_dailymed(ctx: TaskContext) -> list[ArtifactRef]:
    """Acquire DailyMed SPL full-release artifacts over the network."""
    return dailymed.fetch(ctx)


def acquire_faers(ctx: TaskContext) -> list[ArtifactRef]:
    """Acquire FAERS quarterly ASCII artifacts over the network."""
    return faers.fetch(ctx)


def acquire_drugsfda(ctx: TaskContext) -> list[ArtifactRef]:
    """Acquire the Drugs@FDA data-files ZIP over the network."""
    return drugsfda.fetch(ctx)


# --- NER model acquisition ------------------------------------------------------


def default_ner_models(ctx: TaskContext) -> list[str]:
    """NER model ids to cache for ``ctx``: the default GLiNER checkpoint the NER backend loads.

    A real run downloads the GLiNER weights; offline tests pass an explicit ``models=`` list (and
    a fake ``downloader``) to :func:`acquire_ner_models` instead of relying on this default.
    """
    del ctx  # the default model set is fixed; the context is kept for a stable call signature
    return [DEFAULT_MODEL]


def model_ref_to_artifact(ref: model_cache.ModelRef) -> ArtifactRef:
    """Project a cached-model handle into an :class:`ArtifactRef` (path + tree hash + manifest)."""
    return ArtifactRef(uri=ref.path, blake3=ref.b3, media_type=_MODEL_DIR_MEDIA_TYPE, manifest=ref.manifest)


def acquire_ner_models(
    ctx: TaskContext, *, models: Iterable[str] | None = None, downloader: model_cache.Downloader | None = None, cache_dir: Path | str | None = None
) -> list[ArtifactRef]:
    """Download/cache NER model weights via :func:`model_cache.ensure_model`; return manifests.

    Idempotent: a cached model whose content tree hash still matches is a hit (no download).
    ``force`` (from ``ctx.params``) re-downloads unconditionally. ``models`` overrides the default
    (:func:`default_ner_models`); ``downloader`` is injectable for offline tests (defaults to the
    Hugging Face Hub downloader, which needs the NER dependencies).
    """
    with step(logger, "acquire_ner_models"):
        model_ids = list(models) if models is not None else default_ner_models(ctx)
        force = bool(ctx.params.get("force", False))
        stats(logger, "acquire_ner_models", models=model_ids, force=force)
        resolved_cache = Path(cache_dir) if cache_dir is not None else model_cache.default_model_cache_dir(ctx.workdir)
        stats(logger, "acquire_ner_models", cache_dir=str(resolved_cache))
        refs: list[ArtifactRef] = []
        for model_id in model_ids:
            cached = model_cache.ensure_model(model_id, cache_dir=resolved_cache, downloader=downloader, force=force)
            refs.append(model_ref_to_artifact(cached))
            stats(logger, "acquire_ner_models", model_id=model_id, artifact_id=cached.b3, source=cached.source)
        stats(logger, "acquire_ner_models", models_cached=len(refs))
        return refs


# --- aggregate acquisition ------------------------------------------------------


def acquire_all(ctx: TaskContext, *, downloader: model_cache.Downloader | None = None) -> dict[str, list[ArtifactRef]]:
    """Run every acquisition, bounded by :data:`_DOWNLOAD_CONCURRENCY`; return keyed manifests.

    The four acquisitions are independent and content-addressed (order-independent hashes), so
    running them on a bounded thread pool is deterministic. The ``downloader`` is forwarded to
    the NER-model acquisition (the source fetchers own their own monkeypatchable network
    boundaries). Useful as the single acquisition entry point for the DAG + test harness.
    """
    jobs: dict[str, Callable[[], list[ArtifactRef]]] = {
        "dailymed": lambda: acquire_dailymed(ctx),
        "drugsfda": lambda: acquire_drugsfda(ctx),
        "faers": lambda: acquire_faers(ctx),
        "ner_models": lambda: acquire_ner_models(ctx, downloader=downloader),
    }
    results: dict[str, list[ArtifactRef]] = {}
    with ThreadPoolExecutor(max_workers=min(_DOWNLOAD_CONCURRENCY, len(jobs))) as pool:
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
