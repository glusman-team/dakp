"""Shared NER dispatch plumbing for the assertion shapers.

Every shaper that mines DailyMed/FAERS text with the composite NER backend
(:class:`~dakp_pipeline.ner.ner.DiseaseNER`) needs the same three things:

* **backend construction** — :func:`default_ner` builds the deterministic **offline**
  backend (gazetteer from the ontology fixture, else embedded) used by tests and offline
  runs; production shapers receive an injected ``params["ner"]`` instead.
* **device resolution** — :func:`_resolve_devices` caps the hardcoded build-host GPU list
  to the visible, torch-supported CUDA devices (None when unusable → sequential CPU mining).
* **multi-GPU dispatch** — :func:`_mine_multi_gpu` / :func:`mine_passes_multi_gpu` shard
  work items across one spawned worker per GPU (LPT-balanced by text length), with
  byte-identical output regardless of dispatch mode.
* **persistent caching** — :func:`mine_with_cache` fronts a shaper's mining path with the
  Pebble-backed mention cache (:mod:`~dakp_pipeline.ner.mention_cache`), so repeated DAG
  runs re-mine only previously-unseen texts.

Work items are tuple-like ``(set_id, doc_id, text)`` triples: plain tuples or any object
supporting integer indexing (e.g. the contraindication shaper's ``ContraWorkItem``, whose
extra evidence mapping stays invisible to this layer). :mod:`~dakp_pipeline.assertions.contraindications`
re-exports the underscore names for its historical test surface.
"""

from __future__ import annotations

import importlib.machinery
import multiprocessing as mp
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.ner.mention_cache import MentionCache, mention_key, ner_cache_material
from dakp_pipeline.ner.ner import DiseaseNER, Mention, _cuda_device_supported

#: The 4x Tesla P100-PCIE-16GB GPUs on the DAKP build host (wenceslaus). Hardcoded - not
#: auto-detected — so shapers always dispatch across all four when CUDA is available.
#: When CUDA is absent (CI, tests, non-GPU hosts) shapers fall back to sequential.
BUILD_HOST_GPUS: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")

_ONTOLOGY_FIXTURE = Path("ontology") / "disease_map.tsv"


def default_ner(fixture_root: Path | str | None) -> DiseaseNER:
    """The deterministic offline NER backend: gazetteer from the ontology fixture, else embedded.

    Reads the ontology fixture as a term→type gazetteer ONLY (``text`` + ``category`` columns;
    CURIE/name columns are ignored — DAKP does not map terms to ontology concepts). No heavy
    NER dep is imported.
    """
    if fixture_root is not None:
        ontology = Path(fixture_root) / _ONTOLOGY_FIXTURE
        if ontology.exists():
            return DiseaseNER.from_tsv(ontology, text_col="text", type_col="category")
    return DiseaseNER()


def _resolve_devices(ner: DiseaseNER, gpus: Sequence[str] = BUILD_HOST_GPUS) -> Sequence[str] | None:
    """The GPU list capped to the VISIBLE device count and filtered to devices the
    installed torch can actually run on; None when unusable.

    Only the production (GLiNER) backend benefits from multi-GPU dispatch — the offline
    gazetteer is CPU-only and deterministic. ``torch.cuda.is_available()`` guards against
    CI / test hosts with no CUDA (the lazy import never fires at module load). The list is
    capped at ``torch.cuda.device_count()`` because hosts vary: the build server has the full
    4x P100 set, laptops often expose a single GPU — dispatching a worker to a nonexistent
    ``cuda:N`` crashes the whole pool (torch refuses to deserialize onto a missing device).
    Devices whose arch the torch build lacks kernels for (e.g. sm_60 P100s against a cu128
    wheel line) are filtered out via :func:`~dakp_pipeline.ner.ner._cuda_device_supported` —
    ``is_available()`` alone lies there (True, but the first CUDA call raises), so with no
    supported device the shaper falls back to sequential CPU mining with a warning.
    """
    if ner._offline:
        return None
    try:
        import torch  # lazy: no torch at module load
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    visible = torch.cuda.device_count()
    if visible <= 0:
        return None
    supported = tuple(gpus[index] for index in range(min(visible, len(gpus))) if _cuda_device_supported(torch, index))
    if not supported:
        logger.warning(
            "contraindication_gpus_unsupported: no visible CUDA device arch is in the torch build arch list = {}; falling back to sequential CPU mining",
            torch.cuda.get_arch_list(),
        )
        return None
    return supported


def _item_parts(item: Any) -> tuple[str, str, str]:
    """Read ``(set_id, doc_id, text)`` from a tuple or tuple-like work item."""
    return item[0], item[1], item[2]


def _shard_by_text_length(items: Sequence[Any], n: int) -> list[list[Any]]:
    """Distribute work items across ``n`` shards, balanced by text length (LPT scheduling).

    Sorts items by text length descending, then greedily assigns each to the shard with the
    least total text so far. Returns exactly ``n`` shards (some may be empty when ``n > len(items)``).
    """
    shards: list[list[Any]] = [[] for _ in range(n)]
    loads: list[int] = [0] * n
    for item in sorted(items, key=lambda x: len(_item_parts(x)[2]), reverse=True):
        idx = loads.index(min(loads))
        shards[idx].append(item)
        loads[idx] += len(_item_parts(item)[2])
    return shards


def _mine_shard(shard: Sequence[Any], ner_config: dict[str, Any], device: str) -> list[tuple[str, str, list[Mention]]]:
    """ProcessPoolExecutor worker: load GLiNER on ``device``, mine each text, return mentions.

    Reconstructs a :class:`DiseaseNER` from the picklable ``ner_config`` pinned to ``device``,
    then runs extraction over every ``(set_id, doc_id, text)`` item in its shard. The model
    loads lazily on the first extract call, so each worker initializes its own CUDA context
    (safe under the ``spawn`` start method).
    """
    ner = DiseaseNER(device=device, **ner_config)
    return [(set_id, doc_id, ner.extract(text)) for set_id, doc_id, text in (_item_parts(item) for item in shard)]


def _mine_multi_gpu(work_items: Sequence[Any], ner: DiseaseNER, devices: Sequence[str]) -> dict[tuple[str, str], list[Mention]]:
    """Dispatch NER extraction across one worker per GPU and collect results.

    Shards ``work_items`` across ``len(devices)`` groups (LPT-balanced by text length), spawns
    one process per device via :class:`~concurrent.futures.ProcessPoolExecutor` (``spawn``
    start method — CUDA + ``fork`` is unsafe), and returns a ``{(set_id, doc_id): [mentions]}``
    map. The model cache on disk is shared read-only across workers.
    """
    n_workers = min(len(devices), len(work_items))
    shards = _shard_by_text_length(work_items, n_workers)
    ner_config = ner._config()
    ctx = mp.get_context("spawn")
    results: dict[tuple[str, str], list[Mention]] = {}
    with _spawn_safe_main(), ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_mine_shard, shard, ner_config, devices[i]) for i, shard in enumerate(shards)]
        for future in futures:
            for set_id, doc_id, mentions in future.result():
                results[(set_id, doc_id)] = mentions
    return results


def _group_devices(devices: Sequence[str], k: int) -> list[list[str]]:
    """Split ``devices`` into ``k`` contiguous, near-even groups (defensively never empty)."""
    base, extra = divmod(len(devices), k)
    groups: list[list[str]] = []
    start = 0
    for index in range(k):
        size = base + (1 if index < extra else 0)
        groups.append(list(devices[start : start + size]) or list(devices[:1]))
        start += size
    return groups


def mine_passes_multi_gpu(passes: Sequence[Sequence[Any]], ner: DiseaseNER, devices: Sequence[str]) -> dict[tuple[str, str], list[Mention]]:
    """Dispatch several extraction passes concurrently, splitting GPUs between them.

    Generalizes the contraindication two-pass split to any number of passes: empty passes are
    dropped, the device list is divided into contiguous near-even groups (one per remaining
    pass, LPT-sharded by text length), and all passes are dispatched as futures in a single
    :class:`~concurrent.futures.ProcessPoolExecutor` so they run in parallel. With zero or one
    nonempty pass all GPUs fall back to :func:`_mine_multi_gpu`. Results merge into one
    ``{(set_id, doc_id): [mentions]}`` map; output is byte-identical regardless of dispatch.
    """
    nonempty = [list(items) for items in passes if items]
    if not nonempty:
        return {}
    if len(nonempty) == 1:
        return _mine_multi_gpu(nonempty[0], ner, devices)

    groups = _group_devices(list(devices), len(nonempty))
    ner_config = ner._config()
    ctx = mp.get_context("spawn")
    n_workers = sum(min(len(group), len(items)) for items, group in zip(nonempty, groups, strict=True))

    results: dict[tuple[str, str], list[Mention]] = {}
    with _spawn_safe_main(), ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = []
        for items, group in zip(nonempty, groups, strict=True):
            shards = _shard_by_text_length(items, min(len(group), len(items)))
            futures.extend(pool.submit(_mine_shard, shard, ner_config, group[index]) for index, shard in enumerate(shards))
        for future in futures:
            for set_id, doc_id, mentions in future.result():
                results[(set_id, doc_id)] = mentions
    return results


def _mine_two_passes_multi_gpu(
    work_items_p1: Sequence[Any], work_items_p2: Sequence[Any], ner: DiseaseNER, devices: Sequence[str]
) -> dict[tuple[str, str], list[Mention]]:
    """Dispatch two extraction passes concurrently, splitting GPUs between them.

    Thin wrapper over :func:`mine_passes_multi_gpu` — the contraindication shaper's
    (contraindication sections, filtered indication sections) mining shape. With either pass
    empty, all GPUs fall back to the other pass via :func:`_mine_multi_gpu`.
    """
    return mine_passes_multi_gpu([work_items_p1, work_items_p2], ner, devices)


#: A shaper's existing mining path (multi-GPU dispatch or sequential loop) over the given
#: items, returning ``{(set_id, doc_id): [mentions]}``.
MineFn = Callable[[Sequence[Any]], dict[tuple[str, str], list[Mention]]]


def mine_with_cache(work_items: Sequence[Any], ner: DiseaseNER, mine: MineFn, cache: MentionCache | None) -> dict[tuple[str, str], list[Mention]]:
    """Run ``mine`` over ``work_items``, serving repeats from the persistent mention cache.

    Central caching seam for all three NER shapers. Text-level flow: every item's cache key
    (:func:`~dakp_pipeline.ner.mention_cache.mention_key` over model id + model content b3 +
    config fingerprint + normalized text) is batch-fetched up front; only MISSES reach
    ``mine`` (one representative item per distinct missing key, so duplicate texts are mined
    once), and freshly mined results are batch-put back before the hit+miss merge. The
    returned ``{(set_id, doc_id): [mentions]}`` map is byte-identical to a no-cache run —
    hits round-trip :meth:`Mention.to_dict`/:meth:`Mention.from_dict` losslessly and the
    server stores the value bytes verbatim.

    Cache access happens ONLY in this parent process: spawned GPU workers
    (:func:`_mine_shard`) receive no cache handle, which keeps the Pebble store
    single-owner and the worker code untouched. Pass-through (``mine`` over everything)
    when ``cache`` is None, when the backend is offline (the gazetteer is deterministic and
    CPU-cheap — deliberately not cached), or when the cache server is unavailable
    (:class:`~dakp_pipeline.ner.mention_cache.MentionCache` degrades to a no-op).
    """
    if cache is None:
        return mine(work_items)
    material = ner_cache_material(ner)
    if material is None:
        return mine(work_items)
    model_id, model_b3, fingerprint = material

    key_by_item = {_item_parts(item)[:2]: mention_key(model_id, model_b3, fingerprint, _item_parts(item)[2]) for item in work_items}
    hits = cache.get_many(sorted(set(key_by_item.values())))

    representatives: dict[str, Any] = {}  # missing key -> one item carrying that text
    for item in work_items:
        key = key_by_item[_item_parts(item)[:2]]
        if key not in hits and key not in representatives:
            representatives[key] = item
    mined: dict[str, list[Mention]] = {}
    if representatives:
        results = mine(list(representatives.values()))
        mined = {key: results.get(_item_parts(item)[:2], []) for key, item in representatives.items()}
        cache.put_many(mined)
    stats(logger, "ner_mention_cache", items=len(work_items), hits=len(work_items) - len(representatives), mined=len(representatives))

    out: dict[tuple[str, str], list[Mention]] = {}
    for item in work_items:
        set_id, doc_id, _text = _item_parts(item)
        key = key_by_item[(set_id, doc_id)]
        out[(set_id, doc_id)] = hits[key] if key in hits else mined.get(key, [])
    return out


#: Module spawned workers re-import instead of the parent's ``__main__`` script (see
#: :func:`_spawn_safe_main`). Must import with zero side effects. A plain MODULE, not the
#: package itself — ``runpy.run_module`` cannot directly execute a package without a
#: ``__main__.py``.
_SPAWN_SAFE_MAIN_MODULE = "dakp_pipeline.logging_setup"


@contextmanager
def _spawn_safe_main() -> Iterator[None]:
    """Keep spawn children from re-executing the parent's ``__main__`` script.

    The spawn start method re-initializes ``__main__`` in every child: when
    ``__main__.__spec__`` exists it imports that module by name, otherwise it RE-EXECUTES
    ``__main__.__file__``. Under the Airflow task runtime ``__main__`` is the ``airflow`` CLI
    script (no ``__spec__``), so each mining worker re-executed the CLI and died initializing
    Airflow's DB settings there (no parseable ``sql_alchemy_conn`` in the child) — the pool
    broke before its first task (``BrokenProcessPool``). For the pool's lifetime, point spawn
    at a side-effect-free module; the worker callable itself is unpickled via its own
    module, and the original ``__main__`` spec is restored on exit.
    """
    main = sys.modules["__main__"]
    if getattr(main, "__spec__", None) is not None:
        yield  # module context (python -m / console scripts with a spec): spawn imports by name
        return
    main.__spec__ = importlib.machinery.ModuleSpec(_SPAWN_SAFE_MAIN_MODULE, loader=None)
    try:
        yield
    finally:
        main.__spec__ = None


__all__ = ["BUILD_HOST_GPUS", "MineFn", "default_ner", "mine_passes_multi_gpu", "mine_with_cache"]
