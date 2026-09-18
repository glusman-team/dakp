"""Shared NER dispatch plumbing for the assertion shapers.

Every shaper that mines DailyMed text with the composite NER backend
(:class:`~dakp_pipeline.ner.ner.DiseaseNER`) needs the same three things:

* **backend construction** — :func:`default_ner` builds the deterministic **offline**
  backend (gazetteer from the ontology fixture, else embedded) used by tests and offline
  runs; production shapers receive an injected ``params["ner"]`` instead.
* **device resolution** — :func:`_resolve_devices` discovers every visible CUDA ordinal and
  filters it to torch-supported devices (None when unusable → sequential CPU mining).
* **multi-GPU dispatch** — :func:`_mine_multi_gpu` shards work items across one spawned worker
  per GPU (LPT-balanced by text length), with byte-identical output regardless of dispatch mode.
  The shaper's three mining passes share ONE backend profile, so their work items are
  dispatched as a single globally-balanced pool; a per-pass device split would pin the
  dominant warnings pass to one GPU while the other GPUs idle.
* **persistent caching** — :func:`mine_with_cache` fronts a shaper's mining path with the
  Pebble-backed mention cache (:mod:`~dakp_pipeline.ner.mention_cache`), so repeated DAG
  runs re-mine only previously-unseen texts.

Work items are tuple-like ``(set_id, doc_id, text)`` triples: plain tuples or any object
supporting integer indexing (e.g. the contraindication shaper's ``ContraWorkItem``, whose
extra evidence mapping stays invisible to this layer). :mod:`~dakp_pipeline.assertions.contraindications`
re-exports the underscore names for its historical test surface.
"""

from __future__ import annotations

import ctypes
import importlib.machinery
import multiprocessing as mp
import os
import signal
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dakp_pipeline.logging_setup import WORKER_LOG_SUBDIR, configure_worker_logging, logger, prune_worker_logs, stats
from dakp_pipeline.ner.mention_cache import MentionCache, mention_key, ner_cache_material
from dakp_pipeline.ner.ner import DiseaseNER, Mention, _cuda_device_supported

#: Historical build-host default retained for compatibility with callers that import it. Runtime
#: dispatch no longer uses this fixed list: :func:`_resolve_devices` discovers every visible CUDA
#: ordinal and filters it by the installed torch kernels.
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


def _resolve_devices(ner: DiseaseNER, gpus: Sequence[str] | None = None) -> Sequence[str] | None:
    """Discover visible CUDA devices and filter to devices the installed torch can run on.

    Only the production (GLiNER) backend benefits from multi-GPU dispatch — the offline
    gazetteer is CPU-only and deterministic. ``torch.cuda.is_available()`` guards against
    CI / test hosts with no CUDA (the lazy import never fires at module load). Device ordinals
    are taken directly from ``torch.cuda.device_count()``; this respects ``CUDA_VISIBLE_DEVICES``
    and avoids dispatching a worker to a nonexistent ``cuda:N``. An optional ``gpus`` sequence
    remains for tests and legacy callers that need an explicit subset.
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
    candidates = tuple(gpus) if gpus is not None else tuple(f"cuda:{index}" for index in range(visible))
    supported = tuple(candidates[index] for index in range(min(visible, len(candidates))) if _cuda_device_supported(torch, index))
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


def _announce_worker_logs(workdir: Any) -> None:
    """Name the worker-log directory in the TASK log, then prune stale files from it.

    Spawned workers write their narration to their own files
    (:func:`~dakp_pipeline.logging_setup.configure_worker_logging`), so without this line the
    Airflow task log would give no hint that those diagnostics exist or where to read them.
    Pruning runs here, in the single parent, before any worker opens a file.
    """
    if workdir is None:
        return
    stats(logger, "ner_worker_logs", path=str(Path(workdir) / WORKER_LOG_SUBDIR), pruned=prune_worker_logs(workdir))


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


def _set_parent_death_signal() -> None:
    """Linux-only: ask the kernel for ``SIGKILL`` the moment the worker's parent process dies.

    Spawned mining workers hold a GPU model plus the device flock for their whole life. When
    the Airflow task process dies (kill, crash, OOM) nothing reaps them: they stay resident,
    keep the flock, and every retried task then deadlocks behind the orphaned lock holders
    (observed 20+ h with zero GPU compute). ``PR_SET_PDEATHSIG`` closes that leak. Off-Linux
    or without ``prctl`` this is a no-op (the flock timeout in
    :func:`~dakp_pipeline.ner.ner._acquire_gpu_lock` is the remaining safety net). The
    ``getppid`` re-check closes the set-after-death race: if the parent already died before
    the call, the kernel delivers nothing, so an orphaned-at-spawn worker exits itself.
    """
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG = 1
            logger.warning("ner_worker_pdeathsig: prctl failed, errno = {}", ctypes.get_errno())
            return
    except (AttributeError, OSError):
        logger.warning("ner_worker_pdeathsig: prctl unavailable; orphan reaping disabled for this worker")
        return
    if os.getppid() == 1:  # parent already dead before the signal was armed
        logger.error("ner_worker_pdeathsig: parent already dead at spawn; exiting to release GPU + flock")
        os._exit(1)


def _mine_shard(shard: Sequence[Any], ner_config: dict[str, Any], device: str, *, in_worker: bool = False) -> list[tuple[str, str, list[Mention]]]:
    """ProcessPoolExecutor worker: load GLiNER on ``device``, mine each text, return mentions.

    Reconstructs a :class:`DiseaseNER` from the picklable ``ner_config`` pinned to ``device``,
    then runs extraction over every ``(set_id, doc_id, text)`` item in its shard. The model
    loads lazily on the first extract call, so each worker initializes its own CUDA context
    (safe under the ``spawn`` start method).

    ``in_worker`` is passed ONLY by the ``pool.submit`` call site, so the process-global
    logging reconfiguration below can never fire in the parent. It is an explicit flag rather
    than a runtime probe because :func:`multiprocessing.parent_process` does not discriminate:
    it reports the Airflow task process itself as a child (LocalExecutor runs tasks under a
    ``multiprocessing.Process``, and the supervisor forks the task), so an in-process call
    would have wiped the TASK's loguru sinks and root handler.

    When set, logging is reconfigured FIRST, before any heavy import can emit: a spawned child
    inherits no sinks, so loguru's default ``sys.stderr`` sink, ``transformers`` / ``torch``
    (which own their stderr handlers), and :mod:`warnings` would all reach Airflow as
    ERROR-level ``task.stderr`` noise for perfectly healthy records.
    :func:`~dakp_pipeline.logging_setup.configure_worker_logging` redirects the child's stderr
    fd to ``<workdir>/logs/workers/<device>-<pid>.log``.
    """
    if in_worker:
        configure_worker_logging(ner_config.get("workdir"), device)
        _set_parent_death_signal()
        # Fragmentation guard, set BEFORE torch initializes CUDA (it is read once, at init): a
        # multi-hour shard on a 16 GB P100 crept to the memory ceiling and then OOMed on every
        # remaining window. Expandable segments is PyTorch's own remedy for a large
        # reserved-but-unallocated pool. ``setdefault`` so an explicit operator setting wins.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ner = DiseaseNER(device=device, **ner_config)
    items = [_item_parts(item) for item in shard]
    try:
        mentions = ner.extract_batch([text for _set_id, _doc_id, text in items])
    except Exception:
        # The parent only ever receives the pickled exception (``future.result()``), and a spawned
        # child's stderr is this file, so without this line the worker log simply STOPS mid-shard:
        # runs 8 and 9 both died on ``IndexError: string index out of range`` with no traceback
        # anywhere on disk. Name the device and the shard size, keep the traceback, re-raise.
        logger.exception("ner_shard_failed: device = {} items = {}", device, len(items))
        raise
    return [(set_id, doc_id, result) for (set_id, doc_id, _text), result in zip(items, mentions, strict=True)]


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
    _announce_worker_logs(ner_config.get("workdir"))
    ctx = mp.get_context("spawn")
    results: dict[tuple[str, str], list[Mention]] = {}
    with _spawn_safe_main(), ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_mine_shard, shard, ner_config, devices[i], in_worker=True) for i, shard in enumerate(shards)]
        for index, future in enumerate(futures):
            try:
                shard_results = future.result()
            except Exception as exc:
                # Attribute the failure in the TASK log: the traceback itself is in the worker's
                # own file, and without the device name there is no way to know which one to read.
                logger.error(
                    "ner_dispatch_failed: device = {} items = {} error = {!r} worker_logs = {}",
                    devices[index],
                    len(shards[index]),
                    exc,
                    str(Path(ner_config.get("workdir") or "") / WORKER_LOG_SUBDIR),
                )
                raise
            for set_id, doc_id, mentions in shard_results:
                results[(set_id, doc_id)] = mentions
    return results


#: A shaper's existing mining path (multi-GPU dispatch or sequential loop) over the given
#: items, returning ``{(set_id, doc_id): [mentions]}``.
MineFn = Callable[[Sequence[Any]], dict[tuple[str, str], list[Mention]]]


def mine_with_cache(work_items: Sequence[Any], ner: DiseaseNER, mine: MineFn, cache: MentionCache | None) -> dict[tuple[str, str], list[Mention]]:
    """Run ``mine`` over ``work_items``, serving repeats from the persistent mention cache.

    Central caching seam for the DailyMed NER shapers. Text-level flow: every item's cache key
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


def mine_by_position(work_items: Sequence[Any], ner: DiseaseNER, mine: MineFn, cache: MentionCache | None) -> list[list[Mention]]:
    """ ":func:`mine_with_cache` over ``work_items``, returning one mention list PER INPUT ITEM.

    ``(set_id, doc_id)`` is NOT a unique work-item key, and treating it as one silently swaps
    mentions between sections: ``doc_id`` is the SPL *document* id, so one document contributes
    every section it carries — a boxed warning plus its warnings section, two ``34070-3``
    sections, an indication and a contraindication section. Real DailyMed has 20,467 such
    duplicate pairs (11,593 inside the contraindication passes alone), and a mining map keyed by
    the pair keeps only one of them, so the other item receives mentions whose offsets index a
    DIFFERENT text. That is what ended run 12 two seconds after its 2h51m mining:
    ``ValueError: mention offsets must be sentence-relative and within sentence bounds``.

    So items are mined under an ordinal-suffixed doc_id (unique per item) and the results are
    flattened back into input order. The suffix never leaves this function — callers keep their
    real doc_ids in emitted rows, and the mention-cache key is derived from the TEXT, so caching
    and deduplication are unaffected.
    """
    parts = [_item_parts(item) for item in work_items]
    keyed = [(set_id, f"{doc_id}#{index}", text) for index, (set_id, doc_id, text) in enumerate(parts)]
    mined = mine_with_cache(keyed, ner, mine, cache)
    return [mined[(set_id, f"{doc_id}#{index}")] for index, (set_id, doc_id, _text) in enumerate(parts)]


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


__all__ = ["BUILD_HOST_GPUS", "MineFn", "default_ner", "mine_by_position", "mine_with_cache"]
