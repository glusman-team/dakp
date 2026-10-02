"""Shared NER dispatch plumbing for the assertion shapers.

Every shaper that mines DailyMed text with the composite NER backend
(:class:`~dakp_pipeline.ner.ner.DiseaseNER`) needs the same three things:

* **backend construction** — :func:`default_ner` builds the deterministic **offline**
  backend (gazetteer from the ontology fixture, else embedded) used by tests and offline
  runs; production shapers receive an injected ``params["ner"]`` instead.
* **device resolution** — :func:`_resolve_devices` discovers every visible CUDA ordinal and
  filters it to torch-supported devices (None when unusable → sequential CPU mining).
* **device-pinned mining pool** — :class:`MiningPool` owns ONE spawn pool for a whole mining run
  (LPT-balanced by text length), with byte-identical output regardless of dispatch mode. Each worker
  claims a single device slot at startup and keeps its loaded model for the life of the process, so a
  run pays the CUDA-context + flock + weight-load cost once per device instead of once per cache-put
  batch. :func:`dispatch_pool` is the context-managed seam both shapers use; it yields ``None``
  exactly where the shapers used to mine sequentially (offline backend, no usable device).
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
import queue
import signal
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from dakp_pipeline.logging_setup import WORKER_LOG_SUBDIR, configure_worker_logging, logger, prune_worker_logs, stats
from dakp_pipeline.ner.mention_cache import MentionCache, mention_key, ner_cache_material, span_cache_material, span_key
from dakp_pipeline.ner.ner import DiseaseNER, Mention, RawTextSpans, _cuda_device_supported, spans_from_cache

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


_SLOT_CLAIM_TIMEOUT_SECONDS = 600.0
"""How long a worker waits for its device slot before failing loudly.

The claim is a FIFO pop from a queue the parent pre-fills with exactly one entry per pool worker, so
this never blocks in practice. The timeout exists so a mis-sized pool raises here, with a message
naming the situation, instead of hanging a multi-hour DAG task forever.
"""


class NerWorkerError(RuntimeError):
    """A worker chunk failed; carries the device that failed so the parent can name it.

    ``future.result()`` alone re-raises the worker's original exception in the parent, which loses the
    one thing the task log needs: WHICH device died. DAG runs 8 and 9 both surfaced as a bare
    ``IndexError`` with no device and no traceback in the task log. The constructor signature is
    exactly ``(device, items, message)`` because ``BaseException.__reduce__`` replays ``self.args``
    when unpickling, so a mismatched signature would break the exception's trip back to the parent.
    """

    def __init__(self, device: str, items: int, message: str) -> None:
        super().__init__(device, items, message)
        self.device = device
        self.items = items
        self.message = message

    def __str__(self) -> str:
        return f"{self.device}: {self.message}"


_WORKER_SLOT: str | None = None
_WORKER_CONFIG: dict[str, Any] = {}
_WORKER_BACKEND: DiseaseNER | None = None
"""Per-process worker state, populated by :func:`_init_worker` in the child only.

Module globals are the right home because a pool worker is one OS process bound to one device for its
whole life: the slot it claimed, the config it was handed, and the backend it built once. The parent
never populates them (an executor initializer only runs in children), which is what makes calling
:func:`_mine_chunk` directly from a unit test safe: no logging teardown, no death signal, no model.
"""


def _init_worker(slots: Any, ner_config: Mapping[str, Any]) -> None:
    """Pool initializer: claim ONE device slot, then make this process's output land in a file.

    Runs once per worker process, before any task, so it is the earliest moment at which this process
    knows who it is. Three things belong here rather than in the first task:

    1. **Slot claim.** ``ProcessPoolExecutor`` hands a task to whichever worker is free, so a task
       cannot carry its device: the worker must own one. Each worker pops exactly one slot from a
       queue the parent pre-filled, so N workers hold N distinct devices and no card ever sees two
       models (the 16 GB cap that makes the per-device flock necessary).
    2. **Logging redirect.** ``configure_worker_logging`` calls ``logger.remove()`` and
       ``basicConfig(force=True)``, which would wipe the Airflow task process's own sinks if it ever
       ran there. An initializer cannot run in the parent, so this is safe by construction; the old
       ``in_worker`` flag existed only because the same worker function was callable from both sides.
       It also has to happen before the model load: a spawned child inherits no sinks, so loguru's
       default stderr sink plus the ``torch``/``transformers`` stderr handlers would all reach
       Airflow as ERROR-level ``task.stderr`` noise for perfectly healthy records.
    3. **Death signal and allocator env**, both before torch initializes CUDA in this process.
    """
    global _WORKER_SLOT, _WORKER_CONFIG
    try:
        _WORKER_SLOT = str(slots.get(timeout=_SLOT_CLAIM_TIMEOUT_SECONDS))
    except (AttributeError, OSError, queue.Empty) as exc:
        raise RuntimeError(f"NER worker could not claim a device slot ({type(exc).__name__}: {exc})") from exc
    _WORKER_CONFIG = dict(ner_config)
    configure_worker_logging(_WORKER_CONFIG.get("workdir"), _WORKER_SLOT)
    _set_parent_death_signal()
    if _WORKER_SLOT.startswith("cuda"):
        # Fragmentation guard, set BEFORE torch initializes CUDA (it is read once, at init): a
        # multi-hour shard on a 16 GB P100 crept to the memory ceiling and then OOMed on every
        # remaining window. Expandable segments is PyTorch's own remedy for a large
        # reserved-but-unallocated pool. ``setdefault`` so an explicit operator setting wins.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _worker_backend() -> DiseaseNER:
    """This worker's backend, built once per process from its claimed slot and handed config.

    Idempotent on purpose: a worker that receives a chunk before its initializer ran, or a unit test
    calling :func:`_mine_chunk` in-process, still gets a working backend instead of an opaque ``None``
    dereference. The model itself loads lazily inside the first ``extract_spans_batch``, so this
    stays cheap until real work arrives.
    """
    global _WORKER_BACKEND
    if _WORKER_BACKEND is None:
        _WORKER_BACKEND = DiseaseNER(device=_WORKER_SLOT or "cpu", **_WORKER_CONFIG)
    return _WORKER_BACKEND


def _mine_chunk(chunk: Sequence[Any]) -> list[tuple[str, str, RawTextSpans]]:
    """Mine one chunk of work items on this worker's own device slot and return RAW spans.

    The chunk carries no device because the worker already owns one (:func:`_init_worker`), which is
    what lets the parent hand chunks to whichever worker is free. Workers return each text's raw model
    output (:class:`~dakp_pipeline.ner.ner.RawTextSpans`, pickled whole; the JSON-ready cache
    projection happens parent-side in :func:`mine_with_cache`), NOT final mentions: the parent
    re-merges per run so the Tier B span cache stays merge-config-independent.

    Failures are wrapped in :class:`NerWorkerError` so the parent can name the device, and the
    traceback is logged HERE first: the parent only ever sees a re-raised pickled exception, so
    without this the worker's own log file stops mid-chunk with no frame named anywhere on disk.
    """
    if not chunk:
        return []
    slot = _WORKER_SLOT or "cpu"
    items = [_item_parts(item) for item in chunk]
    try:
        backend = _worker_backend()
        spans = backend.extract_spans_batch([text for _set_id, _doc_id, text in items])
    except BaseException as exc:
        logger.error("ner_shard_failed: device = {} items = {}", slot, len(items))
        logger.opt(exception=exc).error("ner_shard_failed_traceback")
        raise NerWorkerError(slot, len(items), f"{type(exc).__name__}: {exc}") from exc
    return [(set_id, doc_id, result) for (set_id, doc_id, _text), result in zip(items, spans, strict=True)]


class MiningPool:
    """A run-scoped, device-pinned NER mining pool with one ``mine`` entry point.

    Why a pool object rather than a function per call: ``mine_with_cache`` slices the cache misses
    into batches of ``DAKP_NERCACHE_PUT_BATCH`` (512 by default) and calls its ``mine`` callable once
    per batch. When that callable built its own ``ProcessPoolExecutor``, a 62k-text build spawned
    ~123 pools, and every worker of every pool re-imported torch/transformers/gliner2, re-acquired the
    per-device flock and re-read the weights from the BLAKE3 model cache. One pool per run turns that
    fixed cost from ``batches x devices`` into ``devices``.

    Construction is free: the executor, the slot queue and the workers are created on the first
    ``mine`` call, so an all-hits run (the warm-cache common case) spawns no process and never
    touches a GPU flock.

    Chunks are still LPT-balanced by text length and, for now, one chunk per slot, so the set of texts
    a worker receives matches what the per-call pool used to give it. WHICH GPU runs a chunk is no
    longer pinned (the executor's own call queue decides), and that is output-neutral: every visible
    device is the same arch running the same deterministic kernels, and the device ordinal never
    enters a cache key.
    """

    def __init__(self, ner: DiseaseNER, slots: Sequence[str]) -> None:
        if not slots:
            raise ValueError("MiningPool needs at least one device slot")
        self._ner_config: dict[str, Any] = ner._config()
        self._slots: tuple[str, ...] = tuple(slots)
        self._stack = ExitStack()
        self._executor: ProcessPoolExecutor | None = None

    @property
    def slots(self) -> tuple[str, ...]:
        """The device slots this pool can run on, in claim order."""
        return self._slots

    @property
    def started(self) -> bool:
        """Whether the executor and its workers exist yet (False for an all-cache-hit run)."""
        return self._executor is not None

    def _worker_log_dir(self) -> str:
        return str(Path(self._ner_config.get("workdir") or "") / WORKER_LOG_SUBDIR)

    def start(self) -> ProcessPoolExecutor:
        """Create the executor and its slot queue once; idempotent.

        ``_spawn_safe_main`` must wrap the executor's whole lifetime, not one call: under Airflow the
        task process has no module spec, and ``spawn`` re-imports ``__main__`` by name in every child
        it creates, which raises ``AttributeError`` without the shim.
        """
        if self._executor is not None:
            return self._executor
        _announce_worker_logs(self._ner_config.get("workdir"))
        ctx = mp.get_context("spawn")
        self._stack.enter_context(_spawn_safe_main())
        slots: Any = ctx.Queue()
        for slot in self._slots:
            slots.put(slot)
        executor = self._stack.enter_context(
            ProcessPoolExecutor(max_workers=len(self._slots), mp_context=ctx, initializer=_init_worker, initargs=(slots, self._ner_config))
        )
        # The feeder thread is the parent's; children hold their own inherited ends, so the parent can
        # stop holding this one open as soon as every slot is queued.
        self._stack.callback(slots.close)
        self._executor = executor
        return executor

    def mine(self, work_items: Sequence[Any]) -> dict[tuple[str, str], RawTextSpans]:
        """Mine ``work_items`` across the pool and collect ``{(set_id, doc_id): RawTextSpans}``."""
        if not work_items:
            return {}
        chunks = [chunk for chunk in _shard_by_text_length(work_items, min(len(self._slots), len(work_items))) if chunk]
        executor = self.start()
        futures: list[Future[Any]] = [executor.submit(_mine_chunk, chunk) for chunk in chunks]
        results: dict[tuple[str, str], RawTextSpans] = {}
        for future in futures:
            try:
                chunk_results = future.result()
            except NerWorkerError as exc:
                self._log_dispatch_failure(exc.device, exc.items, exc.message)
                raise
            except BrokenProcessPool as exc:
                # A worker died without returning anything (OOM kill, SIGKILL from the death signal, a
                # CUDA fault). There is no device to name because no result came back, so point at the
                # per-process worker logs, which is the only place the cause was written.
                self._log_dispatch_failure("unknown", 0, f"{type(exc).__name__}: {exc}")
                raise
            for set_id, doc_id, result in chunk_results:
                results[(set_id, doc_id)] = result
        return results

    def _log_dispatch_failure(self, device: str, items: int, error: str) -> None:
        """Attribute a chunk failure in the TASK log; the traceback itself is in the worker's file."""
        logger.error("ner_dispatch_failed: device = {} items = {} error = {} worker_logs = {}", device, items, error, self._worker_log_dir())

    def close(self) -> None:
        """Shut the pool down, releasing every worker's flock. Safe twice over, or never started."""
        self._executor = None
        self._stack.close()

    def __enter__(self) -> MiningPool:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


@contextmanager
def dispatch_pool(ner: DiseaseNER, devices: Sequence[str] | None) -> Iterator[MiningPool | None]:
    """Yield a run-scoped :class:`MiningPool`, or ``None`` when pooled dispatch is not eligible.

    ``None`` is exactly the condition under which the shapers mine sequentially in-process today: an
    offline gazetteer backend (deliberately never sent to a GPU) or no usable device. Callers keep
    their existing sequential closure for that case, so this seam adds a pool without changing any
    fallback behaviour.
    """
    if not devices or ner._offline:
        yield None
        return
    pool = MiningPool(ner, devices)
    try:
        yield pool
    finally:
        pool.close()


#: A shaper's existing mining path (multi-GPU dispatch or sequential loop) over the given
#: items, returning ``{(set_id, doc_id): [mentions]}``.
MineFn = Callable[[Sequence[Any]], dict[tuple[str, str], list[Mention]]]


def _run_and_merge(
    work_items: Sequence[Any], ner: DiseaseNER, mine: MineFn, spans_by_text: dict[str, dict[str, Any]], tier_b_hits: int
) -> dict[tuple[str, str], list[Mention]]:
    """Run ``mine`` and normalize every result to final mentions, passing span payloads through.

    The single convergence point for all three sources - cache misses, Tier B hits are handled
    by the caller, and the no-cache pass-through. Span-valued results (production) are merged
    here and their ``to_cache`` projection collected into ``spans_by_text`` so the caller can
    store them; mention-valued results (legacy callers and tests) pass through unchanged.
    """
    results = mine(work_items)
    merged: dict[tuple[str, str], list[Mention]] = {}
    for item in work_items:
        set_id, doc_id, text = _item_parts(item)
        value = results.get((set_id, doc_id), [])
        if isinstance(value, RawTextSpans):
            spans_by_text[text] = value.to_cache()
            merged[(set_id, doc_id)] = ner.merge_spans(text, value)
        else:
            merged[(set_id, doc_id)] = list(value)
    return merged


def _mentions_fit(text: str, mentions: Sequence[Mention]) -> bool:
    """True when every cached mention satisfies the mention contract against ``text``.

    ``mention.text == text[mention.start:mention.end]`` is the invariant every DAKP backend emits,
    so it is also the cheapest proof that a cached entry was mined from THIS text. It is needed
    because a cache key hashes the whitespace-FOLDED text while the offsets index the RAW text: an
    entry mined from a whitespace variant of a section — an older DailyMed release, a
    ``raw_text``-instead-of-``clean_text`` fallback — carries offsets into a different string.
    Run 13 served exactly that (mentions at 589..605 for a 568-char contraindication section) and
    the shaper died on ``ValueError: mention offsets must be sentence-relative and within sentence
    bounds``, two seconds after a 2h51m mine. A mismatch is treated as a MISS, so the text is
    re-mined and the stale entry overwritten.
    """
    return all(0 <= mention.start <= mention.end <= len(text) and text[mention.start : mention.end] == mention.text for mention in mentions)


def mine_with_cache(work_items: Sequence[Any], ner: DiseaseNER, mine: MineFn, cache: MentionCache | None) -> dict[tuple[str, str], list[Mention]]:
    """Run ``mine`` over ``work_items``, serving repeats from a TWO-TIER persistent cache.

    Tier A (the original mention cache) stores FINAL mentions keyed by the full config
    fingerprint - a hit means zero post-processing, exactly the warm fast path this seam has
    always provided. Tier B (the span cache) stores RAW MODEL SPANS per text, keyed by model-side
    material only. A Tier B hit is re-merged parent-side (``DiseaseNER.merge_spans`` - cheap,
    deterministic CPU work), which makes gazetteer edits, threshold sweeps, and merge-logic
    changes re-merge on CPU instead of re-mining on GPU: the multi-hour cost of touching anything
    merge-side is what this split exists to delete.

    Lookup order per item: A -> (per distinct text) B -> mine. Freshly mined texts are written to
    BOTH tiers; Tier B hits are written through to Tier A so the next unchanged-config run is all
    A hits. Span-valued ``mine`` results (the production path: workers return raw spans) feed the
    B tier; mention-valued results (legacy callers and tests) feed A only and keep the exact
    historical flow.

    Text-level flow: every item's A key (:func:`~dakp_pipeline.ner.mention_cache.mention_key`)
    is batch-fetched up front; a hit is used only when its offsets index the requesting item's
    own text (:func:`_mentions_fit`); refused entries are purged and re-mined. B keys
    (:func:`~dakp_pipeline.ner.mention_cache.span_key`) are fetched per distinct text; a hit is
    used only when its stored window tiling matches the text's own resolution (a mismatch is a
    MISS, not an error). One representative per distinct TEXT reaches ``mine``, so duplicate texts
    are mined once. The returned ``{(set_id, doc_id): [mentions]}`` map is byte-identical to a
    no-cache run.

    Cache access happens ONLY in this parent process: spawned GPU workers receive no cache
    handle, which keeps the Pebble store single-owner and the worker code untouched. Pass-through
    (``mine`` over everything, verbatim) when ``cache`` is None, when the backend is offline, or
    when the cache server is unavailable.
    """
    if cache is None:
        return _run_and_merge(work_items, ner, mine, {}, 0)
    material = ner_cache_material(ner)
    if material is None:
        return _run_and_merge(work_items, ner, mine, {}, 0)
    model_id, model_b3, fingerprint = material
    span_material = span_cache_material(ner)

    parts = [_item_parts(item) for item in work_items]
    key_by_item = {(set_id, doc_id): mention_key(model_id, model_b3, fingerprint, text) for set_id, doc_id, text in parts}
    hits = cache.get_many(sorted(set(key_by_item.values())))

    usable: dict[tuple[str, str], list[Mention]] = {}
    used_keys: set[str] = set()
    for set_id, doc_id, text in parts:
        raw = hits.get(key_by_item[(set_id, doc_id)])
        if isinstance(raw, list):
            cached = [Mention.from_dict(entry) for entry in raw]
            if _mentions_fit(text, cached):
                usable[(set_id, doc_id)] = cached
                used_keys.add(key_by_item[(set_id, doc_id)])
    # Purge-on-refusal: a hit whose offsets index a different text must not linger in the
    # store, where every future run would re-serve it and re-refuse it (the stale counter
    # that never shrank). Refused keys are re-mined and freshly re-put right below.
    refused = sorted(set(hits) - used_keys)
    if refused:
        cache.delete_many(refused)

    tier_b_hits = 0
    merged_by_text: dict[str, list[Mention]] = {}
    spans_by_text: dict[str, dict[str, Any]] = {}
    representatives: dict[str, Any] = {}  # exact text -> one item carrying it
    a_key_by_text: dict[str, str] = {}
    for item, (set_id, doc_id, text) in zip(work_items, parts, strict=True):
        if (set_id, doc_id) not in usable:
            representatives.setdefault(text, item)
            a_key_by_text.setdefault(text, key_by_item[(set_id, doc_id)])

    if representatives and span_material is not None:
        span_model_id, span_model_b3, span_fp, model_dir = span_material
        b_keys = {text: span_key(span_model_id, span_model_b3, span_fp, text) for text in representatives}
        b_hits = cache.get_many(sorted(set(b_keys.values())))
        for text, b_key in b_keys.items():
            raw = b_hits.get(b_key)
            if raw is None:
                continue
            rebuilt = spans_from_cache(text, model_dir, ner._chunk_words, raw)
            if rebuilt is None:
                # Window-tiling mismatch (or a corrupt entry): a MISS, not an error - the text
                # is re-mined below and the fresh spans overwrite this entry.
                continue
            tier_b_hits += 1
            merged_by_text[text] = ner.merge_spans(text, rebuilt)
            spans_by_text[text] = raw

    still_missing = [item for text, item in representatives.items() if text not in merged_by_text]
    # Chunked mine+put: each batch is mined, re-merged, and flushed to BOTH tiers before the
    # next batch starts, so an OOM crash mid-shard strands at most one batch of work instead of
    # the whole shard (the recorded multi-hour failure mode). A retried task re-fetches per text,
    # so the persisted batches are consumed on resume - see the chunked-resume test.
    batch_size = max(1, int(os.environ.get("DAKP_NERCACHE_PUT_BATCH", "512")))
    tier_b_puts = 0
    for start in range(0, len(still_missing), batch_size):
        batch = still_missing[start : start + batch_size]
        mined = _run_and_merge(batch, ner, mine, spans_by_text, tier_b_hits)
        for item in batch:
            set_id, doc_id, text = _item_parts(item)
            merged_by_text[text] = mined.get((set_id, doc_id), [])
        batch_texts = [text for _, _, text in (_item_parts(item) for item in batch)]
        if span_material is not None:
            span_model_id, span_model_b3, span_fp, _model_dir = span_material
            span_puts = {span_key(span_model_id, span_model_b3, span_fp, text): spans_by_text[text] for text in batch_texts if text in spans_by_text}
            if span_puts:
                cache.put_many(span_puts)
                tier_b_puts += len(span_puts)
        batch_a_puts = {a_key_by_text[text]: [mention.to_dict() for mention in merged_by_text[text]] for text in batch_texts if text in a_key_by_text}
        if batch_a_puts:
            cache.put_many(batch_a_puts)
        stats(logger, "ner_cache_put", chunk=start // batch_size + 1, texts=len(batch), tier_b_puts=tier_b_puts, tier_a_puts=len(batch_a_puts))

    # Legacy mention-valued mine results only ever had Tier A, which the batch loop above
    # already flushed; nothing is left to write here.
    stats(
        logger,
        "ner_mention_cache",
        items=len(work_items),
        hits=len(usable),
        tier_b_hits=tier_b_hits,
        tier_b_puts=tier_b_puts,
        mined=len(representatives),
        stale=len(hits) - len({key_by_item[pair] for pair in usable}),
    )

    out: dict[tuple[str, str], list[Mention]] = {}
    for set_id, doc_id, text in parts:
        if (set_id, doc_id) in usable:
            out[(set_id, doc_id)] = usable[(set_id, doc_id)]
        else:
            out[(set_id, doc_id)] = merged_by_text.get(text, [])
    return out


def mine_by_position(work_items: Sequence[Any], ner: DiseaseNER, mine: MineFn, cache: MentionCache | None) -> list[list[Mention]]:
    """:func:`mine_with_cache` over ``work_items``, returning one mention list PER INPUT ITEM.

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
