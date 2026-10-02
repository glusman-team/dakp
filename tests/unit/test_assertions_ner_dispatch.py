"""Unit tests for the shared NER dispatch plumbing (assertions/ner_dispatch.py).

Covers the device-pinned dispatch primitives (the ``_init_worker`` slot claim, the ``_mine_chunk``
worker, the run-scoped ``MiningPool`` orchestrator, device resolution) plus the persistent
mention-cache seam (``mine_with_cache``). All tests run offline with gazetteer-only DiseaseNERs on
"cpu" devices; the executor and its slot queue are stubbed except where a test says it spawns for
real. Cache tests use a fake in-memory cache and a production-mode backend whose ``extract`` is
monkeypatched, so GLiNER never loads.
"""

from __future__ import annotations

import os
import pickle
import queue
from collections.abc import Iterator, Sequence
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from loguru import logger

import dakp_pipeline.assertions.ner_dispatch as dispatch
from dakp_pipeline.assertions.ner_dispatch import MiningPool, NerWorkerError, _mentions_fit, default_ner, mine_by_position, mine_with_cache
from dakp_pipeline.logging_setup import WORKER_LOG_SUBDIR
from dakp_pipeline.ner import model_cache
from dakp_pipeline.ner.ner import DiseaseNER, Mention


def _ner(*terms: str) -> DiseaseNER:
    return DiseaseNER(gazetteer=dict.fromkeys(terms, "disease"))


@contextmanager
def _captured_logs() -> Iterator[list[str]]:
    """Collect rendered loguru lines emitted inside the block (sink removed on exit)."""
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


# --- _init_worker: the device-slot claim ----------------------------------------


class _StubSlots:
    """Stand-in for the parent's slot queue: FIFO pop, no multiprocessing machinery."""

    def __init__(self, *slots: str) -> None:
        self._slots = list(slots)

    def get(self, timeout: float | None = None) -> str:
        if not self._slots:
            raise queue.Empty
        return self._slots.pop(0)


def test_init_worker_claims_a_slot_then_redirects_logging_then_arms_the_death_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, as_worker: Any
) -> None:
    """The initializer's ORDER is the point: the logging redirect happens before anything can emit.

    A spawned child inherits no sinks, so loguru's default stderr sink plus the torch/transformers
    stderr handlers would reach Airflow as ERROR-level ``task.stderr`` noise for perfectly healthy
    records. The model load comes later (lazily, inside the first chunk), so configuring here is
    early enough, and the death signal must be armed before a GPU plus a flock are ever held.
    """
    calls: list[Any] = []
    monkeypatch.setattr(dispatch, "configure_worker_logging", lambda workdir, name: calls.append(("logging", workdir, name)))
    monkeypatch.setattr(dispatch, "_set_parent_death_signal", lambda: calls.append("pdeathsig"))
    as_worker(None, None)  # reset the globals so the assertions below read what THIS call set
    dispatch._init_worker(_StubSlots("cuda:2"), {"workdir": tmp_path, "gazetteer": {"asthma": "disease"}})
    assert calls == [("logging", tmp_path, "cuda:2"), "pdeathsig"]
    assert dispatch._WORKER_SLOT == "cuda:2"
    assert dispatch._WORKER_CONFIG["workdir"] == tmp_path


def test_init_worker_sets_the_allocator_env_only_for_a_cuda_slot(monkeypatch: pytest.MonkeyPatch, as_worker: Any) -> None:
    """``expandable_segments`` is a CUDA-allocator remedy; a CPU slot must not inherit it.

    The env var is read once, when torch initializes CUDA in this process, so it has to be set here
    rather than at model-load time. ``setdefault`` keeps an explicit operator setting in charge.
    """
    monkeypatch.setattr(dispatch, "configure_worker_logging", lambda *_args: None)
    monkeypatch.setattr(dispatch, "_set_parent_death_signal", lambda: None)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    as_worker(None, None)
    dispatch._init_worker(_StubSlots("cpu"), {})
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    dispatch._init_worker(_StubSlots("cuda:0"), {})
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_init_worker_respects_an_operator_set_allocator_env(monkeypatch: pytest.MonkeyPatch, as_worker: Any) -> None:
    monkeypatch.setattr(dispatch, "configure_worker_logging", lambda *_args: None)
    monkeypatch.setattr(dispatch, "_set_parent_death_signal", lambda: None)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    as_worker(None, None)
    dispatch._init_worker(_StubSlots("cuda:0"), {})
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"


def test_init_worker_fails_loudly_when_no_slot_is_available(monkeypatch: pytest.MonkeyPatch, as_worker: Any) -> None:
    """A mis-sized pool raises here instead of hanging a multi-hour DAG task on an empty queue."""
    as_worker(None, None)
    with pytest.raises(RuntimeError, match="could not claim a device slot"):
        dispatch._init_worker(_StubSlots(), {})
    with pytest.raises(RuntimeError, match="could not claim a device slot"):
        dispatch._init_worker(object(), {})  # not a queue at all: AttributeError branch


# --- _mine_chunk: the worker's unit of work ---------------------------------------


def test_chunk_uses_one_batch_call_per_worker(monkeypatch: pytest.MonkeyPatch, as_worker: Any) -> None:
    """One chunk is ONE batched span pass, and every item comes back aligned to its key."""
    calls: list[list[str]] = []

    class WorkerNER:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def extract_spans_batch(self, texts: Sequence[str]) -> list[Any]:
            calls.append(list(texts))
            # Raw spans of an OFFLINE backend are empty (no model); the chunk contract only cares
            # that ONE batched span pass ran and every item came back aligned.
            return [dispatch.RawTextSpans(windows=[], objects=[], qualifiers=[]) for _ in texts]

    monkeypatch.setattr(dispatch, "DiseaseNER", WorkerNER)
    as_worker("cpu", {})
    result = dispatch._mine_chunk([("S1", "D1", "asthma"), ("S2", "D2", "asthma")])
    assert calls == [["asthma", "asthma"]]
    assert {(set_id, doc_id) for set_id, doc_id, _spans in result} == {("S1", "D1"), ("S2", "D2")}


def test_mine_chunk_leaves_process_logging_untouched(monkeypatch: pytest.MonkeyPatch, as_worker: Any) -> None:
    """Mining a chunk in the PARENT process must not wipe the task's own sinks.

    This is why the logging redirect lives in the executor initializer and not in the chunk worker:
    ``configure_worker_logging`` calls ``logger.remove()`` and ``basicConfig(force=True)``, and
    ``multiprocessing.parent_process()`` cannot discriminate (Airflow's LocalExecutor runs the task
    itself under a ``multiprocessing.Process``), so the old explicit ``in_worker`` flag had to guard
    it. An initializer only ever runs in a child, so the guard is now structural rather than a
    convention a future caller could forget.
    """
    reconfigured: list[Any] = []
    monkeypatch.setattr(dispatch, "configure_worker_logging", lambda workdir, name: reconfigured.append((workdir, name)))
    as_worker("cpu", _ner("asthma")._config())
    dispatch._mine_chunk([("S1", "D1", "asthma")])
    assert reconfigured == []


def test_mine_chunk_wraps_a_failure_with_its_device_and_keeps_the_traceback(monkeypatch: pytest.MonkeyPatch, as_worker: Any) -> None:
    """A chunk failure leaves its traceback in the WORKER log and names the device for the parent.

    The parent sees nothing but ``future.result()`` re-raising, so without this the worker file
    simply stops mid-chunk: DAG runs 8 and 9 both died on ``IndexError: string index out of range``
    with no traceback anywhere on disk to name the frame.
    """

    class BoomNER:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def extract_spans_batch(self, _texts: Sequence[str]) -> list[Any]:
            raise IndexError("string index out of range")

    monkeypatch.setattr(dispatch, "DiseaseNER", BoomNER)
    as_worker("cuda:1", {})
    with _captured_logs() as lines, pytest.raises(NerWorkerError) as info:
        dispatch._mine_chunk([("S1", "D1", "asthma"), ("S2", "D2", "hives")])
    assert any("ner_shard_failed: device = cuda:1 items = 2" in line for line in lines)
    assert any("ner_shard_failed_traceback" in line for line in lines)
    assert (info.value.device, info.value.items) == ("cuda:1", 2)
    assert isinstance(info.value.__cause__, IndexError)  # chained, so the original traceback survives
    assert str(info.value) == "cuda:1: IndexError: string index out of range"


def test_worker_error_survives_the_pickle_trip_back_to_the_parent() -> None:
    """The exception's ``args`` must match its signature, or the parent gets a broken object.

    ``BaseException.__reduce__`` replays ``self.args`` through the constructor on unpickle, which is
    why ``NerWorkerError.__init__`` takes exactly ``(device, items, message)`` and passes all three
    to ``super().__init__``. A narrower ``super().__init__(message)`` would raise on the way back.
    """
    err = NerWorkerError("cuda:3", 7, "RuntimeError: CUDA out of memory")
    restored = pickle.loads(pickle.dumps(err))
    assert (restored.device, restored.items, restored.message) == ("cuda:3", 7, "RuntimeError: CUDA out of memory")
    assert str(restored) == str(err)


# --- _set_parent_death_signal ---------------------------------------------------


def test_pool_names_the_device_whose_chunk_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The task log attributes a chunk failure to its device and points at the worker-log dir."""
    _install_stub_pool(monkeypatch, error=NerWorkerError("cuda:1", 1, "IndexError: string index out of range"))
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with _captured_logs() as lines, pytest.raises(NerWorkerError):
        MiningPool(ner, ("cuda:0", "cuda:1")).mine([("S1", "D1", "asthma"), ("S2", "D2", "diabetes")])
    assert any("ner_dispatch_failed: device = cuda:1 items = 1" in line for line in lines)
    assert any(str(tmp_path / WORKER_LOG_SUBDIR) in line for line in lines)


def test_pool_reports_a_broken_process_pool_without_a_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A worker killed outright (OOM kill, SIGKILL from the death signal) returns no result to name.

    ``BrokenProcessPool`` carries no device, so the honest attribution is "unknown" plus a pointer at
    the per-process worker logs, which is the only place the cause was written. Silently blaming the
    first device would send the next debugger to the wrong file.
    """
    _install_stub_pool(monkeypatch, error=BrokenProcessPool("A worker died"))
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with _captured_logs() as lines, pytest.raises(BrokenProcessPool):
        MiningPool(ner, ("cuda:0",)).mine([("S1", "D1", "asthma")])
    assert any("ner_dispatch_failed: device = unknown" in line for line in lines)
    assert any("A worker died" in line for line in lines)


def test_pdeathsig_is_a_noop_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "darwin")
    boom = lambda *_args: (_ for _ in ()).throw(AssertionError("CDLL must not load off-Linux"))
    monkeypatch.setattr(dispatch.ctypes, "CDLL", boom)
    dispatch._set_parent_death_signal()  # returns without touching ctypes


def _fake_libc(prctl_rc: int = 0) -> Any:
    calls: list[tuple[Any, ...]] = []

    class FakeLibc:
        @staticmethod
        def prctl(*args: Any) -> int:
            calls.append(args)
            return prctl_rc

    return FakeLibc(), calls


def test_pdeathsig_arms_sigkill_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    libc, calls = _fake_libc(prctl_rc=0)
    monkeypatch.setattr(dispatch.ctypes, "CDLL", lambda *_a, **_k: libc)
    monkeypatch.setattr(dispatch.os, "getppid", lambda: 4321)
    dispatch._set_parent_death_signal()
    assert calls == [(1, dispatch.signal.SIGKILL, 0, 0, 0)]  # PR_SET_PDEATHSIG = 1


def test_pdeathsig_warns_and_returns_when_prctl_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    libc, _calls = _fake_libc(prctl_rc=-1)
    monkeypatch.setattr(dispatch.ctypes, "CDLL", lambda *_a, **_k: libc)
    exited: list[int] = []
    monkeypatch.setattr(dispatch.os, "_exit", lambda code: exited.append(code))
    dispatch._set_parent_death_signal()  # no exit: the worker still functions without reaping
    assert exited == []


def test_pdeathsig_warns_and_returns_when_prctl_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "linux")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("no libc here")

    monkeypatch.setattr(dispatch.ctypes, "CDLL", _boom)
    dispatch._set_parent_death_signal()


def test_pdeathsig_exits_when_parent_already_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """The set-after-death race: parent died BEFORE prctl armed, so the kernel delivers nothing."""
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    libc, _calls = _fake_libc(prctl_rc=0)
    monkeypatch.setattr(dispatch.ctypes, "CDLL", lambda *_a, **_k: libc)
    monkeypatch.setattr(dispatch.os, "getppid", lambda: 1)
    exited: list[int] = []
    monkeypatch.setattr(dispatch.os, "_exit", lambda code: exited.append(code))
    dispatch._set_parent_death_signal()
    assert exited == [1]


# --- MiningPool lifecycle: one pool per run ----------------------------------------


def _install_stub_pool(monkeypatch: pytest.MonkeyPatch, *, error: Exception | None = None) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    """Stub the spawn context and the executor; return (submitted chunks, constructed executors).

    Keeps the lifecycle tests in-process and instant: a real ``spawn`` pool costs an interpreter per
    worker, and the one test that needs real children says so explicitly.
    """
    submitted: list[list[Any]] = []
    constructed: list[dict[str, Any]] = []

    class _StubQueue:
        def __init__(self) -> None:
            self.items: list[str] = []

        def put(self, item: str) -> None:
            self.items.append(item)

        def get(self, timeout: float | None = None) -> str:
            return self.items.pop(0)

        def close(self) -> None:
            self.items.clear()

    class _StubCtx:
        def Queue(self) -> _StubQueue:
            return _StubQueue()

    monkeypatch.setattr(dispatch.mp, "get_context", lambda _method: _StubCtx())

    class _Future:
        def __init__(self, chunk: Sequence[Any]) -> None:
            self._chunk = chunk

        def result(self) -> list[tuple[str, str, Any]]:
            if error is not None:
                raise error
            return [(s, d, dispatch.RawTextSpans(windows=[], objects=[], qualifiers=[])) for s, d, _t in self._chunk]

    class _Executor:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

        def __enter__(self) -> _Executor:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, _function: Any, chunk: Sequence[Any]) -> _Future:
            submitted.append(list(chunk))
            return _Future(chunk)

    monkeypatch.setattr(dispatch, "ProcessPoolExecutor", _Executor)
    return submitted, constructed


def test_pool_reuses_one_executor_across_mine_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """THE payoff test: N cache-put batches through ONE pool build the executor once.

    ``mine_with_cache`` slices the misses into batches of ``DAKP_NERCACHE_PUT_BATCH`` (512) and calls
    its ``mine`` callable per batch. When that callable built its own pool, a 62k-text build spawned
    ~123 of them and every worker of every pool re-imported torch, re-acquired the per-device flock
    and re-read the weights from the model cache.
    """
    _submitted, constructed = _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with MiningPool(ner, ("cuda:0", "cuda:1")) as pool:
        for batch in range(3):  # three cache-put batches, as mine_with_cache would issue them
            pool.mine([(f"S{batch}", f"D{i}", "asthma") for i in range(4)])
        assert pool.started
    assert len(constructed) == 1  # one executor for the whole run, not one per batch
    assert constructed[0]["max_workers"] == 2
    assert not pool.started  # closed on exit


def test_pool_announces_worker_logs_once_per_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The parent names the worker-log dir once, and prunes stale files once, per run.

    Worker narration lives in files the Airflow task log never shows, so without this line there is
    no pointer to it - and nothing else ever retires those per-process files. Announcing per batch
    (the old behaviour) re-pruned the directory mid-run, deleting the live workers' own logs.
    """
    announced: list[Any] = []
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 3)
    monkeypatch.setattr(dispatch, "stats", lambda _log, event, **fields: announced.append((event, fields)))
    _install_stub_pool(monkeypatch)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with MiningPool(ner, ("cuda:0",)) as pool:
        pool.mine([("S1", "D1", "asthma"), ("S2", "D2", "asthma")])
        pool.mine([("S3", "D3", "asthma"), ("S4", "D4", "asthma")])
    assert announced == [("ner_worker_logs", {"path": str(tmp_path / WORKER_LOG_SUBDIR), "pruned": 3})]


def test_pool_spawns_nothing_until_there_is_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An all-cache-hit run must not spawn a process or touch a GPU flock.

    Warm cache is the common case for a re-run, so the pool is lazy by construction: nothing is
    created until the first non-empty ``mine`` call.
    """
    _submitted, constructed = _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with MiningPool(ner, ("cuda:0", "cuda:1")) as pool:
        assert pool.mine([]) == {}
        assert not pool.started
    assert constructed == []


def test_pool_hands_every_slot_one_distinct_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The slot queue is pre-filled with exactly one entry per pool worker."""
    _submitted, constructed = _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    pool = MiningPool(ner, ("cuda:0", "cuda:1", "cuda:2", "cuda:3"))
    pool.start()
    assert pool.slots == ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    assert constructed[0]["initargs"][0].items == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert constructed[0]["initializer"] is dispatch._init_worker
    pool.close()


def test_pool_start_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    pool = MiningPool(ner, ("cuda:0",))
    assert pool.start() is pool.start()
    pool.close()
    pool.close()  # closing twice is safe (the shaper's finally block may race a with-exit)


def test_pool_rejects_an_empty_slot_list(tmp_path: Path) -> None:
    """An empty plan is a caller bug, not a silently sequential run."""
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with pytest.raises(ValueError, match="at least one device slot"):
        MiningPool(ner, ())


def test_four_slot_pool_creates_four_distinct_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A sufficiently large workload is cut into one chunk per slot, with no item lost or doubled."""
    submitted, _constructed = _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    items = [("S", f"D{i}", "asthma " * (i + 1)) for i in range(8)]
    with MiningPool(ner, ("cuda:0", "cuda:1", "cuda:2", "cuda:3")) as pool:
        pool.mine(items)
    assert len(submitted) == 4
    assert sorted(item[1] for chunk in submitted for item in chunk) == [f"D{i}" for i in range(8)]
    assert all(chunk for chunk in submitted)  # no empty chunk is ever submitted


def test_pool_fewer_items_than_slots_submits_no_empty_chunk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    submitted, constructed = _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with MiningPool(ner, ("cuda:0", "cuda:1", "cuda:2", "cuda:3")) as pool:
        pool.mine([("S1", "D1", "asthma")])
    assert len(submitted) == 1
    assert constructed[0]["max_workers"] == 4  # the plan is sized by slots; workers spawn on demand


def test_pool_worker_log_dir_names_the_run_workdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    assert MiningPool(ner, ("cuda:0",))._worker_log_dir() == str(tmp_path / WORKER_LOG_SUBDIR)
    no_workdir = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=None)
    assert MiningPool(no_workdir, ("cuda:0",))._worker_log_dir() == str(Path("") / WORKER_LOG_SUBDIR)


# --- dispatch_pool: the shaper-facing seam ----------------------------------------


def test_dispatch_pool_yields_none_for_an_offline_backend() -> None:
    """The offline gazetteer is deliberately never sent to a GPU (CPU-cheap, deterministic)."""
    with dispatch.dispatch_pool(_ner("asthma"), ("cuda:0", "cuda:1")) as pool:
        assert pool is None


def test_dispatch_pool_yields_none_without_devices() -> None:
    """No usable device (CI, a laptop, or an arch the torch build cannot run) means sequential."""
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"})
    with dispatch.dispatch_pool(ner, None) as pool:
        assert pool is None
    with dispatch.dispatch_pool(ner, ()) as pool:
        assert pool is None


def test_dispatch_pool_yields_a_pool_and_closes_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_stub_pool(monkeypatch)
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with dispatch.dispatch_pool(ner, ("cuda:0", "cuda:1")) as pool:
        assert isinstance(pool, MiningPool)
        pool.mine([("S1", "D1", "asthma"), ("S2", "D2", "asthma")])
        assert pool.started
    assert pool is not None  # narrowed for the type checker
    assert not pool.started  # closed on exit, flock released


def test_dispatch_pool_closes_even_when_mining_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed chunk must not leak the pool (and with it the per-device flocks)."""
    _install_stub_pool(monkeypatch, error=NerWorkerError("cuda:0", 1, "boom"))
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 0)
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with pytest.raises(NerWorkerError), dispatch.dispatch_pool(ner, ("cuda:0",)) as pool:
        cast("MiningPool", pool).mine([("S1", "D1", "asthma")])  # narrowed: this seam yields a pool here
    assert pool is not None  # a production backend with devices always gets a pool
    assert not pool.started


def test_announce_worker_logs_without_a_workdir_says_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No workdir: there is no worker-log directory to name or prune (the None branch)."""
    announced: list[Any] = []
    monkeypatch.setattr(dispatch, "stats", lambda _log, event, **_fields: announced.append(event))
    dispatch._announce_worker_logs(None)
    assert announced == []


# --- mine_by_position: (set_id, doc_id) is not a unique work-item key --------------


def test_mine_by_position_gives_duplicate_documents_their_own_mentions() -> None:
    """Two sections of ONE SPL document must not share a mining-map entry.

    Regression guard for run 12: ``doc_id`` is the document id, so the pair collided, one item
    received the other's mentions, and those offsets — relative to a different text — failed
    ``attach_qualifiers_with_scores``' sentence-bounds contract two seconds after a 2h51m mine.
    """
    items = [("SET-A", "DOC-A", "asthma"), ("SET-A", "DOC-A", "diabetes"), ("SET-B", "DOC-B", "asthma")]
    ner = _ner("asthma", "diabetes")

    def mine(batch: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        # Receives the ordinal-suffixed proxies; keys its results by whatever it is handed.
        return {(set_id, doc_id): ner.extract(text) for set_id, doc_id, text in batch}

    mined = mine_by_position(items, ner, mine, None)
    assert [[mention.text for mention in row] for row in mined] == [["asthma"], ["diabetes"], ["asthma"]]


def test_mine_by_position_deduplicates_identical_texts_through_the_cache_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical texts are still mined once: the rekeying leaves the cache/dedup seam intact."""
    items = [("SET-A", "DOC-A", "asthma"), ("SET-A", "DOC-A", "asthma"), ("SET-A", "DOC-B", "diabetes")]
    ner = _ner("asthma", "diabetes")
    monkeypatch.setattr(dispatch, "ner_cache_material", lambda _backend: ("model", "b3deadbeef", "fingerprint"))
    mined_texts: list[str] = []

    def mine(batch: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        mined_texts.extend(text for _set_id, _doc_id, text in batch)
        return {(set_id, doc_id): ner.extract(text) for set_id, doc_id, text in batch}

    cache = _FakeCache()
    out = mine_by_position(items, ner, mine, cache)  # type: ignore[arg-type]
    assert sorted(mined_texts) == ["asthma", "diabetes"]  # one representative per distinct text
    assert [[mention.text for mention in row] for row in out] == [["asthma"], ["asthma"], ["diabetes"]]
    assert (cache.get_calls, cache.put_calls) == (1, 1)


# --- default_ner -----------------------------------------------------------------


def test_default_ner_without_fixture_root_uses_embedded_gazetteer() -> None:
    ner = default_ner(None)
    assert ner._offline
    assert [m.text for m in ner.extract("patients with asthma")] == ["asthma"]


# --- mine_with_cache ------------------------------------------------------------


class _FakeCache:
    """In-memory stand-in for MentionCache, mirroring the real client's raw-value contract.

    Stores mention/span payloads VERBATIM (``to_dict``/``to_cache`` projections) and returns
    them undecoded on read - decoding is the caller's job now (the server is tier-agnostic),
    so a hit is only byte-identical if the caller's own (de)serialization is lossless.
    """

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.get_calls = 0
        self.put_calls = 0
        self.deleted: list[str] = []

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        self.get_calls += 1
        return {key: self.store[key] for key in keys if key in self.store}

    def put_many(self, items: dict[str, Any]) -> None:
        self.put_calls += 1
        self.store.update(items)

    def delete_many(self, keys: list[str]) -> None:
        for key in keys:
            self.deleted.append(key)
            self.store.pop(key, None)


def _production_ner(tmp_path: Path) -> DiseaseNER:
    """A production-mode backend whose model is cached on disk (manifest only, no GLiNER load)."""

    def _write_model(_id: str, dest: Path) -> None:
        (dest / "w.bin").write_bytes(b"w")

    model_cache.ensure_model("acme/test-ner", cache_dir=tmp_path, downloader=_write_model)
    return DiseaseNER(offline=False, model_id="acme/test-ner", cache_dir=tmp_path)


def _counting_extract(ner: DiseaseNER, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``ner.extract`` with a counting fake; returns the call log."""
    calls: list[str] = []

    def fake_extract(text: str) -> list[Mention]:
        calls.append(text)
        return [Mention(text=text, start=0, end=len(text), type="Disease", score=0.9)]

    monkeypatch.setattr(ner, "extract", fake_extract)
    return calls


def _sequential_mine(ner: DiseaseNER):
    def mine(items: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        return {(item[0], item[1]): ner.extract(item[2]) for item in items}

    return mine


def test_mentions_fit_rejects_an_entry_mined_from_another_text() -> None:
    """The mention contract is the cheap proof that a cached entry belongs to THIS text."""
    text = "contraindicated in asthma"
    start = text.index("asthma")
    assert _mentions_fit(text, [Mention("asthma", start, start + 6, "Disease", 1.0)])
    assert _mentions_fit(text, [])  # an empty entry is a valid "no mentions" result, not a miss
    assert not _mentions_fit(text, [Mention("asthma", 589, 605, "Disease", 1.0)])  # run 13: past the end
    assert not _mentions_fit(text, [Mention("asthma", 0, 6, "Disease", 1.0)])  # in bounds, wrong surface
    assert not _mentions_fit(text, [Mention("asthma", -1, start + 6, "Disease", 1.0)])  # negative start


def test_mine_with_cache_reminds_when_a_cached_entry_does_not_fit_the_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale entry is a MISS: the entry is purged, the text re-mined, the value re-stored.

    Regression guard for run 13 — a cached entry mined from a whitespace variant of the section
    served offsets past the end of the requesting text (589..605 into 568 chars), which the
    shaper's sentence-bounds contract turned into a failed task after 2h51m of mining.
    """
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma")]
    first = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert calls == ["asthma"]

    key = next(iter(cache.store))
    stale = [Mention("asthma", 589, 605, "Disease", 1.0).to_dict()]
    cache.store[key] = stale
    calls.clear()
    second = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert calls == ["asthma"]  # re-mined instead of served
    assert second == first  # and the correct mentions are back
    assert cache.store[key] != stale  # the stale entry was overwritten
    assert cache.deleted == [key]  # and the refused entry was purged before the re-put


def test_mine_with_cache_keys_whitespace_variants_separately(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Whitespace variants of one section carry offsets into different strings, so they key apart.

    Keying folds whitespace, which made variants collide: each run re-served one variant's entry
    to the other, refused it on the mention contract, and re-mined — the stale counter that never
    shrank. The raw-text length in the key disambiguates variants, so both variants cache their
    own offsets and a second run is all hits.
    """
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma  in adults"), ("S2", "D2", "asthma in adults")]
    mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert sorted(calls) == ["asthma  in adults", "asthma in adults"]
    assert len(cache.store) == 2  # one key per raw variant: no collision, no see-saw

    calls.clear()
    mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert calls == []  # both variants served from their own entries
    assert cache.deleted == []  # nothing refused, nothing purged


def test_mine_with_cache_none_cache_passes_through() -> None:
    ner = _ner("asthma")
    items = [("S1", "D1", "asthma")]
    assert mine_with_cache(items, ner, _sequential_mine(ner), None) == {("S1", "D1"): ner.extract("asthma")}


def test_mine_with_cache_offline_backend_never_touches_cache() -> None:
    """The offline gazetteer is deterministic and CPU-cheap: deliberately not cached."""
    ner = _ner("asthma")
    cache = _FakeCache()
    mine_with_cache([("S1", "D1", "asthma")], ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert (cache.get_calls, cache.put_calls) == (0, 0)


def test_mine_with_cache_misses_mine_once_per_distinct_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Duplicate texts across work items are mined exactly once; results merge per item."""
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma"), ("S2", "D2", "diabetes"), ("S3", "D3", "asthma")]
    results = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert sorted(calls) == ["asthma", "diabetes"]
    assert set(results.keys()) == {("S1", "D1"), ("S2", "D2"), ("S3", "D3")}
    assert results[("S1", "D1")] == results[("S3", "D3")]  # same text -> same mentions
    assert len(cache.store) == 2


def test_mine_with_cache_second_run_is_all_hits_and_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A warm cache serves everything: extract is never called, output is byte-identical."""
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma"), ("S2", "D2", "diabetes")]
    first = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert len(calls) == 2

    calls.clear()
    second = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert calls == []  # all hits: the backend never ran
    assert second == first
    assert cache.put_calls == 1  # nothing re-put on the warm run


def test_mine_with_cache_results_match_no_cache_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cache hits + misses merge into exactly what a no-cache run produces."""
    ner = _production_ner(tmp_path)
    _counting_extract(ner, monkeypatch)
    items = [("S1", "D1", "asthma"), ("S2", "D2", "diabetes")]
    no_cache = mine_with_cache(items, ner, _sequential_mine(ner), None)
    cache = _FakeCache()
    cached_cold = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    cached_warm = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert cached_cold == no_cache
    assert cached_warm == no_cache


# --- Tier B span-cache flow -------------------------------------------------------


class _SpanMine:
    """A production-style mine closure: returns raw spans and records every text it mined."""

    def __init__(self, ner: DiseaseNER) -> None:
        self.ner = ner
        self.mined: list[str] = []

    def __call__(self, items: Sequence[Any]) -> dict[tuple[str, str], Any]:
        out: dict[tuple[str, str], Any] = {}
        for set_id, doc_id, text in items:
            self.mined.append(text)
            out[(set_id, doc_id)] = self.ner.extract_spans(text)
        return out


def _counting_extract_spans(ner: DiseaseNER, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``ner.extract_spans`` with a counting fake returning ONE full-text span.

    The real ``extract_spans`` loads GLiNER (network + torch); these tests exercise the CACHE
    tiers and the parent-side merge, not the model, so a deterministic single-span fake keeps
    them hermetic while still exercising ``merge_spans`` on real span material.
    """
    calls: list[str] = []

    def fake_extract_spans(text: str) -> Any:
        calls.append(text)
        from dakp_pipeline.ner.ner import RawTextSpans, _ModelSpan

        return RawTextSpans(windows=[(0, text)], objects=[[_ModelSpan(start=0, end=len(text), type="Disease", score=0.9)]], qualifiers=[])

    monkeypatch.setattr(ner, "extract_spans", fake_extract_spans)
    return calls


def test_mine_with_cache_tier_b_stores_spans_and_write_through_tier_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A span-valued cold run fills BOTH tiers: B gets raw spans, A gets the merged mentions."""
    ner = _production_ner(tmp_path)
    calls = _counting_extract_spans(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma in adults")]
    out = mine_with_cache(items, ner, _SpanMine(ner), cache)  # type: ignore[arg-type]
    assert [m.text for m in out[("S1", "D1")]] == ["asthma in adults"]  # the fake span, merged
    assert len(calls) == 1
    assert cache.put_calls == 2  # one put_many per tier
    assert len(cache.store) == 2  # one span payload + one mention list
    span_entries = [v for v in cache.store.values() if isinstance(v, dict)]
    mention_entries = [v for v in cache.store.values() if isinstance(v, list)]
    assert len(span_entries) == 1
    assert "starts" in span_entries[0]
    assert len(mention_entries) == 1


def test_mine_with_cache_tier_b_hit_avoids_the_gpu_on_merge_side_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """THE payoff test: an accept-threshold sweep hits Tier B and never re-mines.

    The threshold change invalidates every Tier A key (full fingerprint) but NOT Tier B
    (model-side fingerprint), so the second run serves spans from B, re-merges on CPU, and the
    extract path (the GPU cost) never runs. This is what makes gazetteer/threshold iteration
    CPU-cheap.
    """
    ner = _production_ner(tmp_path)
    calls = _counting_extract_spans(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma in adults")]
    mine = _SpanMine(ner)
    first = mine_with_cache(items, ner, mine, cache)  # type: ignore[arg-type]
    assert mine.mined == ["asthma in adults"]
    assert calls == ["asthma in adults"]

    swept = DiseaseNER(offline=False, model_id="acme/test-ner", accept_threshold=0.9, cache_dir=tmp_path)
    swept_calls = _counting_extract_spans(swept, monkeypatch)
    swept_out = mine_with_cache(items, swept, _SpanMine(swept), cache)  # type: ignore[arg-type]
    assert swept_calls == []  # Tier B hit: no GPU, only a CPU re-merge under the new floor
    assert swept_out == first  # same fake-model output -> same mentions


def test_mine_with_cache_tier_b_window_mismatch_is_a_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Tier B entry whose stored tiling does not match the text's budget resolution is re-mined.

    Windows are recomputed parent-side and checked against the stored starts; an entry produced
    under a different chunk_words must never be merged with the wrong tiling.
    """
    ner = _production_ner(tmp_path)
    calls = _counting_extract_spans(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma in adults")]
    mine = _SpanMine(ner)
    mine_with_cache(items, ner, mine, cache)  # type: ignore[arg-type]
    assert calls == ["asthma in adults"]
    assert len(cache.store) == 2

    # Corrupt the span payload's tiling: the entry must be treated as a MISS (re-mined),
    # not merged blindly.
    span_key = next(k for k, v in cache.store.items() if isinstance(v, dict))
    cache.store[span_key] = {**cache.store[span_key], "starts": [999]}
    rebudget_ner = DiseaseNER(offline=False, model_id="acme/test-ner", chunk_words=512, cache_dir=tmp_path)
    rebudget_calls = _counting_extract_spans(rebudget_ner, monkeypatch)
    mine_with_cache(items, rebudget_ner, _SpanMine(rebudget_ner), cache)  # type: ignore[arg-type]
    assert rebudget_calls == ["asthma in adults"]  # mismatch -> re-mine, never a corrupt merge


def test_mine_with_cache_none_cache_normalizes_spans_to_mentions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The no-cache pass-through still merges spans: shapers always receive final mentions."""
    ner = _production_ner(tmp_path)
    _counting_extract_spans(ner, monkeypatch)
    items = [("S1", "D1", "asthma in adults")]
    out = mine_with_cache(items, ner, _SpanMine(ner), None)  # type: ignore[arg-type]
    assert [m.text for m in out[("S1", "D1")]] == ["asthma in adults"]


# --- chunked mine+put (crash-bounded shard writes) ---------------------------------


class _ExplodingMine:
    """Mines the first N texts, then raises - simulates an OOM crash mid-shard."""

    def __init__(self, ner: DiseaseNER, survive: int) -> None:
        self.ner = ner
        self.survive = survive
        self.mined: list[str] = []

    def __call__(self, items: Sequence[Any]) -> dict[tuple[str, str], Any]:
        out: dict[tuple[str, str], Any] = {}
        for set_id, doc_id, text in items:
            if len(self.mined) >= self.survive:
                raise RuntimeError("simulated OOM: worker died mid-shard")
            self.mined.append(text)
            out[(set_id, doc_id)] = self.ner.extract_spans(text)
        return out


def test_chunked_puts_resume_after_mid_shard_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A crash mid-shard keeps the completed batches; the retry mines only the remainder.

    Batches are flushed to BOTH tiers as soon as they finish, so the retry's cache state shows
    exactly where the first attempt died: the resumed shard's output must equal an uninterrupted
    run's, with the survived texts served from cache (no GPU) and only the rest mined.
    """
    monkeypatch.setenv("DAKP_NERCACHE_PUT_BATCH", "2")
    ner = _production_ner(tmp_path)
    calls = _counting_extract_spans(ner, monkeypatch)
    cache = _FakeCache()
    texts = [f"indicated for asthma {i}" for i in range(5)]
    items = [("S1", f"D{i}", text) for i, text in enumerate(texts)]

    boom = _ExplodingMine(ner, survive=4)
    with pytest.raises(RuntimeError, match="simulated OOM"):
        mine_with_cache(items, ner, boom, cache)  # type: ignore[arg-type]
    stored_after_crash = len(cache.store)
    assert stored_after_crash == 8  # two flushed batches x (B span + A mention); batch 3 never ran
    assert calls == texts[:4]  # exactly the survived texts were mined

    # Retry with a healthy mine closure: batches 1-2 are all cache hits, the tail is mined.
    retry_calls = _counting_extract_spans(ner, monkeypatch)
    recovered = mine_with_cache(items, ner, _SpanMine(ner), cache)  # type: ignore[arg-type]
    assert len(retry_calls) == 1  # texts[4:] re-mined; texts[:4] came from its own earlier chunks
    assert set(retry_calls) == set(texts[4:])

    # The recovered result is identical to a never-crashed run.
    fresh_cache = _FakeCache()
    fresh_calls = _counting_extract_spans(ner, monkeypatch)
    fresh = mine_with_cache(items, ner, _SpanMine(ner), fresh_cache)  # type: ignore[arg-type]
    assert len(fresh_calls) == 5
    assert recovered == fresh


def test_chunked_puts_single_batch_when_unconfigured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without the env override, the whole shard is one mine+put (the pre-chunking behavior)."""
    ner = _production_ner(tmp_path)
    _counting_extract_spans(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", f"D{i}", f"text {i}") for i in range(5)]
    out = mine_with_cache(items, ner, _SpanMine(ner), cache)  # type: ignore[arg-type]
    assert len(out) == 5
    assert cache.put_calls == 2  # one B flush + one A flush for the whole shard
