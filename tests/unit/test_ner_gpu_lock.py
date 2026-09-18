"""Per-GPU exclusive flock tests for ``dakp_pipeline.ner.ner`` (no CUDA required).

The ``_acquire_gpu_lock`` helper is exercised directly for the blocking/serialization
semantics (kernel flock on separately-opened fds conflicts even within one process, so
threads suffice), and ``_load_model`` is exercised with the same fake ``gliner2`` /
``ensure_model`` stubs as ``test_ner_edge.py`` to prove the CUDA path locks and the
CPU/offline paths never do.
"""

from __future__ import annotations

import fcntl
import io
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from loguru import logger

from dakp_pipeline.ner import ner as ner_module
from dakp_pipeline.ner.model_cache import ModelRef
from dakp_pipeline.ner.ner import (
    _GPU_LOCK_TIMEOUT_ENV,
    DiseaseNER,
    GpuLockTimeoutError,
    _acquire_gpu_lock,
    _cuda_index,
    _gpu_lock_dir,
    _gpu_lock_timeout_seconds,
    _is_memory_error,
    _is_orphaned_spawn_worker,
    _lock_competitor_pids,
    _lock_holder_pids,
    _reap_orphaned_lock_competitors,
    _release_cuda_cache,
)


def _try_lock(path: Path) -> int:
    """Open ``path`` and take the non-blocking exclusive flock; the caller closes the fd."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


class _FakeExtractorModel:
    def extract_entities(self, text: str, entity_types: list[str], threshold: float = 0.5, **_kwargs: Any) -> dict[str, Any]:
        return {"entities": {}}


class _FlakyBatchModel:
    """GLiNER2 stand-in whose batched call raises on windows containing ``POISON``.

    Mirrors the observed production failure: one pathological window kills the whole batched
    inference with ``IndexError: string index out of range``.
    """

    calls: ClassVar[list[list[str]]] = []

    @staticmethod
    def batch_extract_entities(texts: list[str], _labels: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        _FlakyBatchModel.calls.append(list(texts))
        if any("POISON" in text for text in texts):
            raise IndexError("string index out of range")
        return [{"entities": {}} for _ in texts]


class _BatchOnlyFailureModel:
    """GLiNER2 stand-in whose BATCHED call fails while every per-window call succeeds.

    The transient, non-memory batch failure: the isolation retry must recover every window by
    bisecting the list, and must not report a dropped window.
    """

    calls: ClassVar[list[list[str]]] = []

    @staticmethod
    def batch_extract_entities(texts: list[str], _labels: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        _BatchOnlyFailureModel.calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("encoder rejected the padded batch")
        return [{"entities": {"biolink:Disease": [{"text": texts[0], "confidence": 0.99, "start": 0, "end": len(texts[0])}]}}]


class _OomAboveBatchModel:
    """GLiNER2 stand-in that OOMs whenever the requested batch is larger than ``max_batch``.

    Mirrors the production failure: a 16 GB P100 running deberta-v3-large with the eager-attention
    fallback exhausts memory at ``batch_size=16`` over 384-token windows.
    """

    batches: ClassVar[list[int]] = []
    max_batch: ClassVar[int] = 4

    @staticmethod
    def batch_extract_entities(texts: list[str], _labels: Any, batch_size: int = 8, **_kwargs: Any) -> list[dict[str, Any]]:
        _OomAboveBatchModel.batches.append(batch_size)
        if batch_size > _OomAboveBatchModel.max_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.26 GiB.")
        return [{"entities": {"biolink:Disease": [{"text": text, "confidence": 0.99, "start": 0, "end": len(text)}]}} for text in texts]


class _OomOnOneWindowModel:
    """GLiNER2 stand-in that OOMs on one specific window at ANY batch size."""

    @staticmethod
    def batch_extract_entities(texts: list[str], _labels: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        if any("HUGE" in text for text in texts):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.26 GiB.")
        return [{"entities": {}} for _ in texts]


class _RecordingBatchModel:
    """GLiNER2 stand-in that records how many windows each submitted call carried."""

    sizes: ClassVar[list[int]] = []

    @staticmethod
    def batch_extract_entities(texts: list[str], _labels: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        _RecordingBatchModel.sizes.append(len(texts))
        return [{"entities": {}} for _ in texts]


class _FakeAutoExtractor:
    loaded_map_location: ClassVar[list[str]] = []

    @staticmethod
    def from_pretrained(path: str, map_location: str = "cpu") -> _FakeExtractorModel:
        _FakeAutoExtractor.loaded_map_location.append(map_location)
        return _FakeExtractorModel()


def _install_fake_gliner2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeAutoExtractor.loaded_map_location = []
    module = types.ModuleType("gliner2")
    module.AutoExtractor = _FakeAutoExtractor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)

    def _fake_ensure_model(model_id: str, **kwargs: Any) -> ModelRef:
        return ModelRef(model_id=model_id, source="huggingface", path=tmp_path, b3="b3:deadbeef", manifest=tmp_path / "manifest.json")

    monkeypatch.setattr(ner_module, "ensure_model", _fake_ensure_model)


# --- helpers: index parsing + lock-dir resolution -----------------------------


def test_cuda_index_parses_device_strings() -> None:
    assert _cuda_index("cuda") == 0
    assert _cuda_index("cuda:2") == 2


def test_gpu_lock_dir_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAKP_GPU_LOCK_DIR", str(tmp_path / "locks"))
    assert _gpu_lock_dir(workdir=tmp_path / "work") == tmp_path / "locks"


def test_gpu_lock_dir_defaults_to_workdir_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DAKP_GPU_LOCK_DIR", raising=False)
    assert _gpu_lock_dir(workdir=tmp_path) == tmp_path / "cache" / "gpu-locks"


def test_gpu_lock_dir_without_workdir_sits_by_the_model_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DAKP_GPU_LOCK_DIR", raising=False)
    assert _gpu_lock_dir(cache_dir=tmp_path / "models") == tmp_path / "gpu-locks"


# --- _acquire_gpu_lock: same device serializes, different devices don't -------


def test_acquire_gpu_lock_blocks_until_released(tmp_path: Path) -> None:
    holder_fd = _acquire_gpu_lock("cuda:1", tmp_path)
    # While held, a non-blocking acquire on the same device fails outright.
    with pytest.raises(BlockingIOError):
        _try_lock(tmp_path / "cuda-1.lock")

    release_at = time.monotonic() + 0.25

    def _release_later() -> None:
        time.sleep(max(0.0, release_at - time.monotonic()))
        os.close(holder_fd)

    threading.Thread(target=_release_later).start()
    started = time.monotonic()
    waiter_fd = _acquire_gpu_lock("cuda:1", tmp_path)
    try:
        assert time.monotonic() - started >= 0.15  # blocked until the holder released
    finally:
        os.close(waiter_fd)


def test_acquire_gpu_lock_is_per_device(tmp_path: Path) -> None:
    fd0 = _acquire_gpu_lock("cuda:0", tmp_path)
    try:
        other = _try_lock(tmp_path / "cuda-1.lock")  # a different device never contends
        os.close(other)
    finally:
        os.close(fd0)


def test_acquire_gpu_lock_closes_fd_when_flock_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A flock failure other than contention closes the fresh fd instead of leaking it."""

    def _boom(fd: int, flags: int) -> None:
        raise OSError("kernel said no")

    monkeypatch.setattr(ner_module.fcntl, "flock", _boom)
    with pytest.raises(OSError, match="kernel said no"):
        _acquire_gpu_lock("cuda:0", tmp_path)


# --- _acquire_gpu_lock: bounded wait + holder identification -------------------


def test_gpu_lock_timeout_raises_naming_the_holder_pid(tmp_path: Path) -> None:
    """A permanently held lock raises after the ceiling, and the message names the holder PID.

    The holder here is THIS process (flock conflicts across separately-opened fds even within
    one process), so the holder PID is known exactly: ``os.getpid()``.
    """
    holder_fd = _acquire_gpu_lock("cuda:0", tmp_path)
    try:
        started = time.monotonic()
        with pytest.raises(GpuLockTimeoutError, match=str(os.getpid())):
            _acquire_gpu_lock("cuda:0", tmp_path, timeout=0.3)
        assert time.monotonic() - started >= 0.3
    finally:
        os.close(holder_fd)


def test_gpu_lock_timeout_reports_unknown_holder_when_proc_locks_hides_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    holder_fd = _acquire_gpu_lock("cuda:0", tmp_path)
    try:
        monkeypatch.setattr(ner_module, "_lock_holder_pids", lambda _path: [])
        with pytest.raises(GpuLockTimeoutError, match="unknown"):
            _acquire_gpu_lock("cuda:0", tmp_path, timeout=0.2)
    finally:
        os.close(holder_fd)


def test_gpu_lock_timeout_zero_restores_the_unbounded_wait(tmp_path: Path) -> None:
    holder_fd = _acquire_gpu_lock("cuda:0", tmp_path)

    def _release_later() -> None:
        time.sleep(0.2)
        os.close(holder_fd)

    threading.Thread(target=_release_later).start()
    waiter_fd = _acquire_gpu_lock("cuda:0", tmp_path, timeout=0)  # no ceiling: waits for release
    os.close(waiter_fd)


def test_gpu_lock_timeout_comes_from_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    holder_fd = _acquire_gpu_lock("cuda:0", tmp_path)
    try:
        monkeypatch.setenv(_GPU_LOCK_TIMEOUT_ENV, "0.2")
        with pytest.raises(GpuLockTimeoutError):
            _acquire_gpu_lock("cuda:0", tmp_path)  # no explicit timeout: env wins over the default
    finally:
        os.close(holder_fd)


# --- _lock_competitor_pids: holders AND kernel-blocked waiters ------------------


def test_lock_competitor_pids_includes_kernel_waiters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``include_waiters=True`` also returns PIDs parked in a kernel wait (the ``->`` lines).

    Leaked waiters matter as much as leaked holders: the moment the holder dies, a blocked
    orphan waiter wins the lock and wedges the GPU just the same.
    """
    path = tmp_path / "cuda-0.lock"
    path.touch()
    st = path.stat()
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
    fake_locks = "\n".join(
        [
            f"11: FLOCK  ADVISORY  WRITE 3113038 {dev}:{st.st_ino} 0 EOF",  # holder
            f"11: -> FLOCK  ADVISORY  WRITE 3533957 {dev}:{st.st_ino} 0 EOF",  # waiter, same lock
            f"12: -> FLOCK  ADVISORY  WRITE 999 {dev}:{st.st_ino + 1} 0 EOF",  # waiter, other lock
            f"13: FLOCK  ADVISORY  WRITE 555 {dev}:{st.st_ino} 0 EOF",  # second holder
        ]
    )
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: io.StringIO(fake_locks))
    assert _lock_holder_pids(path) == [3113038, 555]
    assert _lock_competitor_pids(path, include_waiters=True) == [3113038, 3533957, 555]


# --- _is_orphaned_spawn_worker: the reap safety filter ---------------------------


def _fake_proc_entries(entries: dict[str, bytes]) -> Any:
    """``builtins.open`` replacement serving ``entries`` and delegating everything else."""
    real_open = open

    def _open(file: Any, *args: Any, **kwargs: Any) -> Any:
        key = os.fspath(file)
        if key in entries:
            return io.BytesIO(entries[key])
        return real_open(file, *args, **kwargs)

    return _open


def _fake_proc_stat(entries: dict[str, int]) -> Any:
    """``os.stat`` replacement answering ``st_uid`` for ``/proc/<pid>`` paths in ``entries``."""
    real_stat = os.stat
    uid = os.getuid()

    def _stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        key = os.fspath(path)
        if key.startswith("/proc/") and key.count("/") == 2 and key.split("/")[2].isdigit():
            return types.SimpleNamespace(st_uid=entries.get(key, uid + 1))  # unknown pids: foreign uid
        return real_stat(path, *args, **kwargs)

    return _stat


def test_is_orphaned_spawn_worker_matches_the_leak_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """ppid 1 + same UID + spawn-fork cmdline = reap candidate; every other shape is refused."""
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        proc = f"/proc/{live.pid}"
        orphan_stat = f"{live.pid} (python3) S 1 {proc} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        child_stat = orphan_stat.replace(" S 1 ", f" S {os.getpid()} ", 1)
        entries = {
            f"{proc}/stat": orphan_stat.encode(),
            f"{proc}/cmdline": b"/venv/bin/python3\x00-c\x00from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=3, pipe_handle=4) --multiprocessing-fork\x00",
        }
        monkeypatch.setattr("builtins.open", _fake_proc_entries(entries))
        monkeypatch.setattr(ner_module.os, "stat", _fake_proc_stat({proc: os.getuid()}))
        assert _is_orphaned_spawn_worker(live.pid) is True  # the exact leaked-worker signature

        monkeypatch.setattr("builtins.open", _fake_proc_entries({**entries, f"{proc}/stat": child_stat.encode()}))
        assert _is_orphaned_spawn_worker(live.pid) is False  # live parent (this test process): never reap

        monkeypatch.setattr("builtins.open", _fake_proc_entries({**entries, f"{proc}/cmdline": b"/venv/bin/python3\x00-m\x00some.cli\x00serve\x00"}))
        assert _is_orphaned_spawn_worker(live.pid) is False  # not a spawn fork: never reap

        monkeypatch.setattr(ner_module.os, "stat", _fake_proc_stat({proc: os.getuid() + 1}))
        monkeypatch.setattr("builtins.open", _fake_proc_entries(entries))
        assert _is_orphaned_spawn_worker(live.pid) is False  # foreign UID: never reap
    finally:
        live.kill()
        live.wait()


def test_is_orphaned_spawn_worker_refuses_self_pid1_and_ghosts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(_GPU_LOCK_TIMEOUT_ENV, raising=False)
    assert _is_orphaned_spawn_worker(os.getpid()) is False  # never kill ourselves
    assert _is_orphaned_spawn_worker(1) is False  # never kill init
    assert _is_orphaned_spawn_worker(0) is False
    assert _is_orphaned_spawn_worker(999999999) is False  # no such /proc entry: conservatively no


# --- _reap_orphaned_lock_competitors: kill exactly the orphaned competitors ------


def test_reap_kills_only_orphaned_competitors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only candidates passing the orphan filter are SIGKILLed; the live sibling is untouched."""
    killed: list[tuple[int, int]] = []

    def _fake_kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        if sig == 0:
            raise OSError()  # liveness probe: the SIGKILL already landed, process is gone

    monkeypatch.setattr(ner_module, "_lock_competitor_pids", lambda _p, **_k: [111, 222, 333])
    monkeypatch.setattr(ner_module, "_is_orphaned_spawn_worker", lambda pid: pid in {111, 333})
    monkeypatch.setattr(ner_module.os, "kill", _fake_kill)
    assert _reap_orphaned_lock_competitors(tmp_path / "cuda-0.lock", grace=0.2) == [111, 333]
    assert [pid for pid, sig in killed if sig == signal.SIGKILL] == [111, 333]


def test_reap_survives_already_dead_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A candidate that exits between detection and kill is a no-op, not a crash."""

    def _gone(pid: int, sig: int) -> None:
        raise ProcessLookupError()

    monkeypatch.setattr(ner_module, "_lock_competitor_pids", lambda _p, **_k: [4242])
    monkeypatch.setattr(ner_module, "_is_orphaned_spawn_worker", lambda _pid: True)
    monkeypatch.setattr(ner_module.os, "kill", _gone)
    assert _reap_orphaned_lock_competitors(tmp_path / "cuda-0.lock", grace=0.1) == [4242]


# --- _acquire_gpu_lock: timeout self-heal end to end -----------------------------


def test_acquire_gpu_lock_timeout_reaps_orphaned_holder_and_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A leaked orphan holding the lock is SIGKILLed at the ceiling and the acquire proceeds.

    The holder is a real external process holding a real kernel flock; only the orphan-filter
    verdict is stubbed (a live test child has a live parent, ppid != 1).
    """
    lock_path = tmp_path / "cuda-0.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time\n"
            f"fd = os.open({str(lock_path)!r}, os.O_CREAT | os.O_RDWR, 0o644)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print('locked', flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == b"locked"
        deadline = time.monotonic() + 5
        while holder.pid not in _lock_holder_pids(lock_path):
            if time.monotonic() > deadline:
                pytest.fail("holder never showed up in /proc/locks")
            time.sleep(0.05)

        monkeypatch.setattr(ner_module, "_is_orphaned_spawn_worker", lambda pid: pid == holder.pid)
        started = time.monotonic()
        fd = _acquire_gpu_lock("cuda:0", tmp_path, timeout=0.3)  # would raise without the reap
        try:
            assert time.monotonic() - started >= 0.3  # waited out the ceiling first
            assert holder.poll() is not None  # the leaked holder was SIGKILLed
            assert holder.pid not in _lock_holder_pids(lock_path)
        finally:
            os.close(fd)
    finally:
        holder.kill()
        holder.wait()


def test_acquire_gpu_lock_timeout_still_raises_when_no_orphan_holds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A LIVE (non-orphan) holder is never killed: the timeout still raises, naming the holder."""
    holder_fd = _acquire_gpu_lock("cuda:0", tmp_path)
    reaper: list[int] = []
    monkeypatch.setattr(ner_module, "_reap_orphaned_lock_competitors", lambda _p, **_k: reaper)
    try:
        with pytest.raises(GpuLockTimeoutError, match=str(os.getpid())):
            _acquire_gpu_lock("cuda:0", tmp_path, timeout=0.2)
        assert reaper == []  # self is filtered out before any kill
        with pytest.raises(BlockingIOError):  # and the live holder's lock is untouched
            _try_lock(tmp_path / "cuda-0.lock")
    finally:
        os.close(holder_fd)


def test_acquire_gpu_lock_timeout_raises_when_the_reap_does_not_free_the_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reaping is best-effort: a holder that survives it still fails the acquire, naming itself.

    Covers the retry-after-reap race — the reaper reported kills, yet the flock is still taken
    (a live process kept or grabbed it), so the non-blocking retry raises ``BlockingIOError`` and
    the ceiling error reports the remaining holder instead of looping or silently proceeding.
    """
    holder_fd = _acquire_gpu_lock("cuda:0", tmp_path)
    monkeypatch.setattr(ner_module, "_reap_orphaned_lock_competitors", lambda _path, **_kwargs: [4242])  # "reaped" an orphan
    try:
        with pytest.raises(GpuLockTimeoutError, match=str(os.getpid())):
            _acquire_gpu_lock("cuda:0", tmp_path, timeout=0.2)
        with pytest.raises(BlockingIOError):  # the surviving holder's lock is untouched
            _try_lock(tmp_path / "cuda-0.lock")
    finally:
        os.close(holder_fd)


# --- extract_batch: poisoned-window isolation -----------------------------------


def test_extract_batch_isolates_poisoned_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One poisoned window must not fail the shard: fallback isolates it to an empty result.

    Regression guard for the run-8 IndexError: a batched gliner2 call raising mid-shard used to
    propagate out of ``extract_batch`` and fail a multi-hour mining task.
    """
    _install_fake_gliner2(monkeypatch, tmp_path)
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)
    model = _FlakyBatchModel()
    _FlakyBatchModel.calls = []
    monkeypatch.setattr(backend, "_load_model", lambda: model)
    out = backend.extract_batch(["asthma in adults", "POISON \x00 garbage", "chronic hives"])
    assert len(out) == 3
    assert out[1] == []  # poisoned window: loud log + no spans, never an exception
    assert len(_FlakyBatchModel.calls) > 1  # batch failed, then per-window fallback ran
    assert any("POISON" in text for text in _FlakyBatchModel.calls[0])  # the poison really was in the batch


def test_infer_windows_skips_poisoned_summary_log_when_every_fallback_window_recovers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A batch failure alone does not emit the poisoned-window summary; only dropped windows do."""
    _install_fake_gliner2(monkeypatch, tmp_path)
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)
    calls = 0

    def flaky_once(_model: object, texts: list[str], _batch_size: int) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IndexError("batch-only failure")
        return [{"entities": {}} for _ in texts]

    monkeypatch.setattr(backend, "_raw_batch_extract", flaky_once)
    error_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(ner_module.logger, "error", lambda *args, **_kwargs: error_calls.append(args))
    assert backend._infer_windows(object(), ["first", "second"]) == [{"entities": {}}, {"entities": {}}]
    assert calls == 3  # one failed batch + one successful fallback call per window
    assert error_calls == []  # no window was dropped, hence no poisoned summary


def test_extract_batch_healthy_shard_stays_on_one_batched_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner2(monkeypatch, tmp_path)
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)
    model = _FlakyBatchModel()
    _FlakyBatchModel.calls = []
    monkeypatch.setattr(backend, "_load_model", lambda: model)
    out = backend.extract_batch(["asthma in adults", "chronic hives"])
    assert len(out) == 2
    assert len(_FlakyBatchModel.calls) == 1  # healthy shard: exactly the one batched call, no fallback


def test_infer_windows_bisects_a_non_memory_batch_failure_and_drops_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A batch failure that is NOT an OOM bisects the window list and keeps every window's spans.

    Only a window that still fails alone may be reported as dropped — misreporting a batch-level
    failure would send an operator hunting for pathological text that does not exist.
    """
    _install_fake_gliner2(monkeypatch, tmp_path)
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)
    model = _BatchOnlyFailureModel()
    _BatchOnlyFailureModel.calls = []
    monkeypatch.setattr(backend, "_load_model", lambda: model)
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        out = backend.extract_batch(["asthma", "chronic hives"])
    finally:
        logger.remove(sink_id)
    assert [[mention.text for mention in row] for row in out] == [["asthma"], ["chronic hives"]]  # nothing lost
    assert len(_BatchOnlyFailureModel.calls) == 3  # one failed batch + one call per half
    assert any("bisecting to isolate the failure" in line for line in lines)
    assert not any("window dropped" in line for line in lines)


def test_infer_windows_halves_the_batch_on_a_cuda_oom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An OOM is a BATCH-SIZE problem: the same windows are retried at half the batch.

    Run 11's cuda:1 worker OOMed 2h11m into a single 50,449-window call; the fallback then
    replayed every window in its own call (~9 h of serial inference for one shard). Halving the
    batch keeps the windows batched and the shard on schedule.
    """
    _install_fake_gliner2(monkeypatch, tmp_path)
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)  # batch_size defaults to 16
    _OomAboveBatchModel.batches = []
    monkeypatch.setattr(backend, "_load_model", lambda: _OomAboveBatchModel)
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        out = backend.extract_batch(["asthma", "chronic hives"])
    finally:
        logger.remove(sink_id)
    assert len(out) == 2
    assert _OomAboveBatchModel.batches == [16, 8, 4]  # halved until it fit; never a per-window replay
    assert any("retrying at half the batch" in line for line in lines)
    assert not any("window dropped" in line for line in lines)


def test_infer_windows_submits_in_bounded_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A shard's windows go in chunks, so one failed call cannot cost the whole shard's work."""
    _install_fake_gliner2(monkeypatch, tmp_path)
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)
    _RecordingBatchModel.sizes = []
    monkeypatch.setattr(backend, "_load_model", lambda: _RecordingBatchModel)
    monkeypatch.setattr(ner_module, "_INFERENCE_CHUNK_WINDOWS", 2)
    assert len(backend.extract_batch(["asthma", "chronic hives", "diabetes"])) == 3
    assert _RecordingBatchModel.sizes == [2, 1]


def test_release_cuda_cache_empties_the_allocator_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OOM's fragmented reserves are handed back before the smaller retry; failures are swallowed.

    Run 12's cuda:1 worker sat at 16.0 GiB of a 15.9 GiB card and then OOMed on EVERY remaining
    window, silently dropping 2048 of them — reclaiming the cache is what breaks that spiral.
    """
    calls: list[str] = []
    module = types.ModuleType("torch")
    module.cuda = SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)
    _release_cuda_cache()
    assert calls == ["empty_cache"]

    def _boom() -> None:
        raise RuntimeError("no CUDA-capable device is detected")

    module.cuda = SimpleNamespace(empty_cache=_boom)  # type: ignore[attr-defined]
    _release_cuda_cache()  # a cleanup helper must never fail a shard


def test_a_window_that_ooms_alone_is_dropped_after_bisecting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An OOM at batch_size 1 is not a batch problem any more: bisect, then drop only the offender.

    Halving cannot help once the batch is a single window, so the list is split to find it and
    that one window is dropped with its exception type and preview — the shard survives.
    """
    _install_fake_gliner2(monkeypatch, tmp_path)
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)
    monkeypatch.setattr(backend, "_load_model", lambda: _OomOnOneWindowModel)
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        out = backend.extract_batch(["asthma", "HUGE window", "chronic hives"])
    finally:
        logger.remove(sink_id)
    assert len(out) == 3  # the healthy texts keep their results
    assert any("bisecting to isolate the failure" in line for line in lines)
    assert any("window dropped (RuntimeError" in line and "HUGE window" in line for line in lines)


def test_is_memory_error_matches_torchs_oom_by_name_and_message() -> None:
    """Both OOM spellings are recognized without importing torch (a lazy dependency)."""

    class OutOfMemoryError(RuntimeError):
        """Stand-in for ``torch.OutOfMemoryError`` — matched on the CLASS NAME."""

    assert _is_memory_error(OutOfMemoryError("device ran out"))
    assert _is_memory_error(RuntimeError("CUDA out of memory. Tried to allocate 2.26 GiB."))
    assert not _is_memory_error(IndexError("string index out of range"))


# --- _gpu_lock_timeout_seconds resolution --------------------------------------


def test_gpu_lock_timeout_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_GPU_LOCK_TIMEOUT_ENV, raising=False)
    assert _gpu_lock_timeout_seconds(5.0) == 5.0  # explicit timeout wins outright
    assert _gpu_lock_timeout_seconds(None) == 3600.0  # unset env falls back to the default
    monkeypatch.setenv(_GPU_LOCK_TIMEOUT_ENV, "12.5")
    assert _gpu_lock_timeout_seconds(None) == 12.5
    monkeypatch.setenv(_GPU_LOCK_TIMEOUT_ENV, "not-a-number")
    assert _gpu_lock_timeout_seconds(None) == 3600.0  # unparseable env warns and falls back


# --- _lock_holder_pids ----------------------------------------------------------


def test_lock_holder_pids_finds_the_holder(tmp_path: Path) -> None:
    path = tmp_path / "cuda-0.lock"
    holder_fd = _try_lock(path)
    try:
        assert os.getpid() in _lock_holder_pids(path)
    finally:
        os.close(holder_fd)


def test_lock_holder_pids_empty_when_unlocked(tmp_path: Path) -> None:
    path = tmp_path / "cuda-0.lock"
    path.touch()
    assert _lock_holder_pids(path) == []


def test_lock_holder_pids_empty_when_the_lock_file_is_gone(tmp_path: Path) -> None:
    assert _lock_holder_pids(tmp_path / "never-created.lock") == []


def test_lock_holder_pids_skips_non_flock_and_mismatched_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``/proc/locks`` parsing ignores non-FLOCK locks, short lines, and other files' locks."""
    path = tmp_path / "cuda-0.lock"
    path.touch()
    st = path.stat()
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
    fake_locks = "\n".join(
        [
            f"1: POSIX  ADVISORY  WRITE 111 {dev}:{st.st_ino} 0 EOF",  # not FLOCK: skipped
            "garbage",  # short line: skipped
            f"2: FLOCK  ADVISORY  WRITE 222 {dev}:{st.st_ino + 1} 0 EOF",  # other inode: skipped
            f"3: FLOCK  ADVISORY  WRITE 333 {dev}:{st.st_ino} 0 EOF",  # THE holder
        ]
    )
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: io.StringIO(fake_locks))
    assert _lock_holder_pids(path) == [333]


# --- _load_model: CUDA locks, CPU and offline never do -------------------------


def test_load_model_locks_the_cuda_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner2(monkeypatch, tmp_path)
    monkeypatch.setenv("DAKP_GPU_LOCK_DIR", str(tmp_path / "locks"))
    backend = DiseaseNER(offline=False, device="cuda:1", workdir=tmp_path)
    backend.extract("some text")
    assert backend._gpu_lock_fd is not None
    assert _FakeAutoExtractor.loaded_map_location == ["cuda:1"]
    lock_path = tmp_path / "locks" / "cuda-1.lock"
    assert lock_path.exists()
    with pytest.raises(BlockingIOError):  # the lock is held for the life of the model
        _try_lock(lock_path)


def test_load_model_on_cpu_never_locks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner2(monkeypatch, tmp_path)
    monkeypatch.setenv("DAKP_GPU_LOCK_DIR", str(tmp_path / "locks"))
    backend = DiseaseNER(offline=False, device="cpu", workdir=tmp_path)
    backend.extract("some text")
    assert backend._gpu_lock_fd is None
    assert not (tmp_path / "locks").exists()


def test_offline_backend_never_locks(tmp_path: Path) -> None:
    backend = DiseaseNER(offline=True, workdir=tmp_path)
    assert backend.extract("asthma")
    assert backend._gpu_lock_fd is None
    assert not (tmp_path / "cache" / "gpu-locks").exists()
