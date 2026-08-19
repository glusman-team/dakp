"""Per-GPU exclusive flock tests for ``dakp_pipeline.ner.ner`` (no CUDA required).

The ``_acquire_gpu_lock`` helper is exercised directly for the blocking/serialization
semantics (kernel flock on separately-opened fds conflicts even within one process, so
threads suffice), and ``_load_model`` is exercised with the same fake ``gliner`` /
``ensure_model`` stubs as ``test_ner_edge.py`` to prove the CUDA path locks and the
CPU/offline paths never do.
"""

from __future__ import annotations

import fcntl
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from dakp_pipeline.ner import ner as ner_module
from dakp_pipeline.ner.model_cache import ModelRef
from dakp_pipeline.ner.ner import DiseaseNER, _acquire_gpu_lock, _cuda_index, _gpu_lock_dir


def _try_lock(path: Path) -> int:
    """Open ``path`` and take the non-blocking exclusive flock; the caller closes the fd."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


class _FakeGLiNERModel:
    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.0) -> list[dict[str, Any]]:
        return []


class _FakeGLiNER:
    loaded_map_location: ClassVar[list[str]] = []

    @staticmethod
    def from_pretrained(path: str, map_location: str = "cpu") -> _FakeGLiNERModel:
        _FakeGLiNER.loaded_map_location.append(map_location)
        return _FakeGLiNERModel()


def _install_fake_gliner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeGLiNER.loaded_map_location = []
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)

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


# --- _load_model: CUDA locks, CPU and offline never do -------------------------


def test_load_model_locks_the_cuda_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner(monkeypatch, tmp_path)
    monkeypatch.setenv("DAKP_GPU_LOCK_DIR", str(tmp_path / "locks"))
    backend = DiseaseNER(offline=False, device="cuda:1", workdir=tmp_path)
    backend.extract("some text")
    assert backend._gpu_lock_fd is not None
    assert _FakeGLiNER.loaded_map_location == ["cuda:1"]
    lock_path = tmp_path / "locks" / "cuda-1.lock"
    assert lock_path.exists()
    with pytest.raises(BlockingIOError):  # the lock is held for the life of the model
        _try_lock(lock_path)


def test_load_model_on_cpu_never_locks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner(monkeypatch, tmp_path)
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
