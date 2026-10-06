"""Test-wide configuration shared by every suite (unit + integration + eval).

Forces the stdlib ``urllib`` download path for the whole suite so the offline tests that
monkeypatch ``urllib.request.urlopen`` — notably ``tests/integration/test_prod_smoke.py``,
which exercises the REAL fetcher download branches through that seam — stay deterministic and
network-free even though the bundled aria2c binary is installed. The aria2c code paths are
covered directly by ``tests/unit/test_downloader.py``, which opts back in per test.

Also manages the ``coverage`` interpreter-startup hook without taxing every test process:

- ``COVERAGE_PROCESS_START`` is what arms the installed ``coverage`` ``.pth`` file so
  ``spawn``-started child processes (the per-GPU NER workers) are measured. It is exported
  ONLY when this very run measures coverage: the variable is inherited verbatim by every
  subprocess, so leaking it into a ``--no-cov`` run started a line tracer in every xdist
  worker from interpreter start (measured: the 3.1 MB LinkML-generated
  ``biolink_model.datamodel.pydanticmodel_v2`` import ran ~4x slower per worker, ~11.5s
  instead of ~3s, landing inside the first config test each worker executed).
- An xdist worker that DID inherit the variable stops its ``.pth`` collector immediately:
  xdist workers are already measured end-to-end by pytest-cov's own engine, so the second
  tracer only double-traced every line. Test-spawned subprocesses are unaffected — each one
  is a fresh interpreter that starts its own collector, and the variable stays in the
  environment for exactly that purpose. Coverage of worker lines is unchanged (the engine
  records everything from ``pytest_configure`` onward, and pre-configure imports are
  site/pytest internals outside the ``dakp_pipeline`` coverage source).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _enable_subprocess_coverage() -> None:
    """Point ``COVERAGE_PROCESS_START`` at this repo's config so child processes are measured.

    The per-GPU NER workers run under the ``spawn`` start method, so a child is a brand-new
    interpreter that inherits no coverage tracer. The installed ``coverage`` ``.pth`` file calls
    ``coverage.process_startup()`` at interpreter start, but ONLY when this variable is set, and
    it must be set before any child is spawned (it is inherited through the environment).
    Pairs with ``parallel``/``concurrency`` in ``[tool.coverage.run]``.
    """
    os.environ.setdefault("COVERAGE_PROCESS_START", str(_REPO_ROOT / "pyproject.toml"))


def _coverage_measurement_enabled(config: pytest.Config) -> bool:
    """True when this run actually measures coverage (``--cov`` given, ``--no-cov`` absent)."""
    cov_source = config.getoption("cov_source", None)
    no_cov = config.getoption("no_cov", False)
    return bool(cov_source) and not no_cov


def _stop_duplicate_worker_coverage() -> None:
    """Stop the ``.pth``-started tracer inside an xdist worker (pytest-cov's engine measures it).

    ``coverage.process_startup`` records its instance on the function object, so this only ever
    stops the interpreter-start collector — never pytest-cov's own engine, whichever of the two
    ``pytest_configure`` hooks runs first.
    """
    import coverage

    pth_collector = getattr(coverage.process_startup, "coverage", None)
    if pth_collector is None:
        return
    try:
        pth_collector.stop()
        pth_collector.save()
    except Exception:
        pass


def pytest_configure(config: pytest.Config) -> None:
    if _coverage_measurement_enabled(config):
        _enable_subprocess_coverage()
    if hasattr(config, "workerinput") and os.environ.get("COVERAGE_PROCESS_START"):
        _stop_duplicate_worker_coverage()


@pytest.fixture(autouse=True)
def _force_urllib_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to the urllib download backend (aria2c off).

    Individual tests re-enable aria2c with ``monkeypatch.setenv("DAKP_ARIA2", "1")`` or
    ``monkeypatch.delenv("DAKP_ARIA2", raising=False)``.
    """
    monkeypatch.setenv("DAKP_ARIA2", "0")


@pytest.fixture(autouse=True)
def _hermetic_gpu_lock_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the CUDA flock at a per-test directory (``DAKP_GPU_LOCK_DIR`` wins in ``_gpu_lock_dir``).

    On CPU-only dev boxes device selection never resolves to CUDA and no flock is taken, so tests
    are hermetic by accident. On a GPU host (e.g. wenceslaus) the production path activates for
    tests that fake only the model seam, and without this override every such test would flock the
    REAL user cache lock (``~/.cache/dakp/gpu-locks/cuda-0.lock``) — contending with live builds
    and, worse, with other xdist workers: one holder parked the whole remote gate for 4 h (41
    serialized one-hour ``GpuLockTimeoutError`` ceilings). A per-test directory keeps workers
    independent of each other and of the host.
    """
    monkeypatch.setenv("DAKP_GPU_LOCK_DIR", str(tmp_path_factory.mktemp("gpu-locks")))
