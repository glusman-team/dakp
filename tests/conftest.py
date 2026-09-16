"""Test-wide configuration shared by every suite (unit + integration + eval).

Forces the stdlib ``urllib`` download path for the whole suite so the offline tests that
monkeypatch ``urllib.request.urlopen`` — notably ``tests/integration/test_prod_smoke.py``,
which exercises the REAL fetcher download branches through that seam — stay deterministic and
network-free even though the bundled aria2c binary is installed. The aria2c code paths are
covered directly by ``tests/unit/test_downloader.py``, which opts back in per test.

Also arms coverage measurement inside ``spawn``-started child processes (see
:func:`_enable_subprocess_coverage`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _enable_subprocess_coverage() -> None:
    """Point ``COVERAGE_PROCESS_START`` at this repo's config so child processes are measured.

    The per-GPU NER workers run under the ``spawn`` start method, so a child is a brand-new
    interpreter that inherits no coverage tracer. The installed ``coverage`` ``.pth`` file calls
    ``coverage.process_startup()`` at interpreter start, but ONLY when this variable is set, and
    it must be set before any child is spawned (it is inherited through the environment).
    Pairs with ``parallel``/``concurrency`` in ``[tool.coverage.run]``.
    """
    os.environ.setdefault("COVERAGE_PROCESS_START", str(Path(__file__).resolve().parents[1] / "pyproject.toml"))


_enable_subprocess_coverage()


@pytest.fixture(autouse=True)
def _force_urllib_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to the urllib download backend (aria2c off).

    Individual tests re-enable aria2c with ``monkeypatch.setenv("DAKP_ARIA2", "1")`` or
    ``monkeypatch.delenv("DAKP_ARIA2", raising=False)``.
    """
    monkeypatch.setenv("DAKP_ARIA2", "0")
