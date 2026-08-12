"""Test-wide configuration shared by every suite (unit + integration + eval).

Forces the stdlib ``urllib`` download path for the whole suite so the offline tests that
monkeypatch ``urllib.request.urlopen`` — notably ``tests/integration/test_prod_smoke.py``,
which exercises the REAL fetcher download branches through that seam — stay deterministic and
network-free even though the bundled aria2c binary is installed. The aria2c code paths are
covered directly by ``tests/unit/test_downloader.py``, which opts back in per test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_urllib_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to the urllib download backend (aria2c off).

    Individual tests re-enable aria2c with ``monkeypatch.setenv("DAKP_ARIA2", "1")`` or
    ``monkeypatch.delenv("DAKP_ARIA2", raising=False)``.
    """
    monkeypatch.setenv("DAKP_ARIA2", "0")
