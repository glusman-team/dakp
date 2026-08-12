"""aria2c-accelerated downloads with a stdlib ``urllib`` fallback.

The PyPI ``aria2`` wheel bundles a statically-linked ``aria2c`` binary (no separate system
install, no ``sudo``, no PATH change), so :func:`download` gets multi-connection / segmented
downloads out of the box. It prefers aria2c and falls back to stdlib ``urllib`` streaming when
aria2c is unavailable OR fails at runtime, so the pipeline still works — just slower — on any
host. FAERS (≈80 quarterly zips) and Drugs@FDA route their bulk downloads through
:func:`download`; DailyMed keeps its own conditional-GET ``urllib`` path (304 / ETag handling its
freshness gate depends on — see :mod:`dakp_pipeline.sources.dailymed`).

``DAKP_ARIA2=0`` (or ``false`` / empty) forces the urllib fallback everywhere — an escape hatch
for a host where the bundled binary misbehaves. The whole offline test suite sets it (see
``tests/conftest.py``) so the tests that monkeypatch ``urllib.request.urlopen`` (notably the
real-fetcher smoke test) stay deterministic and network-free; aria2c is exercised by the
dedicated ``tests/unit/test_downloader.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from dakp_pipeline.logging_setup import logger, stats

#: aria2c per-server connection / segment parallelism (the aria2c ``-x`` maximum of 16).
_DEFAULT_CONNECTIONS = 16
#: aria2c per-operation network timeout (seconds); aria2c retries up to :data:`_MAX_TRIES` on stall.
_DEFAULT_TIMEOUT = 120.0
_MAX_TRIES = 5
#: Smallest aria2c segment (1 MiB) — stops tiny files from being over-split.
_MIN_SPLIT = 1 << 20
#: ``urllib`` streaming chunk (1 MiB) — a whole quarter zip never sits in memory.
_STREAM_CHUNK = 1 << 20
#: User-Agent sent on every download (mirrors the legacy Perl fetchers + Drugs@FDA's expectation).
_USER_AGENT = "dakp-pipeline/0.1"


def _bundled_aria2c() -> Path | None:
    """Path to the aria2c binary bundled by the PyPI ``aria2`` wheel, or ``None`` if absent.

    Imported lazily so this module loads on hosts without the wheel (the stdlib fallback then
    applies). The wheel exposes the binary path as ``aria2c.ARIA2C`` (a :class:`pathlib.Path`).
    """
    try:
        from aria2c import ARIA2C  # PyPI `aria2` wheel — bundles a static aria2c binary
    except Exception:  # pragma: no cover - wheel not installed (CI/dev installs it)
        return None
    return Path(ARIA2C) if ARIA2C else None


def resolve_aria2c() -> Path | None:
    """Best-effort aria2c executable: bundled wheel binary, then a system ``aria2c`` on PATH.

    Returns ``None`` (callers fall back to stdlib ``urllib``) when ``DAKP_ARIA2`` is
    ``0``/``false``/empty, or when no aria2c is available. Never raises.
    """
    if os.environ.get("DAKP_ARIA2", "1") in ("0", "false", ""):
        return None
    bundled = _bundled_aria2c()
    if bundled is not None and bundled.exists():
        return bundled
    found = shutil.which("aria2c")
    return Path(found) if found else None


def _build_aria2c_args(exe: Path, url: str, dest: Path, *, timeout: float, headers: dict[str, str] | None, connections: int) -> list[str]:
    """Assemble the aria2c command line for a single-file, multi-connection download into ``dest``.

    ``--dir``/``--out`` pin the exact output path; ``--allow-overwrite`` + ``--auto-file-renaming=false``
    make a re-run overwrite cleanly rather than emitting ``.aria2``/``.1`` sidecars; ``--file-allocation=none``
    skips preallocation (fast start, no ``fallocate`` issues on odd filesystems). Range-request
    support is assumed (FDA / NLM servers provide it); aria2c silently falls back to one connection
    when a server rejects ranges.
    """
    headers = headers or {}
    args = [
        str(exe),
        "--no-conf",
        "--console-log-level=error",
        "--summary-interval=0",
        "--file-allocation=none",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        f"--min-split-size={_MIN_SPLIT}",
        f"--max-tries={_MAX_TRIES}",
        "--retry-wait=2",
        f"--timeout={int(timeout)}",
        f"--connect-timeout={min(60, int(timeout))}",
        "--dir",
        str(dest.parent),
        "--out",
        dest.name,
    ]
    user_agent = headers.get("User-Agent")
    if user_agent:
        args += ["--user-agent", user_agent]
    for key, value in headers.items():
        if key.lower() != "user-agent":
            args += ["--header", f"{key}: {value}"]
    args.append(url)
    return args


def aria2_download(
    url: str,
    dest: Path,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    connections: int = _DEFAULT_CONNECTIONS,
    executable: Path | None = None,
) -> Path:
    """Download ``url`` to ``dest`` via aria2c (multi-connection). Raises on aria2c failure.

    aria2c manages its own liveness (per-operation ``--timeout`` + ``--max-tries`` retries), so no
    subprocess wall-clock cap is imposed — a multi-GB download is never killed mid-flight. A
    default :data:`_USER_AGENT` is applied when ``headers`` omits one (some FDA endpoints reject
    the empty default). Failures propagate (:class:`subprocess.CalledProcessError`,
    :class:`subprocess.TimeoutExpired`, :class:`OSError`) so :func:`download` can retry via urllib.
    """
    exe = executable if executable is not None else resolve_aria2c()
    if exe is None:
        msg = "aria2c unavailable (install the `aria2` wheel or aria2c on PATH, and set DAKP_ARIA2!=0)"
        raise RuntimeError(msg)
    if headers is None or "User-Agent" not in headers:
        headers = {**dict(headers or {}), "User-Agent": _USER_AGENT}
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = _build_aria2c_args(exe, url, dest, timeout=timeout, headers=headers, connections=connections)
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dest


def stream_download(url: str, dest: Path, *, timeout: float = _DEFAULT_TIMEOUT, headers: dict[str, str] | None = None) -> Path:
    """Stdlib ``urllib`` streaming fallback: ``url`` -> ``dest`` in :data:`_STREAM_CHUNK` chunks.

    HTTP/URL errors propagate (fail loudly). This is the only download path the offline test
    suite exercises (it monkeypatches ``urllib.request.urlopen``).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=_STREAM_CHUNK)
    return dest


def download(url: str, dest: Path, *, timeout: float = _DEFAULT_TIMEOUT, headers: dict[str, str] | None = None) -> Path:
    """Download ``url`` to ``dest``, preferring aria2c and falling back to ``urllib`` streaming.

    An aria2c failure (nonzero exit, timeout, missing binary at runtime) is logged once as a
    WARNING and retried via the stdlib path, so a transient aria2c hiccup never aborts
    acquisition.
    """
    exe = resolve_aria2c()
    if exe is not None:
        try:
            return aria2_download(url, dest, timeout=timeout, headers=headers, executable=exe)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            stats(logger, "download", level="WARNING", backend="aria2c", failed=True, error=type(exc).__name__, fallback="urllib")
    return stream_download(url, dest, timeout=timeout, headers=headers)


__all__ = ["aria2_download", "download", "resolve_aria2c", "stream_download"]
