"""Persistent NER mention cache — Python client for the ``dakp-nercache`` Go server.

GLiNER mining is the slowest pipeline step (~20+ min per full DAG run) and its results are
pure functions of ``(model, backend config, text)``, so mined mentions are cached in a
Pebble key/value store (``go/internal/nercache``) served by the ``dakp-nercache`` binary
over localhost HTTP. The store lives at ``<workdir>/cache/ner/``; the server listens on
127.0.0.1 with an ephemeral port and publishes ``<workdir>/cache/ner/server.json``
(``{"pid", "port"}``) for discovery.

Cache keys are 64-char lowercase hex BLAKE3 digests — the bare hex of the project's
``b3:<hex>`` convention (see :func:`dakp_pipeline.io.content_hash.hash_bytes`), WITHOUT the
``b3:`` prefix; the Go side treats keys as opaque hex strings. Values are JSON lists of
``Mention.to_dict()`` payloads, stored verbatim so a hit round-trips byte-identically.

Everything here degrades to a **no-op** when the server/binary is unavailable: offline
tests and CPU-only runs need zero setup, and a missing cache must never fail a DAG run.
The offline gazetteer backend is never cached (see :func:`ner_cache_material`).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from dakp_pipeline.io.content_hash import digest_dirname, hash_bytes
from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.ner.lexical import Mention
from dakp_pipeline.ner.ner import DiseaseNER

#: Environment variable overriding the server binary location (wins over all defaults).
BINARY_ENV_VAR = "DAKP_NERCACHE_BIN"
#: Cache directory (Pebble store + server.json), relative to the workdir.
CACHE_DIRNAME = ("cache", "ner")
#: Per-request HTTP timeout; the server is localhost so anything longer means trouble.
_REQUEST_TIMEOUT_SECONDS = 30.0
_HEALTH_TIMEOUT_SECONDS = 1.0
#: Startup budget waiting for a spawned server to publish server.json and answer /health.
_START_WAIT_SECONDS = 5.0
_START_POLL_SECONDS = 0.1
#: How long close() waits for SIGTERM to stop a server this instance started.
_STOP_WAIT_SECONDS = 3.0


def normalize_key_text(text: str) -> str:
    """Canonical text for cache keys: strip, then collapse internal whitespace runs to one space.

    Only whitespace is folded — case and punctuation are significant, because the NER
    backend itself is case/punctuation-sensitive (mention offsets index the raw text).
    """
    return " ".join(text.split())


def mention_key(model_id: str, model_b3: str, config_fingerprint: str, text: str) -> str:
    """The 64-char hex BLAKE3 cache key for one ``(model, config, text)`` triple.

    ``model_b3`` may be given with or without the ``b3:`` prefix (normalized away). The
    keyed string is ``"<model_id>|<model_b3>|<config_fingerprint>|<normalized text>"`` —
    see :func:`normalize_key_text` for the text normalization.
    """
    canonical = f"{model_id}|{digest_dirname(model_b3)}|{config_fingerprint}|{normalize_key_text(text)}"
    return digest_dirname(hash_bytes(canonical.encode("utf-8")))


def _jsonable(value: Any) -> Any:
    """Reduce a ``DiseaseNER._config()`` value to deterministic JSON-serializable form."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if hasattr(value, "items") and callable(value.items):  # Gazetteer: sorted (term, type) pairs
        items: Any = value.items()
        return [[str(key), _jsonable(val)] for key, val in items]
    return str(value)


def config_fingerprint(ner: DiseaseNER) -> str:
    """64-hex BLAKE3 fingerprint of the backend's serializable construction config."""
    canonical = json.dumps(_jsonable(ner._config()), sort_keys=True, separators=(",", ":"))
    return digest_dirname(hash_bytes(canonical.encode("utf-8")))


def ner_cache_material(ner: DiseaseNER) -> tuple[str, str, str] | None:
    """The ``(model_id, model_b3, config_fingerprint)`` key material for a backend, or None.

    The **offline gazetteer backend is deliberately not cached** (returns ``None``): it is
    deterministic, CPU-cheap, and keyed by nothing heavier than its embedded/fixture
    gazetteer, so a cache would add a server dependency for no measurable gain. For the
    production backend the model content hash comes from the model-cache manifest
    (:func:`~dakp_pipeline.ner.model_cache.lookup_model` — a manifest read, never a
    re-hash, model load, or download); ``None`` when that cannot be resolved, in
    which case callers mine without caching.
    """
    if ner._offline:
        return None
    from dakp_pipeline.ner.model_cache import lookup_model

    try:
        ref = lookup_model(ner._model_id, cache_dir=ner._cache_dir, workdir=ner._workdir)
    except Exception as exc:  # never let cache key material break mining
        logger.warning("mention_cache: model ref for {} unavailable ({}); mining without cache", ner._model_id, type(exc).__name__)
        return None
    if ref is None:
        # Cold model cache: mining without caching, not a model download as a side effect
        # of building a cache key.
        return None
    return ner._model_id, digest_dirname(ref.b3), config_fingerprint(ner)


class MentionCache:
    """Lazy, fail-soft client for the ``dakp-nercache`` server.

    The server is located/started on first use: a live server from an existing
    ``server.json`` (pid alive + ``/health`` answering) is reused; otherwise the binary is
    spawned (:data:`BINARY_ENV_VAR` override, else ``<workdir>/bin/dakp-nercache``, else
    ``dakp-nercache`` on PATH) and given up to ~5s to come up. When no server can be
    reached the instance logs one warning and becomes a no-op — :meth:`get_many` returns
    ``{}`` and :meth:`put_many` does nothing. No method ever raises for cache reasons.
    """

    def __init__(self, workdir: Path | str, binary: Path | str | None = None) -> None:
        self._workdir = Path(workdir)
        self._binary = str(binary) if binary is not None else None
        self._base_url: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None  # set only when WE started the server
        self._resolved = False  # lazy resolution attempted (success or failure)
        self._warned = False

    # -- server location/startup -------------------------------------------------
    def _server_file(self) -> Path:
        return self._workdir.joinpath(*CACHE_DIRNAME, "server.json")

    def _warn_once(self, message: str, *args: Any) -> None:
        if not self._warned:
            logger.warning("mention_cache: " + message, *args)
            self._warned = True

    def _healthy(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=_HEALTH_TIMEOUT_SECONDS) as response:
                return response.status == 200
        except Exception:
            return False

    def _read_live_server(self) -> str | None:
        """Base URL of an already-running server from server.json, else None."""
        try:
            data = json.loads(self._server_file().read_text("utf-8"))
            pid, port = int(data["pid"]), int(data["port"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return f"http://127.0.0.1:{port}" if self._healthy(port) else None

    def _find_binary(self) -> str | None:
        if self._binary:
            return self._binary
        env = os.environ.get(BINARY_ENV_VAR, "").strip()
        if env:
            return env
        local = self._workdir / "bin" / "dakp-nercache"
        if local.is_file():
            return str(local)
        return shutil.which("dakp-nercache")

    def _ensure(self) -> str | None:
        """The server's base URL, locating/starting it on first call; None = no-op mode."""
        if self._resolved:
            return self._base_url
        self._resolved = True
        existing = self._read_live_server()
        if existing is not None:
            self._base_url = existing
            stats(logger, "mention_cache", reused=True, url=existing)
            return self._base_url
        binary = self._find_binary()
        if binary is None:
            self._warn_once("no dakp-nercache binary found ({} / <workdir>/bin / PATH); caching disabled", BINARY_ENV_VAR)
            return None
        try:
            self._proc = subprocess.Popen(
                [binary, "--workdir", str(self._workdir)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self._warn_once("cannot start {} ({}); caching disabled", binary, exc)
            return None
        deadline = time.monotonic() + _START_WAIT_SECONDS
        while time.monotonic() < deadline:
            started = self._read_live_server()
            if started is not None:
                self._base_url = started
                stats(logger, "mention_cache", started=True, url=started)
                return self._base_url
            if self._proc.poll() is not None:
                break  # exited already (e.g. DB locked by a stale server that lost its file)
            time.sleep(_START_POLL_SECONDS)
        self._warn_once("dakp-nercache did not come up within {}s; caching disabled", _START_WAIT_SECONDS)
        self._stop_owned()
        return None

    # -- HTTP boundary (monkeypatch-friendly) --------------------------------------
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """POST ``payload`` to the server, returning the decoded body; None on any failure."""
        base = self._ensure()
        if base is None:
            return None
        request = urllib.request.Request(
            f"{base}{path}", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self._warn_once("{} request failed ({}); caching disabled", path, type(exc).__name__)
            return None
        return body if isinstance(body, dict) else None

    # -- public API -----------------------------------------------------------------
    def get_many(self, keys: list[str]) -> dict[str, list[Mention]]:
        """Cached mentions for ``keys`` that hit; missing/failed keys are simply absent."""
        if not keys:
            return {}
        body = self._post("/batch_get", {"keys": keys})
        if body is None:
            return {}
        hits = body.get("hits")
        if not isinstance(hits, dict):
            return {}
        out: dict[str, list[Mention]] = {}
        for key, value in hits.items():
            if isinstance(value, list):
                out[str(key)] = [Mention.from_dict(item) for item in value]
        return out

    def put_many(self, items: dict[str, list[Mention]]) -> None:
        """Store ``{key: mentions}``; a no-op when the cache is unavailable."""
        if not items:
            return
        payload = {key: [mention.to_dict() for mention in mentions] for key, mentions in items.items()}
        self._post("/batch_put", {"items": payload})

    def _stop_owned(self) -> None:
        """SIGTERM the server process, but only when THIS instance started it."""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=_STOP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.kill(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=1.0)

    def close(self) -> None:
        """Release the client; stops the server only if this instance spawned it."""
        self._stop_owned()

    def __enter__(self) -> MentionCache:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ["BINARY_ENV_VAR", "CACHE_DIRNAME", "MentionCache", "config_fingerprint", "mention_key", "ner_cache_material", "normalize_key_text"]
