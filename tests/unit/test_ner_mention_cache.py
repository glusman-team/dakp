"""Unit tests for the persistent NER mention cache client (``dakp_pipeline.ner.mention_cache``).

Covers the lossless ``Mention`` JSON round-trip the cache depends on, cache-key stability and
invalidation (a model swap — id OR content hash — must change every key), the lazy key-material
resolution from the on-disk model-cache manifest, and the fail-soft ``MentionCache`` behavior
(no server => no-op, never raises). A threaded stdlib HTTP server fakes ``dakp-nercache`` for
the client protocol round-trip — the real server is tested in Go (``go/internal/nercache``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline.ner import mention_cache, model_cache
from dakp_pipeline.ner.mention_cache import BINARY_ENV_VAR, MentionCache, config_fingerprint, mention_key, ner_cache_material, normalize_key_text
from dakp_pipeline.ner.ner import DiseaseNER, Mention

_MODEL_ID = "gliner-community/gliner_large-v2.5"
_MODEL_B3 = "b3:" + "ab" * 32
_FINGERPRINT = "cd" * 32


def _mention(text: str = "asthma") -> Mention:
    return Mention(text=text, start=0, end=len(text), type="disease", score=0.9, normalized=text, notes="n", section="ind")


# --- Mention JSON round-trip -------------------------------------------------------


def test_mention_to_dict_from_dict_round_trip_is_lossless() -> None:
    mention = _mention()
    restored = Mention.from_dict(json.loads(json.dumps(mention.to_dict())))
    assert restored == mention
    assert restored.to_dict() == mention.to_dict()


def test_mention_from_dict_defaults_optional_fields() -> None:
    """Cache values written before notes/section existed still load."""
    restored = Mention.from_dict({"text": "asthma", "start": 0, "end": 6, "type": "disease", "score": 0.5})
    assert restored == Mention(text="asthma", start=0, end=6, type="disease", score=0.5)


# --- mention_key ---------------------------------------------------------------------


def test_mention_key_is_stable_64_hex() -> None:
    key1 = mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma")
    key2 = mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma")
    assert key1 == key2
    assert len(key1) == 64
    assert all(c in "0123456789abcdef" for c in key1)


def test_mention_key_collapses_whitespace() -> None:
    """Strip + internal whitespace runs folded to one space: variants share a key."""
    base = mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma")
    assert mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "  severe   asthma\n") == base
    assert mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe\tasthma") == base


def test_mention_key_is_case_and_punctuation_sensitive() -> None:
    """Only whitespace folds — the NER backend is case/punctuation-sensitive."""
    base = mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma")
    assert mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "Severe asthma") != base
    assert mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma.") != base


def test_mention_key_accepts_b3_with_or_without_prefix() -> None:
    """The ``b3:`` prefix is normalized away (keys are bare hex by convention)."""
    assert mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "x") == mention_key(_MODEL_ID, _MODEL_B3.removeprefix("b3:"), _FINGERPRINT, "x")


def test_mention_key_model_swap_invalidates_everything() -> None:
    """Changing the model id OR the checkpoint content hash changes every key."""
    base = mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma")
    assert mention_key("gliner-community/gliner_small-v2.1", _MODEL_B3, _FINGERPRINT, "severe asthma") != base
    assert mention_key(_MODEL_ID, "b3:" + "ef" * 32, _FINGERPRINT, "severe asthma") != base
    assert mention_key(_MODEL_ID, _MODEL_B3, "99" * 32, "severe asthma") != base


def test_normalize_key_text() -> None:
    assert normalize_key_text("  a \n b\tc  ") == "a b c"


# --- config fingerprint + key material -----------------------------------------------


def test_config_fingerprint_is_stable_and_config_sensitive(tmp_path: Path) -> None:
    ner = DiseaseNER(offline=False, model_id=_MODEL_ID, cache_dir=tmp_path)
    assert config_fingerprint(ner) == config_fingerprint(ner)
    other = DiseaseNER(offline=False, model_id=_MODEL_ID, cache_dir=tmp_path, threshold=0.01)
    assert config_fingerprint(other) != config_fingerprint(ner)


def test_ner_cache_material_offline_backend_is_never_cached() -> None:
    assert ner_cache_material(DiseaseNER()) is None


def test_ner_cache_material_reads_manifest_without_downloading(tmp_path: Path) -> None:
    """Key material comes from the on-disk model-cache manifest — no model load, no download."""
    calls: list[str] = []

    def fake_downloader(model_id: str, dest: Path) -> None:
        calls.append(model_id)
        (dest / "weights.bin").write_bytes(b"weights")

    ref = model_cache.ensure_model(_MODEL_ID, cache_dir=tmp_path, downloader=fake_downloader)
    assert calls == [_MODEL_ID]

    ner = DiseaseNER(offline=False, model_id=_MODEL_ID, cache_dir=tmp_path)
    material = ner_cache_material(ner)
    assert material is not None
    model_id, model_b3, fingerprint = material
    assert model_id == _MODEL_ID
    assert model_b3 == ref.b3.removeprefix("b3:")
    assert fingerprint == config_fingerprint(ner)
    assert calls == [_MODEL_ID]  # the manifest hit never re-invoked the downloader
    assert ner._model is None  # and no GLiNER model was loaded


# --- MentionCache fail-soft behavior -----------------------------------------------


def test_mention_cache_without_any_server_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No server.json, no binary anywhere: one warning, then get/put silently do nothing."""
    monkeypatch.delenv(BINARY_ENV_VAR, raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    cache = MentionCache(tmp_path)
    assert cache.get_many(["ab" * 32]) == {}
    cache.put_many({"ab" * 32: [_mention()]})  # must not raise
    cache.close()


def test_mention_cache_missing_binary_path_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An explicit binary override that does not exist fails soft, never raises."""
    monkeypatch.delenv(BINARY_ENV_VAR, raising=False)
    cache = MentionCache(tmp_path, binary=tmp_path / "nope" / "dakp-nercache")
    assert cache.get_many(["ab" * 32]) == {}
    cache.close()


# --- MentionCache against a fake server --------------------------------------------

_FAKE_STORE: dict[str, list[dict[str, object]]] = {}


class _FakeNercacheHandler(BaseHTTPRequestHandler):
    """The ``dakp-nercache`` wire protocol over an in-memory dict."""

    def _respond(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond({"ok": True})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/batch_get":
            self._respond({"hits": {key: _FAKE_STORE[key] for key in body["keys"] if key in _FAKE_STORE}})
        elif self.path == "/batch_put":
            _FAKE_STORE.update(body["items"])
            self._respond({"ok": True})
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def fake_server(tmp_path: Path):
    """A live fake ``dakp-nercache`` discovered via ``<workdir>/cache/ner/server.json``."""
    _FAKE_STORE.clear()
    server = HTTPServer(("127.0.0.1", 0), _FakeNercacheHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server_dir = tmp_path / "cache" / "ner"
    server_dir.mkdir(parents=True)
    (server_dir / "server.json").write_text(json.dumps({"pid": os.getpid(), "port": server.server_address[1]}), encoding="utf-8")
    yield tmp_path
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def test_mention_cache_round_trip_via_live_server(fake_server: Path) -> None:
    """A live server from server.json is reused; mentions round-trip losslessly over HTTP."""
    cache = MentionCache(fake_server)
    key = mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma")
    mention = _mention()
    with cache:
        assert cache.get_many([key]) == {}  # miss
        cache.put_many({key: [mention]})
        hits = cache.get_many([key, "ff" * 32])
    assert hits == {key: [mention]}
    assert hits[key][0] == mention  # exact field-for-field equality


def test_mention_cache_reuses_server_without_spawning(fake_server: Path) -> None:
    """The client must not spawn a second server when server.json points at a live one."""
    cache = MentionCache(fake_server)
    assert cache.get_many(["ab" * 32]) == {}
    assert cache._proc is None  # reused, not spawned — close() must not kill anything
    cache.close()


def test_fake_server_unknown_paths_404(fake_server: Path) -> None:
    """The fake server rejects unknown routes (keeps the test double honest + covered)."""
    import urllib.error
    import urllib.request

    port = json.loads((fake_server / "cache" / "ner" / "server.json").read_text("utf-8"))["port"]
    base = f"http://127.0.0.1:{port}"
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{base}/nope", timeout=5)
    assert excinfo.value.code == 404
    request = urllib.request.Request(f"{base}/nope", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=5)
    assert excinfo.value.code == 404


# --- fail-soft branches (no server may ever break a run) --------------------------


def test_jsonable_falls_back_to_str_for_exotic_values() -> None:
    """Config values that are neither scalar/Path/dict/Gazetteer stringify deterministically."""
    assert mention_cache._jsonable((1, 2)) == "(1, 2)"
    assert mention_cache._jsonable(Path("/x/y")) == "/x/y"


def test_ner_cache_material_degrades_when_the_model_ref_is_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A model-cache lookup failure (corrupt manifest, unreadable dir) means: mine without cache."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("corrupt manifest")

    monkeypatch.setattr(model_cache, "lookup_model", boom)
    ner = DiseaseNER(offline=False, model_id=_MODEL_ID, cache_dir=tmp_path)
    assert ner_cache_material(ner) is None


def test_ner_cache_material_cold_cache_mines_without_caching(tmp_path: Path) -> None:
    """A cold model cache (no manifest) yields no key material — never a model download."""
    ner = DiseaseNER(offline=False, model_id=_MODEL_ID, cache_dir=tmp_path)
    assert ner_cache_material(ner) is None
    assert ner._model is None  # and no GLiNER model was loaded


def test_warn_once_logs_only_the_first_warning(tmp_path: Path) -> None:
    cache = MentionCache(tmp_path)
    cache._warn_once("first")
    cache._warn_once("second")  # already warned -> silent


def test_healthy_is_false_when_the_port_does_not_answer(tmp_path: Path) -> None:
    """Connection refused (or any urlopen error) reads as unhealthy, never raises."""
    assert MentionCache(tmp_path)._healthy(1) is False


def test_read_live_server_rejects_a_dead_pid_and_an_unhealthy_port(tmp_path: Path) -> None:
    cache = MentionCache(tmp_path)
    server_dir = tmp_path / "cache" / "ner"
    server_dir.mkdir(parents=True)
    server_file = server_dir / "server.json"

    # A reaped pid fails the os.kill(pid, 0) liveness probe.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    server_file.write_text(json.dumps({"pid": proc.pid, "port": 1}), encoding="utf-8")
    assert cache._read_live_server() is None

    # A live pid whose port does not answer /health is equally unusable.
    server_file.write_text(json.dumps({"pid": os.getpid(), "port": 1}), encoding="utf-8")
    assert cache._read_live_server() is None


def test_find_binary_resolution_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Explicit override > DAKP_NERCACHE_BIN > <workdir>/bin > PATH."""
    assert MentionCache(tmp_path, binary="/opt/dakp-nercache")._find_binary() == "/opt/dakp-nercache"

    monkeypatch.setenv(BINARY_ENV_VAR, "/env/dakp-nercache")
    assert MentionCache(tmp_path)._find_binary() == "/env/dakp-nercache"
    monkeypatch.delenv(BINARY_ENV_VAR)

    local = tmp_path / "bin" / "dakp-nercache"
    local.parent.mkdir(parents=True)
    local.write_text("#!/bin/sh\n", encoding="utf-8")
    assert MentionCache(tmp_path)._find_binary() == str(local)

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/dakp-nercache")
    assert MentionCache(tmp_path / "other")._find_binary() == "/usr/bin/dakp-nercache"


def test_get_many_empty_keys_never_touches_the_server(tmp_path: Path) -> None:
    cache = MentionCache(tmp_path)
    assert cache.get_many([]) == {}
    assert cache._resolved is False  # no server lookup happened at all


def test_put_many_empty_items_is_a_noop(tmp_path: Path) -> None:
    cache = MentionCache(tmp_path)
    cache.put_many({})
    assert cache._resolved is False


class _StaticPayloadHandler(BaseHTTPRequestHandler):
    """Answers /health 200 and every POST with a fixed payload (possibly malformed)."""

    payload = b"{}"

    def _respond(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._respond(b'{"ok": true}')

    def do_POST(self) -> None:
        self._respond(type(self).payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def static_server(tmp_path: Path):
    """A live ``server.json``-discoverable endpoint answering POSTs with ``payload``."""
    server = HTTPServer(("127.0.0.1", 0), _StaticPayloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server_dir = tmp_path / "cache" / "ner"
    server_dir.mkdir(parents=True)
    (server_dir / "server.json").write_text(json.dumps({"pid": os.getpid(), "port": server.server_address[1]}), encoding="utf-8")
    yield tmp_path
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def test_post_failure_degrades_to_noop(static_server: Path) -> None:
    """An undecodable response body warns once and reads as a cache miss, never an error."""
    _StaticPayloadHandler.payload = b"not json"
    try:
        cache = MentionCache(static_server)
        assert cache.get_many(["ab" * 32]) == {}
        assert cache.get_many(["cd" * 32]) == {}  # still fail-soft on the second failure
    finally:
        _StaticPayloadHandler.payload = b"{}"


def test_get_many_ignores_malformed_hits(static_server: Path) -> None:
    """Non-dict ``hits`` and non-list values are dropped; well-formed hits still load."""
    key = "ab" * 32
    mention = _mention()
    _StaticPayloadHandler.payload = json.dumps({"hits": {key: [mention.to_dict()], "cd" * 32: "bogus"}}).encode("utf-8")
    try:
        assert MentionCache(static_server).get_many([key, "cd" * 32]) == {key: [mention]}
        _StaticPayloadHandler.payload = json.dumps({"hits": "not-a-dict"}).encode("utf-8")
        # A fresh client re-resolves the (still live) server and gets the new payload.
        assert MentionCache(static_server).get_many([key]) == {}
    finally:
        _StaticPayloadHandler.payload = b"{}"


def _write_spawnable_fake_binary(path: Path, port: int) -> Path:
    """An executable 'server' that publishes server.json (its own live pid + ``port``), then sleeps."""
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys, time\n"
        "workdir = pathlib.Path(sys.argv[2])\n"
        "target = workdir / 'cache' / 'ner'\n"
        "target.mkdir(parents=True, exist_ok=True)\n"
        f"(target / 'server.json').write_text(json.dumps({{'pid': os.getpid(), 'port': {port}}}))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_mention_cache_spawns_the_binary_and_stops_it_on_close(static_server: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no live server.json the client spawns the binary, uses it, and SIGTERMs it on close."""
    port = json.loads((static_server / "cache" / "ner" / "server.json").read_text("utf-8"))["port"]
    workdir = tmp_path / "work"
    binary = _write_spawnable_fake_binary(tmp_path / "dakp-nercache", port)
    cache = MentionCache(workdir, binary=binary)

    key = mention_key(_MODEL_ID, _MODEL_B3, _FINGERPRINT, "severe asthma")
    cache.put_many({key: [_mention()]})  # first use spawns
    assert cache._proc is not None
    assert cache.get_many([key]) == {}  # the static payload is {} — the SPAWN itself is what's covered

    proc = cache._proc
    real_wait = proc.wait
    wait_calls = 0

    def flaky_wait(timeout: float | None = None) -> int:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="dakp-nercache", timeout=timeout or 0)
        return real_wait(timeout=timeout)

    monkeypatch.setattr(proc, "wait", flaky_wait)  # force the SIGKILL escalation branch
    cache.close()
    assert proc.poll() is not None  # the spawned server is really gone
    cache.close()  # idempotent: no owned process left to stop


def test_mention_cache_spawn_exit_before_ready_is_a_noop(tmp_path: Path) -> None:
    """A binary that dies before publishing server.json disables caching (one warning, no raise)."""
    binary = tmp_path / "dakp-nercache"
    binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    binary.chmod(0o755)
    cache = MentionCache(tmp_path, binary=binary)
    assert cache.get_many(["ab" * 32]) == {}
    cache.close()


def test_mention_cache_spawn_that_never_publishes_times_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A live binary that never writes server.json exhausts the startup budget -> no-op."""
    monkeypatch.setattr(mention_cache, "_START_WAIT_SECONDS", 0.0)  # expire the wait loop instantly
    binary = tmp_path / "dakp-nercache"
    binary.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    binary.chmod(0o755)
    cache = MentionCache(tmp_path, binary=binary)
    assert cache.get_many(["ab" * 32]) == {}
    cache.close()  # stops the still-running spawned process
