"""Unit tests for the idempotent NER model cache (``dakp_pipeline.ner.model_cache``).

The model cache is unchanged by the single-backend refactor; these tests cover its main paths
(default cache-dir resolution, idempotent download, manifest provenance, force/verify behavior,
model-id path sanitization) using an injected fake downloader — no network, no heavy deps imported.
Edge cases live in ``test_ner_model_cache_edge.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline.ner.model_cache import (
    SCHEMA_VERSION,
    default_model_cache_dir,
    ensure_model,
    manifest_path,
    model_root,
    read_manifest,
    write_manifest,
)


def _fake_downloader(calls: list[str], payload: bytes = b"weights"):
    def download(model_id: str, dest: Path) -> None:
        calls.append(model_id)
        (dest / "weights.bin").write_bytes(payload)

    return download


# --- default cache-dir resolution ----------------------------------------------


def test_default_cache_dir_uses_workdir(tmp_path: Path) -> None:
    assert default_model_cache_dir(tmp_path) == tmp_path / "models"


def test_default_cache_dir_honors_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_model_cache_dir() == tmp_path / "dakp" / "models"


def test_default_cache_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert default_model_cache_dir() == Path.home() / ".cache" / "dakp" / "models"


# --- idempotent download + manifest --------------------------------------------


def test_ensure_model_is_idempotent(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert ref1.path.exists()
    assert ref1.manifest.exists()
    assert ref1.b3.startswith("b3:")
    assert calls == ["acme/tiny-ner"]

    ref2 = ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert calls == ["acme/tiny-ner"]  # cache hit: not re-downloaded
    assert ref2.b3 == ref1.b3
    assert ref2.path == ref1.path


def test_ensure_model_writes_manifest(tmp_path: Path) -> None:
    calls: list[str] = []
    ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    data = read_manifest(manifest_path(model_root(tmp_path, "acme/tiny-ner")))
    assert data is not None
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["model_id"] == "acme/tiny-ner"
    assert data["source"] == "huggingface"
    assert data["b3"].startswith("b3:")


def test_ensure_model_force_redownloads(tmp_path: Path) -> None:
    calls: list[str] = []
    ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls), force=True)
    assert len(calls) == 2


def test_ensure_model_verify_detects_drift(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"original"))
    # Size-changing tamper: the default fast check (file count + total bytes) catches it.
    (ref1.path / "weights.bin").write_bytes(b"tampered-content")
    ref2 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"restored"))
    assert calls == ["m/x", "m/x"]  # drifted content triggered a re-download
    assert ref2.b3 != ref1.b3


def test_ensure_model_manifest_records_content_stats(tmp_path: Path) -> None:
    calls: list[str] = []
    ref = ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    data = read_manifest(ref.manifest)
    assert data is not None
    assert data["file_count"] == 1
    assert data["total_bytes"] == len(b"weights")


def test_ensure_model_stat_mismatch_falls_back_to_hash_and_redownloads(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"original"))
    (ref1.path / "extra.bin").write_bytes(b"stowaway")  # file-count drift, same payload bytes elsewhere
    ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"restored"))
    assert calls == ["m/x", "m/x"]  # stat mismatch -> full tree hash -> drift -> re-download


def test_ensure_model_backfills_stats_on_a_legacy_manifest(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    # Simulate a manifest written before the stats fields existed.
    data = read_manifest(ref1.manifest)
    assert data is not None
    del data["file_count"], data["total_bytes"]
    write_manifest(ref1.manifest, data)

    ref2 = ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert calls == ["acme/tiny-ner"]  # full-hash once, still a cache hit (no re-download)
    assert ref2.b3 == ref1.b3
    backfilled = read_manifest(ref1.manifest)
    assert backfilled is not None
    assert backfilled["file_count"] == 1
    assert backfilled["total_bytes"] == len(b"weights")


def test_ensure_model_verify_full_rehashes_and_detects_same_size_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAKP_MODEL_VERIFY", "full")
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"original"))
    ref_hit = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert calls == ["m/x"]  # full-hash verification passes -> cache hit
    assert ref_hit.b3 == ref1.b3

    # Same-size tamper: invisible to the stat check, caught by the full tree hash.
    (ref1.path / "weights.bin").write_bytes(b"tampered")
    ref2 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"restored"))
    assert calls == ["m/x", "m/x"]
    assert ref2.b3 != ref1.b3


def test_ensure_model_sanitizes_model_id_in_path(tmp_path: Path) -> None:
    calls: list[str] = []
    ref = ensure_model("urchade/gliner_small-v2.1", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert "urchade--gliner_small-v2.1" in ref.path.as_posix()
