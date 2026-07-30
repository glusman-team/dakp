"""Edge-case tests for ``dakp_pipeline.ner.model_cache`` (drive to 100% branch coverage).

Targets the uncovered lines: ``read_manifest`` absent/corrupt/non-dict paths, the
``default_downloader`` missing-dep AND successful (fake ``huggingface_hub``) paths, and the
``ensure_model`` re-download branches (corrupt manifest, mismatched provenance, deleted /
drifted content, ``verify=False``, ``workdir`` cache resolution). All offline / no ``[ner]`` extra.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from dakp_pipeline.ner.model_cache import (
    SCHEMA_VERSION,
    NERDependencyError,
    content_dir,
    default_downloader,
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


# --- read_manifest: absent / corrupt / non-dict ---------------------------------


def test_read_manifest_absent_returns_none(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "does_not_exist.json") is None


def test_read_manifest_corrupt_json_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    assert read_manifest(bad) is None


def test_read_manifest_directory_raises_oserror_returns_none(tmp_path: Path) -> None:
    # Passing a directory: path.exists() is True but read_text() raises IsADirectoryError
    # (an OSError) -> swallowed -> None.
    subdir = tmp_path / "manifest.json"
    subdir.mkdir()
    assert read_manifest(subdir) is None


def test_read_manifest_non_dict_json_returns_none(tmp_path: Path) -> None:
    for payload in ("[1, 2, 3]", '"just a string"', "42", "null"):
        path = tmp_path / "manifest.json"
        path.write_text(payload, encoding="utf-8")
        assert read_manifest(path) is None


def test_write_then_read_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    write_manifest(path, {"b": 2, "a": 1})
    data = read_manifest(path)
    assert data == {"a": 1, "b": 2}
    # Sorted keys + trailing newline on disk.
    assert path.read_text("utf-8").endswith("}\n")
    assert path.read_text("utf-8").index('"a"') < path.read_text("utf-8").index('"b"')


# --- default_downloader: missing dep + successful fake --------------------------


def test_default_downloader_raises_clear_error_without_extra(tmp_path: Path) -> None:
    # huggingface_hub is not installed (no [ner] extra) -> NERDependencyError with install cmd.
    assert "huggingface_hub" not in sys.modules
    with pytest.raises(NERDependencyError, match=r"uv sync --extra ner"):
        default_downloader("acme/tiny-ner", tmp_path)


def test_default_downloader_uses_huggingface_hub_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    module = types.ModuleType("huggingface_hub")

    def _snapshot_download(repo_id: str, local_dir: Path) -> None:
        captured["repo_id"] = repo_id
        captured["local_dir"] = local_dir

    module.snapshot_download = _snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    default_downloader("acme/tiny-ner", tmp_path)
    assert captured == {"repo_id": "acme/tiny-ner", "local_dir": tmp_path}


# --- ensure_model: re-download branches -----------------------------------------


def test_ensure_model_verify_false_skips_rehash_and_returns_hit(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"original"))
    # Tamper, then verify=False: the drift is NOT detected -> cache hit, no re-download.
    (ref1.path / "weights.bin").write_bytes(b"tampered")
    ref2 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls), verify=False)
    assert calls == ["m/x"]  # not re-downloaded
    assert ref2.b3 == ref1.b3  # returns the manifest's recorded hash


def test_ensure_model_missing_content_dir_triggers_redownload(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"original"))
    # Delete the content dir but keep the manifest: verify=True sees content.exists() False.
    for child in ref1.path.iterdir():
        child.unlink()
    ref1.path.rmdir()
    ref2 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"restored"))
    assert calls == ["m/x", "m/x"]  # re-downloaded
    assert ref2.path.exists()


def test_ensure_model_corrupt_manifest_triggers_redownload(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    ref1.manifest.write_text("{ corrupt", encoding="utf-8")  # read_manifest -> None
    ref2 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert calls == ["m/x", "m/x"]  # unreadable manifest -> re-download
    assert ref2.b3 == ref1.b3


def test_ensure_model_mismatched_provenance_triggers_redownload(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    # Rewrite the manifest with a DIFFERENT model_id -> provenance mismatch -> re-download.
    data = json.loads(ref1.manifest.read_text("utf-8"))
    data["model_id"] = "m/other"
    ref1.manifest.write_text(json.dumps(data), encoding="utf-8")
    ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert calls == ["m/x", "m/x"]

    # A mismatched source likewise forces a re-download.
    data2 = json.loads(ref1.manifest.read_text("utf-8"))
    data2["source"] = "somewhere-else"
    ref1.manifest.write_text(json.dumps(data2), encoding="utf-8")
    ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert calls == ["m/x", "m/x", "m/x"]


def test_ensure_model_resolves_cache_from_workdir(tmp_path: Path) -> None:
    calls: list[str] = []
    workdir = tmp_path / "work"
    ref = ensure_model("m/x", workdir=workdir, downloader=_fake_downloader(calls))
    # cache_dir defaults to <workdir>/models.
    assert default_model_cache_dir(workdir) == workdir / "models"
    assert str(ref.path).startswith(str(workdir / "models"))
    assert ref.manifest.exists()


def test_ensure_model_cache_dir_wins_over_workdir(tmp_path: Path) -> None:
    calls: list[str] = []
    cache_dir = tmp_path / "explicit-cache"
    workdir = tmp_path / "work"
    ref = ensure_model("m/x", cache_dir=cache_dir, workdir=workdir, downloader=_fake_downloader(calls))
    assert str(ref.path).startswith(str(cache_dir))
    assert not (workdir / "models").exists()


def test_ensure_model_manifest_records_provenance(tmp_path: Path) -> None:
    calls: list[str] = []
    ref = ensure_model("acme/tiny-ner", source="huggingface", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    data = read_manifest(ref.manifest)
    assert data is not None
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["model_id"] == "acme/tiny-ner"
    assert data["source"] == "huggingface"
    assert data["b3"] == ref.b3
    assert "retrieved_at" in data
    # model_root / content_dir / manifest_path helpers agree with the ref.
    root = model_root(tmp_path, "acme/tiny-ner")
    assert content_dir(root) == ref.path
    assert manifest_path(root) == ref.manifest
