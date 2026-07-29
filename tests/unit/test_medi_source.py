"""Unit tests for the MEDI contraindication list fetcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import read_manifest
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import medi

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(tmp_path: Path, *, profile: str = "mock", params: dict[str, object] | None = None) -> TaskContext:
    return TaskContext(profile=profile, workdir=tmp_path / "work", fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params=params or {})


def _workdir_created(tmp_path: Path) -> Path:
    wd = tmp_path / "work"
    Workdir(wd).create()
    return wd


# --- mock profile ---------------------------------------------------------------


def test_mock_fetch_returns_content_addressed_fixture_ref(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    refs = medi.fetch(ctx)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.blake3.startswith("b3:")
    assert ref.uri.exists()
    # Stored under the content-addressed by-hash tree, original filename preserved.
    assert "by-hash" in ref.uri.parts
    assert ref.uri.name == "medi_contraindications.tsv"
    assert ref.media_type == "text/tab-separated-values"


def test_mock_fetch_blake3_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ref_a = medi.fetch(ctx)[0]
    ref_b = medi.fetch(ctx)[0]
    # Same fixture bytes -> same BLAKE3 identity -> identical stored path.
    assert ref_a.blake3 == ref_b.blake3
    assert ref_a.uri == ref_b.uri


def test_mock_fetch_captures_version_in_alias_and_manifest(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ref = medi.fetch(ctx)[0]
    assert ref.manifest is not None
    manifest = read_manifest(ref.manifest)
    # Operation config_hash deterministically encodes the pinned mock version.
    assert manifest.operation is not None
    assert manifest.operation.name == "fetch_medi"
    pinned = hash_version(medi.DEFAULT_MEDI_VERSION)
    assert manifest.operation.config_hash == pinned
    # Human-readable alias encodes the version and resolves to the same artifact id.
    alias_dir = Workdir(ctx.workdir).aliases / "medi"
    alias_id = (alias_dir / f"contraindicationList-{medi.DEFAULT_MEDI_VERSION}").read_text()
    assert alias_id == ref.blake3
    # Fixture provenance (not a network retrieval).
    assert manifest.source.url == "fixture:medi/medi_contraindications.tsv"
    assert manifest.source.retrieved_at is None


def test_mock_fetch_respects_overridden_version(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, params={"medi_version": "1.4.1"})
    ref = medi.fetch(ctx)[0]
    assert ref.manifest is not None
    manifest = read_manifest(ref.manifest)
    assert manifest.operation is not None
    assert manifest.operation.config_hash == hash_version("1.4.1")


def test_mock_fetch_requires_fixture_root(tmp_path: Path) -> None:
    ctx = TaskContext(profile="mock", workdir=tmp_path / "work", fixture_root=None, threads=1, memory_budget_gb=1, params={})
    with pytest.raises(ValueError, match="fixture_root"):
        medi.fetch(ctx)


def test_mock_fetch_raises_on_missing_fixture(tmp_path: Path) -> None:
    ctx = TaskContext(profile="mock", workdir=tmp_path / "work", fixture_root=tmp_path / "nope", threads=1, memory_budget_gb=1, params={})
    with pytest.raises(FileNotFoundError, match=r"fixture not found|MEDI contraindication fixture"):
        medi.fetch(ctx)


# --- real profile (monkeypatched download seam) ---------------------------------


def test_real_fetch_downloads_xlsx_via_monkeypatched_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _workdir_created(tmp_path)
    ctx = _ctx(tmp_path, profile="sample", params={"medi_version": "1.4.1"})

    downloaded: dict[str, str] = {}

    def fake_download(url: str, dest: Path) -> Path:
        downloaded["url"] = url
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PK\x03\x04pretend-xlsx")  # xlsx magic + payload
        return dest

    monkeypatch.setattr(medi, "http_download", fake_download)

    ref = medi.fetch(ctx)[0]

    # The versioned release URL was requested exactly once.
    expected_url = "https://github.com/everycure-org/matrix-indication-list/releases/download/v1.4.1/contraindicationList-1.4.1.xlsx"
    assert downloaded["url"] == expected_url
    # Stored as the xlsx release asset with network provenance.
    assert ref.uri.name == "contraindicationList-1.4.1.xlsx"
    assert ref.media_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert ref.manifest is not None
    manifest = read_manifest(ref.manifest)
    assert manifest.source.url == downloaded["url"]
    assert manifest.source.retrieved_at is not None  # real retrieval timestamped


def test_real_fetch_accepts_explicit_url_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _workdir_created(tmp_path)
    ctx = _ctx(tmp_path, profile="sample", params={"medi_version": "9.9.9", "medi_url": "https://example.test/cl.xlsx"})

    def fake_download(url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PK\x03\x04x")
        return dest

    monkeypatch.setattr(medi, "http_download", fake_download)
    ref = medi.fetch(ctx)[0]
    assert ref.manifest is not None
    assert read_manifest(ref.manifest).source.url == "https://example.test/cl.xlsx"


def test_real_fetch_latest_version_without_url_raises(tmp_path: Path) -> None:
    # No monkeypatch: real path with unresolved "latest" must fail loudly, not network.
    ctx = _ctx(tmp_path, profile="sample")  # version defaults to "latest"
    with pytest.raises(NotImplementedError, match="latest"):
        medi.fetch(ctx)


# --- protocol / monkeypatchability ----------------------------------------------


def test_fetch_is_module_level_monkeypatchable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = ArtifactRef(uri=tmp_path / "x", blake3="b3:deadbeef", media_type="text/tab-separated-values")

    monkeypatch.setattr(medi, "fetch", lambda ctx: [sentinel])
    ctx = _ctx(tmp_path)
    assert medi.fetch(ctx) == [sentinel]


def test_fetcher_satisfies_fetcher_protocol() -> None:
    from dakp_pipeline.io.contracts import Fetcher

    assert isinstance(medi.MEDIFetcher(), Fetcher)


# --- helper ---------------------------------------------------------------------


def hash_version(version: str) -> str:
    from dakp_pipeline.io.content_hash import hash_bytes

    return hash_bytes(version.encode("utf-8"))
