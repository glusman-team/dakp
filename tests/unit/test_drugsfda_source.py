"""Tests for the Drugs@FDA source fetcher (Milestone 2).

Covers the mock-profile fixture path, idempotent content-addressing, and the real
download path (with the network downloader monkeypatched to serve a local fixture ZIP).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import drugsfda

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_DRUGSFDA_FIXTURE_DIR = _FIXTURE_ROOT / "drugsfda"


def _ctx(profile: str, workdir: Path, *, params: dict[str, object] | None = None) -> TaskContext:
    merged = dict(params) if params else {}
    return TaskContext(profile=profile, workdir=workdir, fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params=merged)


def _write_fixture_zip(tmp_path: Path) -> Path:
    """A ZIP mirroring the real Drugs@FDA data-files layout (Products.txt / etc.)."""
    zip_path = tmp_path / "drugsfda.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in ("drugsfda_products.tsv", "drugsfda_applications.tsv", "drugsfda_submissions.tsv"):
            archive.write(_DRUGSFDA_FIXTURE_DIR / name, arcname=name.replace("drugsfda_", "").capitalize())
    return zip_path


# --- mock profile ---------------------------------------------------------------


def test_mock_fetch_returns_fixture_refs(tmp_path: Path) -> None:
    Workdir(tmp_path / "work").create()
    ctx = _ctx("mock", tmp_path / "work")

    refs = drugsfda.fetch(ctx)

    assert len(refs) == 3
    names = sorted(ref.uri.name for ref in refs)
    assert names == ["drugsfda_applications.tsv", "drugsfda_products.tsv", "drugsfda_submissions.tsv"]
    for ref in refs:
        assert ref.blake3.startswith("b3:")
        assert ref.uri.exists()
        # Mock ingest writes a content-addressed manifest for every fixture.
        assert ref.manifest is not None
        assert ref.manifest.exists()


def test_mock_fetch_is_idempotent(tmp_path: Path) -> None:
    """Re-fetching identical fixtures yields identical artifact ids (cache hit semantics)."""
    Workdir(tmp_path / "work").create()
    ctx = _ctx("mock", tmp_path / "work")

    first = drugsfda.fetch(ctx)
    second = drugsfda.fetch(ctx)

    assert sorted(r.blake3 for r in first) == sorted(r.blake3 for r in second)


# --- real download path (network monkeypatched) --------------------------------


def test_real_fetch_ingests_zip_and_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    fixture_zip = _write_fixture_zip(tmp_path)
    captured: list[str] = []

    def fake_download(url: str, dest: Path, **kwargs: object) -> Path:
        captured.append(url)
        shutil.copyfile(fixture_zip, dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    Workdir(tmp_path / "work").create()
    ctx = _ctx("sample", tmp_path / "work")

    first = drugsfda.fetch(ctx)
    assert len(first) == 1
    ref = first[0]
    assert ref.media_type == "application/zip"
    assert ref.blake3.startswith("b3:")
    assert ref.manifest is not None
    assert ref.manifest.exists()
    # The downloader was actually invoked (no silent fixture fallback for non-mock).
    assert captured == [drugsfda.DRUGSFDA_DATA_FILES_URL]

    # Idempotent: identical bytes hash to the same artifact id on re-download.
    second = drugsfda.fetch(ctx)
    assert second[0].blake3 == ref.blake3


def test_real_fetch_url_overridable_via_params(monkeypatch, tmp_path: Path) -> None:
    fixture_zip = _write_fixture_zip(tmp_path)
    seen: list[str] = []

    def fake_download(url: str, dest: Path, **kwargs: object) -> Path:
        seen.append(url)
        shutil.copyfile(fixture_zip, dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    Workdir(tmp_path / "work").create()
    override = "https://example.test/drugsfda-snapshot.zip"
    ctx = _ctx("sample", tmp_path / "work", params={"drugsfda_url": override})

    drugsfda.fetch(ctx)
    assert seen == [override]


def test_download_drugsfda_zip_streams_to_dest(tmp_path: Path) -> None:
    """The real downloader (stdlib urllib) copies bytes verbatim from a file:// URL."""
    fixture_zip = _write_fixture_zip(tmp_path)
    dest = tmp_path / "downloaded.zip"
    result = drugsfda.download_drugsfda_zip(fixture_zip.as_uri(), dest)
    assert result == dest
    assert dest.read_bytes() == fixture_zip.read_bytes()
