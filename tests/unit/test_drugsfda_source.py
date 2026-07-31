"""Tests for the Drugs@FDA source fetcher.

Covers the real download path (with the network downloader monkeypatched to serve a local fixture
ZIP), content-addressing idempotence, the ``drugsfda_url`` override, and the stdlib downloader
itself (via a ``file://`` URL). The mock fixture branch is gone — fetchers always run real.
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


def _ctx(workdir: Path, *, params: dict[str, object] | None = None) -> TaskContext:
    merged = dict(params) if params else {}
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params=merged)


def _write_fixture_zip(tmp_path: Path) -> Path:
    """A ZIP mirroring the real Drugs@FDA data-files layout (Products.txt / etc.)."""
    zip_path = tmp_path / "drugsfda.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in ("drugsfda_products.tsv", "drugsfda_applications.tsv", "drugsfda_submissions.tsv"):
            archive.write(_DRUGSFDA_FIXTURE_DIR / name, arcname=name.replace("drugsfda_", "").capitalize())
    return zip_path


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
    ctx = _ctx(tmp_path / "work")

    first = drugsfda.fetch(ctx)
    assert len(first) == 1
    ref = first[0]
    assert ref.media_type == "application/zip"
    assert ref.blake3.startswith("b3:")
    assert ref.manifest is not None
    assert ref.manifest.exists()
    # The downloader was actually invoked with the canonical FDA URL.
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
    ctx = _ctx(tmp_path / "work", params={"drugsfda_url": override})

    drugsfda.fetch(ctx)
    assert seen == [override]


def test_download_drugsfda_zip_streams_to_dest(tmp_path: Path) -> None:
    """The real downloader (stdlib urllib) copies bytes verbatim from a file:// URL."""
    fixture_zip = _write_fixture_zip(tmp_path)
    dest = tmp_path / "downloaded.zip"
    result = drugsfda.download_drugsfda_zip(fixture_zip.as_uri(), dest)
    assert result == dest
    assert dest.read_bytes() == fixture_zip.read_bytes()
