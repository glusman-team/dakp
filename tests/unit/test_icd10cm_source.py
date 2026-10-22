"""Tests for the ICD-10-CM Table of Drugs and Chemicals fetcher.

Covers the real download path (network seam monkeypatched to serve a local ZIP), the
release-pinned cache behaviour (no age gate by default: a pinned URL is canonical), the
``force`` and ``icd10cm_url`` overrides, and the alias layout.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.io.manifests import read_manifest
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import icd10cm


def _ctx(workdir: Path, *, params: dict[str, object] | None = None) -> TaskContext:
    return TaskContext(workdir=workdir, fixture_root=None, params=params or {})


def _fixture_zip(tmp_path: Path, member: str = "icd10cm-table-and-index-2027/icd10cm-drug-2027.xml") -> Path:
    zip_path = tmp_path / "icd10cm.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(member, "<ICD10CM.index><version>2027</version></ICD10CM.index>")
    return zip_path


def _patch_download(monkeypatch, fixture_zip: Path) -> list[str]:
    captured: list[str] = []

    def fake_download(url: str, dest: Path, **kwargs: object) -> Path:
        captured.append(url)
        shutil.copyfile(fixture_zip, dest)
        return dest

    monkeypatch.setattr(icd10cm, "download_icd10cm_zip", fake_download)
    return captured


def test_fetch_ingests_the_release_zip(monkeypatch, tmp_path: Path) -> None:
    captured = _patch_download(monkeypatch, _fixture_zip(tmp_path))
    Workdir(tmp_path / "work").create()
    refs = icd10cm.fetch(_ctx(tmp_path / "work"))

    assert captured == [icd10cm.ICD10CM_DRUG_TABLE_ZIP_URL]
    assert len(refs) == 1
    ref = refs[0]
    assert ref.media_type == "application/zip"
    assert ref.blake3.startswith("b3:")
    assert ref.uri.suffix == ".zip"
    assert ref.manifest is not None
    assert ref.manifest.exists()
    manifest = read_manifest(ref.manifest)
    assert manifest is not None
    assert manifest.source.url == icd10cm.ICD10CM_DRUG_TABLE_ZIP_URL


def test_release_pinned_artifact_is_reused_without_an_age_gate(monkeypatch, tmp_path: Path) -> None:
    """A cached release-pinned ZIP is canonical: the second fetch never re-downloads it."""
    fixture = _fixture_zip(tmp_path)
    captured = _patch_download(monkeypatch, fixture)
    workdir = tmp_path / "work"
    Workdir(workdir).create()

    first = icd10cm.fetch(_ctx(workdir))
    second = icd10cm.fetch(_ctx(workdir))
    assert captured == [icd10cm.ICD10CM_DRUG_TABLE_ZIP_URL]  # one download only
    assert second[0].blake3 == first[0].blake3
    store = ArtifactStore(Workdir(workdir))
    assert store.cached_ref(f"icd10cm/{Path(icd10cm.ICD10CM_DRUG_TABLE_ZIP_URL).name}") is not None


def test_max_age_days_defaults_to_no_gate_and_honours_the_param() -> None:
    """The release URL is version-pinned, so the gate is off unless a caller asks for one."""
    assert icd10cm._max_age_days(_ctx(Path("unused"))) is None
    assert icd10cm._max_age_days(_ctx(Path("unused"), params={"icd10cm_max_age_days": 30})) == 30.0
    assert icd10cm._max_age_days(_ctx(Path("unused"), params={"icd10cm_max_age_days": 0})) is None
    assert icd10cm._max_age_days(_ctx(Path("unused"), params={"icd10cm_max_age_days": "nope"})) is None


def test_cache_age_days_reads_the_retrieval_provenance() -> None:
    fresh = datetime.now(UTC) - timedelta(days=2)
    manifest = lambda value: SimpleNamespace(source=SimpleNamespace(retrieved_at=value))
    age = icd10cm._cache_age_days(manifest(fresh.isoformat()))
    assert age is not None
    assert 1.9 < age < 2.1
    assert icd10cm._cache_age_days(None) is None
    assert icd10cm._cache_age_days(manifest(None)) is None
    assert icd10cm._cache_age_days(manifest("not-a-date")) is None


def test_force_param_redownloads(monkeypatch, tmp_path: Path) -> None:
    fixture = _fixture_zip(tmp_path)
    captured = _patch_download(monkeypatch, fixture)
    workdir = tmp_path / "work"
    Workdir(workdir).create()
    icd10cm.fetch(_ctx(workdir))
    icd10cm.fetch(_ctx(workdir, params={"force": True}))
    assert len(captured) == 2


def test_url_override_acquires_under_its_own_alias(monkeypatch, tmp_path: Path) -> None:
    """A different fiscal-year URL is a different release: its own alias, its own download."""
    fixture = _fixture_zip(tmp_path)
    captured = _patch_download(monkeypatch, fixture)
    workdir = tmp_path / "work"
    Workdir(workdir).create()
    override = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2028/icd10cm-table-and-index-2028.zip"

    refs = icd10cm.fetch(_ctx(workdir, params={"icd10cm_url": override}))
    assert captured == [override]
    store = ArtifactStore(Workdir(workdir))
    assert store.cached_ref("icd10cm/icd10cm-table-and-index-2028.zip") is not None
    assert refs[0].manifest is not None
    assert read_manifest(refs[0].manifest) is not None
    assert read_manifest(refs[0].manifest).source.url == override


def test_pinned_release_member_matches_the_prefix_constant() -> None:
    """The pinned URL's release must carry the drug table member the extractor looks for."""
    name = Path(icd10cm.ICD10CM_DRUG_TABLE_ZIP_URL).name
    assert name.endswith(".zip")
    assert icd10cm.ICD10CM_RELEASE in name or icd10cm.ICD10CM_RELEASE[-2:] in name
    assert icd10cm.ICD10CM_DRUG_XML_PREFIX == "icd10cm-drug-"
