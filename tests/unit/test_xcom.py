"""Tests for the ArtifactRef <-> XCom codec (the Python<->Go serialization contract)."""

from __future__ import annotations

import json
from pathlib import Path

from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.io.xcom import REFS_FILE_MEDIA_TYPE, ref_from_xcom, ref_to_xcom, refs_from_file, refs_from_xcom, refs_to_xcom


def test_ref_to_xcom_full() -> None:
    ref = ArtifactRef(
        uri=Path("data/raw/by-hash/ab/file.xml.gz"),
        blake3="b3:ab",
        media_type="application/gzip",
        rows=5,
        schema_fingerprint="b3:cd",
        manifest=Path("data/manifests/ab.json"),
    )
    assert ref_to_xcom(ref) == {
        "uri": "data/raw/by-hash/ab/file.xml.gz",
        "blake3": "b3:ab",
        "media_type": "application/gzip",
        "rows": 5,
        "schema_fingerprint": "b3:cd",
        "manifest": "data/manifests/ab.json",
    }


def test_ref_to_xcom_nullables() -> None:
    ref = ArtifactRef(uri=Path("x.tsv"), blake3="b3:zz", media_type="text/tab-separated-values")
    assert ref_to_xcom(ref) == {
        "uri": "x.tsv",
        "blake3": "b3:zz",
        "media_type": "text/tab-separated-values",
        "rows": None,
        "schema_fingerprint": None,
        "manifest": None,
    }


def test_ref_round_trip() -> None:
    ref = ArtifactRef(
        uri=Path("data/tabular/approved_treats_assertions.tsv"),
        blake3="b3:aa",
        media_type="text/tab-separated-values",
        rows=3,
        schema_fingerprint="b3:fp",
        manifest=Path("data/manifests/aa.json"),
    )
    assert ref_from_xcom(ref_to_xcom(ref)) == ref


def test_ref_from_xcom_nullables_default_to_none() -> None:
    data = {"uri": "y.parquet", "blake3": "b3:bb", "media_type": "application/vnd.apache.parquet"}
    ref = ref_from_xcom(data)
    assert ref.uri == Path("y.parquet")
    assert ref.rows is None
    assert ref.schema_fingerprint is None
    assert ref.manifest is None


def test_refs_to_xcom_and_back() -> None:
    refs = [
        ArtifactRef(uri=Path("a.parquet"), blake3="b3:a", media_type="application/vnd.apache.parquet", rows=1),
        ArtifactRef(uri=Path("b.tsv"), blake3="b3:b", media_type="text/tab-separated-values", rows=2),
    ]
    serialized = refs_to_xcom(refs)
    assert isinstance(serialized, list)
    assert all(isinstance(item, dict) for item in serialized)
    assert refs_from_xcom(serialized) == refs


def test_refs_from_xcom_none_is_empty() -> None:
    assert refs_from_xcom(None) == []
    assert refs_from_xcom([]) == []


# --- single-file refs handoff (the DailyMed acquire XCom shrink) ----------------


def _refs_file(tmp_path: Path, refs: list[ArtifactRef]) -> Path:
    """Write refs in the exact shape the producer's single-file handoff writes."""
    path = tmp_path / "spl-refs.json"
    path.write_text(json.dumps(refs_to_xcom(refs), indent=2), encoding="utf-8")
    return path


def test_refs_from_xcom_resolves_refs_file_sentinel(tmp_path: Path) -> None:
    """A one-element sentinel payload resolves to the full refs list from the store JSON."""
    refs = [
        ArtifactRef(uri=Path("m1.xml"), blake3="b3:1", media_type="application/xml", manifest=Path("data/manifests/1.json")),
        ArtifactRef(uri=Path("m2.xml.gz"), blake3="b3:2", media_type="application/gzip", manifest=Path("data/manifests/2.json")),
    ]
    sentinel = ref_to_xcom(ArtifactRef(uri=_refs_file(tmp_path, refs), blake3="b3:refs", media_type=REFS_FILE_MEDIA_TYPE))
    assert refs_from_xcom([sentinel]) == refs


def test_refs_from_xcom_single_non_sentinel_ref_stays_inline() -> None:
    """A genuine one-ref inline list (e.g. the Drugs@FDA ZIP) is never mistaken for the sentinel."""
    ref = ArtifactRef(uri=Path("drugsfda.zip"), blake3="b3:z", media_type="application/zip")
    assert refs_from_xcom(refs_to_xcom([ref])) == [ref]


def test_refs_from_file_reads_the_handoff_json(tmp_path: Path) -> None:
    refs = [ArtifactRef(uri=Path("only.xml"), blake3="b3:o", media_type="application/xml", rows=4)]
    assert refs_from_file(_refs_file(tmp_path, refs)) == refs
