"""Tests for the ArtifactRef <-> XCom codec (the Python<->Go serialization contract)."""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.io.xcom import ref_from_xcom, ref_to_xcom, refs_from_xcom, refs_to_xcom


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
