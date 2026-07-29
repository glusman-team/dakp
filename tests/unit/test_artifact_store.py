from __future__ import annotations

from pathlib import Path

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.manifests import SCHEMA_VERSION, read_manifest
from dakp_pipeline.paths import Workdir


def _store(tmp_path: Path) -> ArtifactStore:
    wd = Workdir(tmp_path / "work")
    wd.create()
    return ArtifactStore(wd)


def test_ingest_copies_into_by_hash_and_writes_manifest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "raw.zip"
    src.write_bytes(b"pretend dailymed release")

    ref, cache_hit = store.ingest(src, alias="dailymed/latest")

    assert cache_hit is False
    assert ref.blake3.startswith("b3:")
    assert ref.uri.exists()
    # Stored under by-hash/<hex>/<name>.
    assert "by-hash" in ref.uri.parts
    assert ref.uri.name == "raw.zip"
    assert ref.manifest is not None
    assert ref.manifest.exists()

    manifest = read_manifest(ref.manifest)
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.artifact_id == ref.blake3
    assert manifest.hash.file == ref.blake3
    assert manifest.hash.sha256_sri is not None  # secondary interop metadata recorded
    assert manifest.media_type == "application/zip"

    # Alias resolves to the same artifact id.
    alias_id = (Workdir(store.workdir.root).aliases / "dailymed" / "latest").read_text()
    assert alias_id == ref.blake3


def test_ingest_cache_hit_skips_copy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "same.bin"
    src.write_bytes(b"identical bytes")

    ref1, hit1 = store.ingest(src)
    ref2, hit2 = store.ingest(src)

    assert hit1 is False
    assert hit2 is True  # identical content already present -> cache hit, no copy
    assert ref1.blake3 == ref2.blake3
    assert ref1.uri == ref2.uri


def test_register_in_place_writes_manifest_without_copy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Simulate an interim parquet that already lives under the workdir.
    interim = store.workdir.interim / "faers" / "demo.parquet"
    interim.parent.mkdir(parents=True, exist_ok=True)
    interim.write_bytes(b"PAR1...pretend")  # not a real parquet; only hashed here

    ref = store.register(interim, media_type=schemas.PARQUET_MEDIA_TYPE, rows=42)
    assert ref.uri == interim  # in place, not copied
    assert ref.rows == 42
    assert ref.manifest is not None
    assert ref.manifest.exists()
    manifest = read_manifest(ref.manifest)
    assert manifest.table.rows == 42


def test_read_manifest_round_trips_for_artifact_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "x.tsv"
    src.write_text("a\tb\n1\t2\n")
    ref, _ = store.ingest(src)

    fetched = store.read_manifest(ref.blake3)
    assert fetched is not None
    assert fetched.artifact_id == ref.blake3
    assert store.read_manifest("b3:doesnotexist") is None


def test_schema_fingerprint_is_deterministic() -> None:
    cols = schemas.APPROVED_TREATS_COLUMNS
    fp1 = schemas.schema_fingerprint(cols)
    fp2 = schemas.schema_fingerprint(list(cols))
    assert fp1 == fp2
    assert fp1.startswith("b3:")
    # Reordered columns -> different fingerprint (order is significant).
    reordered = list(reversed(cols))
    assert schemas.schema_fingerprint(reordered) != fp1


def test_write_and_read_tsv_round_trips(tmp_path: Path) -> None:
    import polars as pl

    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "out.tsv"
    rows = schemas.write_tsv(frame, path)
    assert rows == 2
    assert path.exists()
    back = schemas.read_table(path)
    assert back.columns == ["a", "b"]
    assert back.height == 2
