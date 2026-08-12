from __future__ import annotations

import json
from pathlib import Path

import pytest

from dakp_pipeline.io import artifact_store, schemas
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


def test_ingest_cache_hit_skips_copy_and_secondary_hash(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    src = tmp_path / "same.bin"
    src.write_bytes(b"identical bytes")
    real_sri = artifact_store.sha256_sri
    sri_calls: list[Path] = []

    def record_sri(path: Path) -> str:
        sri_calls.append(path)
        return real_sri(path)

    monkeypatch.setattr(artifact_store, "sha256_sri", record_sri)
    ref1, hit1 = store.ingest(src)
    sri_calls.clear()
    ref2, hit2 = store.ingest(src)

    assert hit1 is False
    assert hit2 is True  # identical content already present -> cache hit, no copy
    assert sri_calls == []  # the BLAKE3 check is enough; reuse the stored SRI metadata
    assert ref1.blake3 == ref2.blake3
    assert ref1.uri == ref2.uri


def test_ingest_repairs_corrupt_cache_manifest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "same.bin"
    src.write_bytes(b"identical bytes")
    ref, _ = store.ingest(src)
    assert ref.manifest is not None
    ref.manifest.write_text("not json", encoding="utf-8")

    repaired, hit = store.ingest(src)

    assert hit is True
    assert repaired.blake3 == ref.blake3
    assert repaired.manifest is not None
    assert read_manifest(repaired.manifest).artifact_id == ref.blake3


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


def test_cached_ref_resolves_ingested_alias(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "quarter.zip"
    src.write_bytes(b"faers quarter bytes")
    ref, _ = store.ingest(src, alias="faers/faers_ascii_24Q3.zip")

    cached = store.cached_ref("faers/faers_ascii_24Q3.zip")
    assert cached is not None
    assert cached.blake3 == ref.blake3
    assert cached.uri == ref.uri
    assert cached.uri.exists()
    assert cached.manifest == ref.manifest


def test_cached_refs_requires_completed_current_alias_set(tmp_path: Path) -> None:
    store = _store(tmp_path)
    release = tmp_path / "release.zip"
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    release.write_bytes(b"release")
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    release_ref, _ = store.ingest(release, alias="dailymed/release.zip")
    store.ingest(first, alias="dailymed/release.zip::b.xml")
    store.ingest(second, alias="dailymed/release.zip::a.xml")
    store.write_cached_refs("dailymed/release.zip", release_ref.blake3, ["dailymed/release.zip::a.xml", "dailymed/release.zip::b.xml"])

    refs = store.cached_refs("dailymed/release.zip", release_ref.blake3)
    assert refs is not None
    assert [ref.uri.name for ref in refs] == ["second.xml", "first.xml"]
    assert store.cached_refs("dailymed/other.zip", release_ref.blake3) is None

    # A member alias left behind from a replacement release invalidates the fast path.
    store.write_cached_refs("dailymed/release.zip", "b3:new-release", ["dailymed/release.zip::a.xml"])
    assert store.cached_refs("dailymed/release.zip", "b3:new-release") is not None

    incomplete = store.workdir.aliases / "dailymed" / "broken.zip::x.xml"
    incomplete.write_text("b3:missing", encoding="utf-8")
    assert store.cached_refs("dailymed/broken.zip", release_ref.blake3) is None


# --- cached_refs: every rejection is a cache MISS, never a crash ----------------
#
# The fast path exists purely to skip re-expanding a fixed-name release ZIP, so anything it
# cannot fully verify must degrade to ``None`` (re-expand) rather than raise or, worse, hand
# back a half-populated member list. These drive each guard from a real on-disk state.


_ALIAS = "dailymed/release.zip"


def _fanout_store(tmp_path: Path) -> tuple[ArtifactStore, str]:
    """A store holding one two-member fan-out with a published completion record."""
    store = _store(tmp_path)
    release = tmp_path / "release.zip"
    release.write_bytes(b"release")
    for name in ("a.xml", "b.xml"):
        member = tmp_path / name
        member.write_bytes(name.encode())
        store.ingest(member, alias=f"{_ALIAS}::{name}")
    release_ref, _ = store.ingest(release, alias=_ALIAS)
    store.write_cached_refs(_ALIAS, release_ref.blake3, [f"{_ALIAS}::a.xml", f"{_ALIAS}::b.xml"])
    assert store.cached_refs(_ALIAS, release_ref.blake3) is not None  # the fast path is live before each edit below
    return store, release_ref.blake3


def _marker(store: ArtifactStore) -> Path:
    return store.workdir.aliases / "dailymed" / "release.zip.members.json"


def test_cached_refs_rejects_a_record_for_a_different_release(tmp_path: Path) -> None:
    """The record is keyed by the source artifact id: a replaced fixed-name ZIP invalidates it."""
    store, artifact_id = _fanout_store(tmp_path)
    assert store.cached_refs(_ALIAS, "b3:some-other-release") is None
    assert store.cached_refs(_ALIAS, artifact_id) is not None


def test_cached_refs_rejects_a_malformed_alias_list(tmp_path: Path) -> None:
    """``aliases`` must be a list of strings; anything else is a corrupt record, not a cache."""
    store, artifact_id = _fanout_store(tmp_path)
    for aliases in ("not-a-list", {"a": 1}, [f"{_ALIAS}::a.xml", 7]):
        _marker(store).write_text(json.dumps({"artifact_id": artifact_id, "aliases": aliases}), encoding="utf-8")
        assert store.cached_refs(_ALIAS, artifact_id) is None


def test_cached_refs_rejects_foreign_or_duplicated_member_aliases(tmp_path: Path) -> None:
    """A member must live under ``<alias>::`` and appear once — a hand-edited record cannot widen it."""
    store, artifact_id = _fanout_store(tmp_path)
    foreign = [f"{_ALIAS}::a.xml", "dailymed/other.zip::a.xml"]
    duplicated = [f"{_ALIAS}::a.xml", f"{_ALIAS}::a.xml"]
    for aliases in (foreign, duplicated):
        _marker(store).write_text(json.dumps({"artifact_id": artifact_id, "aliases": aliases}), encoding="utf-8")
        assert store.cached_refs(_ALIAS, artifact_id) is None


def test_cached_refs_rejects_an_unlistable_alias_namespace(tmp_path: Path) -> None:
    """A flat alias makes the member scan hit a FILE where the namespace directory should be."""
    store = _store(tmp_path)
    payload = tmp_path / "release.zip"
    payload.write_bytes(b"release")
    ref, _ = store.ingest(payload, alias="release.zip")  # no "/" -> the "namespace" IS the alias file
    marker = store.workdir.aliases / "release.zip.members.json"
    marker.write_text(json.dumps({"artifact_id": ref.blake3, "aliases": ["release.zip::a.xml"]}), encoding="utf-8")

    assert store.cached_refs("release.zip", ref.blake3) is None  # NotADirectoryError -> miss, not a crash


def test_cached_refs_rejects_a_member_set_that_drifted_on_disk(tmp_path: Path) -> None:
    """The recorded set must match the aliases actually present — a deleted member is a miss."""
    store, artifact_id = _fanout_store(tmp_path)
    (store.workdir.aliases / f"{_ALIAS}::b.xml").unlink()

    assert store.cached_refs(_ALIAS, artifact_id) is None


def test_cached_refs_rejects_an_unreadable_member_id(tmp_path: Path, monkeypatch) -> None:
    """An alias file that lists but will not read (permissions, I/O error) is a miss."""
    store, artifact_id = _fanout_store(tmp_path)
    unreadable = store.workdir.aliases / f"{_ALIAS}::a.xml"
    real_read_text = Path.read_text

    def failing_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == unreadable:
            raise OSError("simulated I/O error")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", failing_read_text)
    assert store.cached_refs(_ALIAS, artifact_id) is None


def test_cached_refs_rejects_a_member_missing_its_path_pointer(tmp_path: Path) -> None:
    """The ``.path`` sidecar is excluded from the alias scan, so its loss surfaces only here."""
    store, artifact_id = _fanout_store(tmp_path)
    (store.workdir.aliases / f"{_ALIAS}::a.xml.path").unlink()

    assert store.cached_refs(_ALIAS, artifact_id) is None


def test_cached_refs_rejects_a_member_whose_payload_or_manifest_is_gone(tmp_path: Path) -> None:
    """Alias records outlive the files they point at; both the payload and its manifest must exist."""
    store, artifact_id = _fanout_store(tmp_path)
    pointer = store.workdir.aliases / f"{_ALIAS}::a.xml.path"
    stored = Path(pointer.read_text(encoding="utf-8").strip())

    stored.unlink()  # payload evicted from the store
    assert store.cached_refs(_ALIAS, artifact_id) is None

    stored.write_bytes(b"a.xml")  # payload back, manifest gone
    store.manifest_path((store.workdir.aliases / f"{_ALIAS}::a.xml").read_text(encoding="utf-8").strip()).unlink()
    assert store.cached_refs(_ALIAS, artifact_id) is None


def test_write_cached_refs_refuses_a_member_outside_its_source_alias(tmp_path: Path) -> None:
    """Publishing is the guard's other half: a foreign member can never enter the record."""
    store, artifact_id = _fanout_store(tmp_path)
    with pytest.raises(ValueError, match="outside source alias"):
        store.write_cached_refs(_ALIAS, artifact_id, [f"{_ALIAS}::a.xml", "dailymed/other.zip::a.xml"])
