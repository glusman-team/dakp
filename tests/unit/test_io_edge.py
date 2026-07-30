"""Edge-case tests for the ``dakp_pipeline.io`` layer (100% branch coverage drive).

Targets the error/defensive branches the happy-path suite never reaches:

* ``artifact_store`` — ingest/register of missing paths (FileNotFoundError), the
  ``is_tree=True`` registration path (tree hash, no file hash / no SRI), cache hit vs
  miss, alias + ``.path`` pointer writes, tree-hash determinism + empty dir.
* ``contracts`` — ``TaskContext.fixture`` with a ``None`` fixture_root (ValueError) and a
  missing fixture (FileNotFoundError); frozen-dataclass reprs/equality.
* ``downloads`` — ``infer_media_type`` compound/unknown/uppercase suffixes.
* ``schemas`` — ``columns_for`` unknown-table KeyError + registry-copy isolation,
  ``read_table`` parquet-vs-TSV dispatch, fingerprint order-sensitivity.
* ``content_hash`` / ``manifests`` — already 100%; adversarial robustness only
  (empty file, unicode, symlinked dir non-recursion, manifest round-trip with nulls).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.io import contracts, downloads, manifests, schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import artifact_id, digest_dirname, hash_bytes, hash_file, hash_tree, sha256_sri
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import ArtifactManifest, HashBlock, OperationBlock, SourceBlock, TableBlock, read_manifest
from dakp_pipeline.paths import Workdir


def _store(tmp_path: Path) -> ArtifactStore:
    wd = Workdir(tmp_path / "work")
    wd.create()
    return ArtifactStore(wd)


# --- artifact_store: ingest error paths ----------------------------------------


def test_ingest_missing_file_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(FileNotFoundError, match="cannot ingest missing file"):
        store.ingest(tmp_path / "does-not-exist.txt")


def test_register_missing_path_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(FileNotFoundError, match="cannot register missing path"):
        store.register(tmp_path / "nope.parquet")


# --- artifact_store: ingest cache hit / miss + alias pointers ------------------


def test_ingest_cache_hit_on_identical_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "a.txt"
    src.write_text("same-bytes", encoding="utf-8")
    ref1, hit1 = store.ingest(src, alias="x/first.txt")
    assert hit1 is False
    # Re-ingest the same name+content -> cache hit (store path already present), no recopy.
    ref2, hit2 = store.ingest(src, alias="x/second.txt")
    assert hit2 is True
    assert ref1.blake3 == ref2.blake3
    assert ref1.uri == ref2.uri


def test_ingest_writes_alias_id_and_path_pointer(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "f.txt"
    src.write_text("hello", encoding="utf-8")
    ref, _ = store.ingest(src, alias="ns/dir/f.txt")
    wd = Workdir(store.workdir.root)
    assert (wd.aliases / "ns/dir/f.txt").read_text(encoding="utf-8") == ref.blake3
    assert Path((wd.aliases / "ns/dir/f.txt.path").read_text(encoding="utf-8")) == ref.uri


def test_ingest_infers_media_type_when_unspecified(tmp_path: Path) -> None:
    store = _store(tmp_path)
    src = tmp_path / "blob.zip"
    src.write_bytes(b"PK\x03\x04")
    ref, _ = store.ingest(src)
    assert ref.media_type == "application/zip"


# --- artifact_store: register in-place + is_tree path --------------------------


def test_register_in_place_records_file_hash_and_sri(tmp_path: Path) -> None:
    store = _store(tmp_path)
    out = Workdir(store.workdir.root).interim / "t.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"a": ["1"]}).write_parquet(out)
    ref = store.register(out, media_type=schemas.PARQUET_MEDIA_TYPE, rows=1, schema_fingerprint="b3:fp", operation=OperationBlock(name="op"))
    assert ref.blake3 == hash_file(out)
    assert ref.manifest is not None
    manifest = read_manifest(ref.manifest)
    assert manifest.hash.file == ref.blake3
    assert manifest.hash.tree is None
    assert manifest.hash.sha256_sri == sha256_sri(out)
    assert manifest.table.rows == 1
    assert manifest.table.schema_fingerprint == "b3:fp"


def test_register_is_tree_uses_tree_hash_and_no_sri(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tree_dir = Workdir(store.workdir.root).store / "tree"
    (tree_dir / "sub").mkdir(parents=True)
    (tree_dir / "sub" / "one.txt").write_text("1", encoding="utf-8")
    (tree_dir / "two.txt").write_text("22", encoding="utf-8")
    ref = store.register(tree_dir, media_type="application/x-directory", is_tree=True)
    assert ref.blake3 == hash_tree(tree_dir)
    assert ref.manifest is not None
    manifest = read_manifest(ref.manifest)
    assert manifest.hash.tree == ref.blake3
    assert manifest.hash.file is None  # tree registration carries no per-file hash
    assert manifest.hash.sha256_sri is None  # ...and no SRI


# --- content_hash: adversarial robustness (module already 100%) ----------------


def test_hash_bytes_and_file_agree_and_handle_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert hash_file(empty) == hash_bytes(b"")
    assert hash_file(empty).startswith("b3:")


def test_hash_file_handles_unicode(tmp_path: Path) -> None:
    p = tmp_path / "u.txt"
    data = "héllo→世界".encode()
    p.write_bytes(data)
    assert hash_file(p) == hash_bytes(data)


def test_artifact_id_and_digest_dirname_roundtrip() -> None:
    bare = "ab" * 32
    assert artifact_id(bare) == f"b3:{bare}"
    assert artifact_id(f"b3:{bare}") == f"b3:{bare}"  # idempotent
    assert digest_dirname(f"b3:{bare}") == bare
    assert digest_dirname(bare) == bare


def test_hash_tree_is_deterministic_and_ignores_empty_dirs_and_mtimes(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "x.txt").write_text("X", encoding="utf-8")
        (root / "y.txt").write_text("YY", encoding="utf-8")
        (root / "empty-dir").mkdir()  # empty dirs must not affect the hash
    assert hash_tree(a) == hash_tree(b)


def test_hash_tree_does_not_recurse_into_symlinked_dirs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.txt").write_text("r", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s", encoding="utf-8")
    (root / "link").symlink_to(outside)
    # rglob does not descend into the symlinked directory, so secret.txt is excluded.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "real.txt").write_text("r", encoding="utf-8")
    assert hash_tree(root) == hash_tree(bare)


# --- contracts: TaskContext.fixture error paths + dataclass semantics ----------


def _ctx(tmp_path: Path, fixture_root: Path | None) -> TaskContext:
    return TaskContext(profile="mock", workdir=tmp_path, fixture_root=fixture_root, threads=1, memory_budget_gb=1, params={})


def test_fixture_with_none_root_raises(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, fixture_root=None)
    with pytest.raises(ValueError, match="fixture_root is None"):
        ctx.fixture("anything.txt")


def test_fixture_missing_file_raises(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, fixture_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="fixture not found"):
        ctx.fixture("absent.txt")


def test_fixture_resolves_existing_file(tmp_path: Path) -> None:
    (tmp_path / "present.txt").write_text("data", encoding="utf-8")
    ctx = _ctx(tmp_path, fixture_root=tmp_path)
    ref = ctx.fixture("present.txt")
    assert ref.uri == tmp_path / "present.txt"
    assert ref.blake3 == hash_file(tmp_path / "present.txt")
    assert ref.media_type == "text/plain"


def test_artifactref_is_frozen_and_hashable(tmp_path: Path) -> None:
    ref = ArtifactRef(uri=tmp_path / "x.parquet", blake3="b3:aa", media_type="application/vnd.apache.parquet")
    assert ref.rows is None
    assert ref.schema_fingerprint is None
    assert ref.manifest is None
    with pytest.raises(dataclasses.FrozenInstanceError):  # frozen dataclass rejects assignment
        ref.rows = 5  # type: ignore[misc]
    # Hashable (usable in sets / dict keys) and repr-able.
    assert {ref} == {ref}
    assert "b3:aa" in repr(ref)
    assert "mock" in repr(_ctx(tmp_path, None))


def test_protocol_classes_are_runtime_checkable() -> None:
    class F:
        def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
            return []

    assert isinstance(F(), contracts.Fetcher)
    assert not isinstance(object(), contracts.Fetcher)


# --- downloads: media-type inference -------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.xml.gz", "application/gzip"),  # compound suffix wins over .gz
        ("a.xml", "application/xml"),
        ("a.gz", "application/gzip"),
        ("a.zip", "application/zip"),
        ("a.parquet", "application/vnd.apache.parquet"),
        ("a.tsv", "text/tab-separated-values"),
        ("a.csv", "text/csv"),
        ("a.json", "application/json"),
        ("a.jsonl", "application/x-ndjson"),
        ("a.txt", "text/plain"),
        ("a.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("A.ZIP", "application/zip"),  # case-insensitive
        ("noext", "application/octet-stream"),  # unknown -> octet-stream
        ("archive.tar", "application/octet-stream"),
    ],
)
def test_infer_media_type(name: str, expected: str) -> None:
    assert downloads.infer_media_type(Path(name)) == expected


# --- schemas: columns_for / read_table / fingerprint ---------------------------


def test_columns_for_unknown_table_raises() -> None:
    with pytest.raises(KeyError, match="unknown assertion table"):
        schemas.columns_for("not_a_table")


def test_columns_for_returns_isolated_copy() -> None:
    cols = schemas.columns_for("approved_treats_assertions")
    cols.append("MUTATED")
    # Registry is untouched by mutating the returned list.
    assert "MUTATED" not in schemas.APPROVED_TREATS_COLUMNS
    assert schemas.columns_for("approved_treats_assertions") == schemas.APPROVED_TREATS_COLUMNS


def test_schema_fingerprint_is_order_sensitive_and_stable() -> None:
    assert schemas.schema_fingerprint(["a", "b"]) == schemas.schema_fingerprint(["a", "b"])
    assert schemas.schema_fingerprint(["a", "b"]) != schemas.schema_fingerprint(["b", "a"])
    assert schemas.schema_fingerprint([]).startswith("b3:")


def test_read_table_dispatches_parquet_and_tsv(tmp_path: Path) -> None:
    frame = pl.DataFrame({"col": ["v1", "v2"]})
    pq = tmp_path / "t.parquet"
    tsv = tmp_path / "t.tsv"
    frame.write_parquet(pq)
    frame.write_csv(tsv, separator="\t")
    assert schemas.read_table(pq)["col"].to_list() == ["v1", "v2"]
    assert schemas.read_table(tsv)["col"].to_list() == ["v1", "v2"]


def test_write_tsv_and_parquet_return_row_counts(tmp_path: Path) -> None:
    frame = pl.DataFrame({"c": ["1", "2", "3"]})
    assert schemas.write_tsv(frame, tmp_path / "o.tsv") == 3
    assert schemas.write_parquet(frame, tmp_path / "o.parquet") == 3


# --- manifests: round-trip with nulls / defaults (module already 100%) ---------


def test_manifest_roundtrip_minimal(tmp_path: Path) -> None:
    m = ArtifactManifest(artifact_id="b3:xx", path="/tmp/x", media_type="text/plain")
    out = m.write(tmp_path / "m.json")
    back = read_manifest(out)
    assert back.artifact_id == "b3:xx"
    assert back.schema_version == manifests.SCHEMA_VERSION
    # Defaults populated.
    assert back.hash.algorithm == "BLAKE3"
    assert back.inputs == []
    assert back.operation is None
    assert back.source.url is None
    assert back.table.rows is None


def test_manifest_roundtrip_full_blocks(tmp_path: Path) -> None:
    m = ArtifactManifest(
        artifact_id="b3:yy",
        path="/tmp/y",
        media_type="application/zip",
        hash=HashBlock(file="b3:yy", tree=None, sha256_sri="sha256-zz"),
        inputs=["b3:in1", "b3:in2"],
        operation=OperationBlock(name="op", config_hash="b3:cfg"),
        source=SourceBlock(url="https://x", etag="e", last_modified="lm", retrieved_at="now"),
        table=TableBlock(rows=7, partitions=2, schema_fingerprint="b3:fp", warnings=3),
    )
    back = read_manifest(m.write(tmp_path / "full.json"))
    assert back == m
    assert back.source.etag == "e"
    assert back.table.partitions == 2
