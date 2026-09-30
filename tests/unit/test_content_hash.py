from __future__ import annotations

from pathlib import Path

from dakp_pipeline.io import content_hash


def test_hash_bytes_is_deterministic_and_prefixed() -> None:
    a = content_hash.hash_bytes(b"hello world")
    b = content_hash.hash_bytes(b"hello world")
    assert a == b
    assert a.startswith("b3:")
    # Different content yields a different id.
    assert content_hash.hash_bytes(b"hello world!") != a


def test_hash_file_matches_hash_bytes(tmp_path: Path) -> None:
    payload = b"dailymed spl fragment"
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert content_hash.hash_file(path) == content_hash.hash_bytes(payload)


def test_artifact_id_normalization() -> None:
    assert content_hash.artifact_id("deadbeef") == "b3:deadbeef"
    assert content_hash.artifact_id("b3:deadbeef") == "b3:deadbeef"
    assert content_hash.digest_dirname("b3:deadbeef") == "deadbeef"
    assert content_hash.digest_dirname("deadbeef") == "deadbeef"


def test_hash_tree_is_deterministic_regardless_of_write_order(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()

    # Same contents, written in a different order into the two trees.
    (root_a / "zeta.txt").write_text("zzz")
    (root_a / "alpha.txt").write_text("aaa")
    (root_a / "sub").mkdir()
    (root_a / "sub" / "mid.txt").write_text("mmm")

    (root_b / "sub").mkdir()
    (root_b / "sub" / "mid.txt").write_text("mmm")
    (root_b / "alpha.txt").write_text("aaa")
    (root_b / "zeta.txt").write_text("zzz")

    assert content_hash.hash_tree(root_a) == content_hash.hash_tree(root_b)


def test_hash_tree_changes_on_content_or_layout_change(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "f.txt").write_text("x")
    h0 = content_hash.hash_tree(base)

    # Content change -> different hash.
    (base / "f.txt").write_text("y")
    assert content_hash.hash_tree(base) != h0
    (base / "f.txt").write_text("x")

    # Added file -> different hash.
    (base / "g.txt").write_text("x")
    assert content_hash.hash_tree(base) != h0


def test_hash_tree_empty_dir_is_stable(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert content_hash.hash_tree(empty) == content_hash.hash_tree(empty)


def _two_file_tree(root: Path) -> Path:
    root.mkdir()
    (root / "sub").mkdir()
    (root / "sub" / "big.bin").write_bytes(b"B" * 3_000_000)  # large enough for chunk parallelism
    (root / "small.txt").write_bytes(b"hello")
    return root


def test_hash_tree_mt_is_deterministic_across_worker_counts(tmp_path: Path) -> None:
    """b3-tree-v2-mt must not depend on thread-pool size or file completion order.

    The model cache records v2 digests in manifests; a digest that wobbled with scheduling
    would randomly fail verification and re-download multi-GB weights.
    """
    tree = _two_file_tree(tmp_path / "tree")
    assert content_hash.hash_tree_mt(tree) == content_hash.hash_tree_mt(tree)
    assert content_hash.hash_tree_mt(tree, max_workers=1) == content_hash.hash_tree_mt(tree, max_workers=8)


def test_hash_tree_mt_matches_its_reference_construction(tmp_path: Path) -> None:
    """Pin the v2 layout: sorted (rel, size, per-file-digest) records fed to one BLAKE3.

    This is the contract the manifest hash_algo tag promises; if the layout ever changes,
    the tag must change with it (existing recorded digests would otherwise not verify).
    """
    tree = _two_file_tree(tmp_path / "tree")
    hasher = content_hash.blake3()
    for path in sorted(p for p in tree.rglob("*") if p.is_file()):
        file_hasher = content_hash.blake3()
        file_hasher.update(path.read_bytes())
        hasher.update(path.relative_to(tree).as_posix().encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(path.stat().st_size).encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(file_hasher.hexdigest().encode("ascii"))
        hasher.update(b"\x00")
    assert content_hash.hash_tree_mt(tree) == content_hash.artifact_id(hasher.hexdigest())


def test_hash_tree_mt_differs_from_v1_and_is_tagged_accordantly(tmp_path: Path) -> None:
    """v1 and v2 MUST produce different digests for the same tree.

    The whole point of the hash_algo manifest field is that a v2 digest can never be
    silently compared against a v1 digest (that comparison would fail and re-download a
    multi-GB cache). This test pins the divergence so an accidental unification is loud.
    """
    tree = _two_file_tree(tmp_path / "tree")
    assert content_hash.hash_tree(tree) != content_hash.hash_tree_mt(tree)
    assert content_hash.TREE_HASH_V1 == "b3-tree-v1"
    assert content_hash.TREE_HASH_V2_MT == "b3-tree-v2-mt"


def test_hash_tree_mt_tracks_content_changes(tmp_path: Path) -> None:
    tree = _two_file_tree(tmp_path / "tree")
    h0 = content_hash.hash_tree_mt(tree)
    (tree / "small.txt").write_bytes(b"goodbye")
    assert content_hash.hash_tree_mt(tree) != h0


def test_hash_tree_mt_empty_dir_is_stable(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert content_hash.hash_tree_mt(empty) == content_hash.hash_tree_mt(empty)


def test_sha256_sri_format(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    sri = content_hash.sha256_sri(path)
    assert sri.startswith("sha256-")
    # The SRI base64 must be a valid base64 payload (no newline, url-safe not required).
    assert "\n" not in sri


def test_hash_file_with_sri_matches_the_two_single_hashers(tmp_path: Path) -> None:
    payload = b"dailymed spl fragment" * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    # Small chunk window forces several read iterations through both hashers.
    assert content_hash.hash_file_with_sri(path, chunk_size=4096) == (
        content_hash.hash_file(path, chunk_size=4096),
        content_hash.sha256_sri(path, chunk_size=4096),
    )
