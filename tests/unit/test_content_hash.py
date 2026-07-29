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


def test_sha256_sri_format(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    sri = content_hash.sha256_sri(path)
    assert sri.startswith("sha256-")
    # The SRI base64 must be a valid base64 payload (no newline, url-safe not required).
    assert "\n" not in sri
