"""BLAKE3 content + tree hashing.

BLAKE3 is the primary DAKP content hash for speed on large source files and extracted
trees (a Nix-store-inspired artifact and cryptography model with BLAKE3).
Optional SHA-256/SRI metadata is computed only as interoperability sugar; the canonical
artifact id is always ``b3:<hex>``.

Pure code path: uses the ``blake3`` Rust-extension wheel, so tests/CI require no external
CLI tools (no ``nix-hash``, no ``b3sum``). File hashing is memory-mapped and multithreaded
(``blake3(max_threads=blake3.AUTO)`` + ``Hasher.update_mmap``); BLAKE3 is a tree hash
internally, so digests are byte-identical regardless of threading. ``hash_file_with_sri``
overlaps the (slower) SHA-256 stream with BLAKE3 on a thread pool.
"""

from __future__ import annotations

import base64
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from blake3 import blake3

BLAKE3_ALGORITHM = "BLAKE3"
_DEFAULT_CHUNK = 1 << 20  # 1 MiB read window


def artifact_id(hex_digest: str) -> str:
    """Normalize a hex digest into the canonical ``b3:<hex>`` artifact id."""
    if hex_digest.startswith("b3:"):
        return hex_digest
    return f"b3:{hex_digest}"


def _hex(hex_digest: str) -> str:
    """Strip the ``b3:`` prefix to get the bare hex digest (for directory names)."""
    return hex_digest.split(":", 1)[1] if hex_digest.startswith("b3:") else hex_digest


def hash_bytes(data: bytes) -> str:
    """BLAKE3 of a bytes blob, returned as ``b3:<hex>``."""
    return artifact_id(blake3(data).hexdigest())


def hash_file(path: Path, *, chunk_size: int = _DEFAULT_CHUNK) -> str:
    """Multithreaded mmap BLAKE3 of a file's bytes, returned as ``b3:<hex>``.

    ``chunk_size`` is accepted for call-site compatibility but is advisory: hashing uses
    ``Hasher.update_mmap`` with Rayon multithreading rather than a chunked read loop.
    """
    del chunk_size  # advisory only on the mmap path
    hasher = blake3(max_threads=blake3.AUTO)
    hasher.update_mmap(str(path))
    return artifact_id(hasher.hexdigest())


def _blake3_file(path: Path) -> str:
    hasher = blake3(max_threads=blake3.AUTO)
    hasher.update_mmap(str(path))
    return artifact_id(hasher.hexdigest())


def _sha256_file(path: Path, chunk_size: int) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    digest = base64.b64encode(hasher.digest()).decode("ascii")
    return f"sha256-{digest}"


def hash_file_with_sri(path: Path, *, chunk_size: int = _DEFAULT_CHUNK) -> tuple[str, str]:
    """Concurrent BLAKE3 id + SHA-256 SRI of a file's bytes.

    Runs BLAKE3 (mmap, multithreaded) and the streaming SHA-256 loop on a two-worker
    thread pool — hashlib releases the GIL on large buffers and the Rust blake3 extension
    does its own threading, so the two truly overlap. The file is read twice (once mapped,
    once streamed). Returns ``(b3:<hex>, sha256-<base64>)`` in exactly the formats of
    :func:`hash_file` and :func:`sha256_sri`.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        b3_future = pool.submit(_blake3_file, path)
        sha_future = pool.submit(_sha256_file, path, chunk_size)
        return b3_future.result(), sha_future.result()


def hash_tree(root: Path, *, chunk_size: int = _DEFAULT_CHUNK) -> str:
    """Deterministic BLAKE3 tree hash over a directory (algorithm ``b3-tree-v1``).

    Nix-NAR-like in spirit but BLAKE3-based: stable over sorted relative paths, file
    sizes, and file contents. Directory mtimes, traversal order, and empty dirs do not
    affect the result. Returns ``b3:<hex>``. File contents are fed via
    ``Hasher.update_mmap``; ``chunk_size`` is advisory only on that path.

    Single-threaded by construction: the interleaved metadata updates (path/size between
    file payloads) cannot use Rayon chunk parallelism, and the hasher is constructed
    without ``max_threads``. Callers hashing large trees (model weights) should prefer
    :func:`hash_tree_mt` - at the cost of a DIFFERENT digest (see there). This function's
    digest is frozen forever: the Go-side artifact store mirrors it byte-for-byte
    (``go/internal/blake3store``), and existing content-addressed artifacts depend on it.
    """
    del chunk_size  # advisory only on the mmap path
    hasher = blake3()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(size).encode("ascii"))
        hasher.update(b"\x00")
        hasher.update_mmap(str(path))
        hasher.update(b"\x00")
    return artifact_id(hasher.hexdigest())


#: Algorithm tag recorded in model-cache manifests so old and new tree hashes can be
#: told apart. ``b3-tree-v1`` is :func:`hash_tree` (frozen); ``b3-tree-v2-mt`` is
#: :func:`hash_tree_mt` (multithreaded, different digest by construction).
TREE_HASH_V1 = "b3-tree-v1"
TREE_HASH_V2_MT = "b3-tree-v2-mt"


def hash_tree_mt(root: Path, *, max_workers: int | None = None) -> str:
    """Multithreaded BLAKE3 tree hash (algorithm ``b3-tree-v2-mt``).

    BLAKE3 scales ~linearly with cores, and model-weight trees run into gigabytes, so the
    sequential ``b3-tree-v1`` walk leaves the whole machine idle. This variant parallelizes
    twice over: per-FILE hashing runs on a thread pool (files are independent), and each
    file's own BLAKE3 uses Rayon chunk parallelism (``max_threads=blake3.AUTO`` over the
    mmap). The per-file results (relative path, size, file digest) are then combined in the
    same sorted-path order ``b3-tree-v1`` uses, so the result is deterministic regardless
    of worker count or completion order.

    The digest is deliberately DIFFERENT from :func:`hash_tree` for the same tree - the
    per-file contribution is the file's own digest, not its raw bytes - which is why the
    algorithm name is recorded alongside the hash (model-cache manifests carry
    ``hash_algo``). Never feed a ``b3-tree-v2-mt`` digest into a path expecting ``b3-tree-v1``
    (the Go-side artifact store) or vice versa.
    """
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())

    def file_entry(path: Path) -> tuple[bytes, int, str]:
        hasher = blake3(max_threads=blake3.AUTO)
        hasher.update_mmap(str(path))
        return path.relative_to(root).as_posix().encode("utf-8"), path.stat().st_size, hasher.hexdigest()

    workers = max_workers if max_workers is not None else min(32, (os.cpu_count() or 4))
    if workers <= 1 or len(files) <= 1:
        entries = [file_entry(path) for path in files]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            entries = list(pool.map(file_entry, files))
    hasher = blake3()
    for rel, size, file_hex in entries:
        hasher.update(rel)
        hasher.update(b"\x00")
        hasher.update(str(size).encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(file_hex.encode("ascii"))
        hasher.update(b"\x00")
    return artifact_id(hasher.hexdigest())


def sha256_sri(path: Path, *, chunk_size: int = _DEFAULT_CHUNK) -> str:
    """Optional secondary interoperability hash: a Subresource Integrity string.

    Returns ``sha256-<base64>`` (the W3C SRI format). Computed in addition to BLAKE3 so
    downstream tooling that expects SRI/Nix-style hashes can consume it without forcing
    DAKP to abandon BLAKE3 as the primary key.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    digest = base64.b64encode(hasher.digest()).decode("ascii")
    return f"sha256-{digest}"


def digest_dirname(artifact_id_str: str) -> str:
    """Return the bare hex digest used as the store directory name for an artifact id."""
    return _hex(artifact_id_str)


__all__ = [
    "BLAKE3_ALGORITHM",
    "TREE_HASH_V1",
    "TREE_HASH_V2_MT",
    "artifact_id",
    "digest_dirname",
    "hash_bytes",
    "hash_file",
    "hash_file_with_sri",
    "hash_tree",
    "hash_tree_mt",
    "sha256_sri",
]
