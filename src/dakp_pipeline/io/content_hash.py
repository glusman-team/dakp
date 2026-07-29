"""BLAKE3 content + tree hashing.

BLAKE3 is the primary DAKP content hash for speed on large source files and extracted
trees (per ``PLAN.md`` "Nix-store-inspired artifact and cryptography model with BLAKE3").
Optional SHA-256/SRI metadata is computed only as interoperability sugar; the canonical
artifact id is always ``b3:<hex>``.

Pure code path: uses the ``blake3`` Rust-extension wheel, so tests/CI require no external
CLI tools (no ``nix-hash``, no ``b3sum``).
"""

from __future__ import annotations

import base64
import hashlib
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
    """Streaming BLAKE3 of a file's bytes, returned as ``b3:<hex>``."""
    hasher = blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return artifact_id(hasher.hexdigest())


def hash_tree(root: Path, *, chunk_size: int = _DEFAULT_CHUNK) -> str:
    """Deterministic BLAKE3 tree hash over a directory.

    Nix-NAR-like in spirit but BLAKE3-based: stable over sorted relative paths, file
    sizes, and file contents. Directory mtimes, traversal order, and empty dirs do not
    affect the result. Returns ``b3:<hex>``.
    """
    hasher = blake3()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(size).encode("ascii"))
        hasher.update(b"\x00")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                hasher.update(chunk)
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


__all__ = ["BLAKE3_ALGORITHM", "artifact_id", "digest_dirname", "hash_bytes", "hash_file", "hash_tree", "sha256_sri"]
