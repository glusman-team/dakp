"""Artifact manifests — the JSON provenance record for every content-addressed artifact.

The manifest is the BLAKE3 artifact manifest shape (``schema_version``
``dakp.artifact.v1``). Modeled with pydantic v2 for validation and (de)serialization.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

SCHEMA_VERSION = "dakp.artifact.v1"


class HashBlock(BaseModel):
    algorithm: str = Field(default="BLAKE3", description="Primary content-hash algorithm.")
    file: str | None = Field(default=None, description="b3:<hex> file content hash.")
    tree: str | None = Field(default=None, description="b3:<hex> deterministic tree hash (directories only).")
    sha256_sri: str | None = Field(default=None, description="Optional secondary SHA-256 SRI string for interoperability.")


class OperationBlock(BaseModel):
    name: str
    version: str = "v1"
    config_hash: str | None = None


class SourceBlock(BaseModel):
    url: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    retrieved_at: str | None = None


class EnvironmentBlock(BaseModel):
    git_commit: str | None = None
    uv_lock_hash: str | None = None
    tablassert_commit: str | None = None
    fullmap_hash: str | None = None


class TableBlock(BaseModel):
    rows: int | None = None
    partitions: int | None = None
    schema_fingerprint: str | None = None
    warnings: int | None = None


class ArtifactManifest(BaseModel):
    """Provenance record for one artifact, written alongside the content-addressed store."""

    schema_version: str = SCHEMA_VERSION
    artifact_id: str
    path: str
    media_type: str
    hash: HashBlock = Field(default_factory=HashBlock)
    inputs: list[str] = Field(default_factory=list)
    operation: OperationBlock | None = None
    source: SourceBlock = Field(default_factory=SourceBlock)
    environment: EnvironmentBlock = Field(default_factory=EnvironmentBlock)
    table: TableBlock = Field(default_factory=TableBlock)

    def write(self, path: Path) -> Path:
        """Atomically-ish write this manifest as indented JSON. Returns ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def read_manifest(path: Path) -> ArtifactManifest:
    """Read and validate a manifest JSON file."""
    return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = ["SCHEMA_VERSION", "ArtifactManifest", "EnvironmentBlock", "HashBlock", "OperationBlock", "SourceBlock", "TableBlock", "read_manifest"]
