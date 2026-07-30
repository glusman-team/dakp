"""ArtifactRef <-> XCom serialization for the Airflow-native pipeline.

Tasks communicate ``list[ArtifactRef]`` over XCom. Airflow serializes XCom as JSON, and the native
Go SDK bundle workers read/write the same manifests, so an ArtifactRef crosses the boundary as a
plain JSON object with snake_case string paths. These helpers are the single serialization point:
the Go ``internal/airflow.ArtifactRef`` struct has matching JSON tags, so refs round-trip
unchanged across the Python <-> Go boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dakp_pipeline.io.contracts import ArtifactRef

__all__ = ["ref_from_xcom", "ref_to_xcom", "refs_from_xcom", "refs_to_xcom"]


def ref_to_xcom(ref: ArtifactRef) -> dict[str, Any]:
    """Render an ArtifactRef as a JSON-able dict (paths as strings) for XCom / the Go workers."""
    return {
        "uri": str(ref.uri),
        "blake3": ref.blake3,
        "media_type": ref.media_type,
        "rows": ref.rows,
        "schema_fingerprint": ref.schema_fingerprint,
        "manifest": str(ref.manifest) if ref.manifest is not None else None,
    }


def ref_from_xcom(data: dict[str, Any]) -> ArtifactRef:
    """Reconstruct an ArtifactRef from its XCom dict (the inverse of :func:`ref_to_xcom`)."""
    manifest = data.get("manifest")
    return ArtifactRef(
        uri=Path(data["uri"]),
        blake3=data["blake3"],
        media_type=data["media_type"],
        rows=data.get("rows"),
        schema_fingerprint=data.get("schema_fingerprint"),
        manifest=Path(manifest) if manifest is not None else None,
    )


def refs_to_xcom(refs: list[ArtifactRef]) -> list[dict[str, Any]]:
    """Serialize a list of ArtifactRefs for a task's XCom return value."""
    return [ref_to_xcom(ref) for ref in refs]


def refs_from_xcom(items: list[dict[str, Any]] | None) -> list[ArtifactRef]:
    """Deserialize an upstream task's XCom (a list of dicts, or None) into ArtifactRefs."""
    return [ref_from_xcom(item) for item in (items or [])]
