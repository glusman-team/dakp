"""ArtifactRef <-> XCom serialization for the Airflow-native pipeline.

Tasks communicate ``list[ArtifactRef]`` over XCom. Airflow serializes XCom as JSON, and the native
Go SDK bundle workers read/write the same manifests, so an ArtifactRef crosses the boundary as a
plain JSON object with snake_case string paths. These helpers are the single serialization point:
the Go ``internal/airflow.ArtifactRef`` struct has matching JSON tags, so refs round-trip
unchanged across the Python <-> Go boundary.

Very large fan-outs (DailyMed acquires one ref per SPL member — tens of thousands) do NOT cross
XCom inline: the producer writes the full list to ONE JSON file in the content-addressed store and
pushes a single ref marked with :data:`REFS_FILE_MEDIA_TYPE`; :func:`refs_from_xcom` (and the Go
``DecodeArtifactRefs`` mirror) resolve that sentinel back to the full list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dakp_pipeline.io.contracts import ArtifactRef

__all__ = ["REFS_FILE_MEDIA_TYPE", "ref_from_xcom", "ref_to_xcom", "refs_from_file", "refs_from_xcom", "refs_to_xcom"]

#: Sentinel media type marking the single-file refs handoff: a one-element XCom list whose ref
#: points at a JSON file (in the store) holding the full ``refs_to_xcom`` list. Mirrored by
#: ``go/internal/airflow/artifactref.go`` (``RefsFileMediaType``) — keep the two in lockstep.
REFS_FILE_MEDIA_TYPE = "application/vnd.dakp.refs+json"


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


def refs_from_file(path: Path) -> list[ArtifactRef]:
    """Read the refs JSON file of the single-file handoff (the inverse of the producer's write)."""
    return [ref_from_xcom(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def refs_from_xcom(items: list[dict[str, Any]] | None) -> list[ArtifactRef]:
    """Deserialize an upstream task's XCom (a list of dicts, or None) into ArtifactRefs.

    Resolves the single-file handoff transparently: a one-element list whose ref carries the
    :data:`REFS_FILE_MEDIA_TYPE` sentinel is read from the store JSON it points at; any other
    payload is decoded inline (backward compatible with pre-handoff XComs and small lists).
    """
    if items is None:
        return []
    if len(items) == 1 and items[0].get("media_type") == REFS_FILE_MEDIA_TYPE:
        return refs_from_file(Path(items[0]["uri"]))
    return [ref_from_xcom(item) for item in items]
