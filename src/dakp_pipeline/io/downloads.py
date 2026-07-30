"""Source acquisition helpers.

Network acquisition lives in the source-specific fetchers. This module provides the shared
media-type inference used by artifact manifests.
"""

from __future__ import annotations

from pathlib import Path

# Suffix -> IANA-ish media type used in artifact manifests.
_MEDIA_TYPES: dict[str, str] = {
    ".xml": "application/xml",
    ".xml.gz": "application/gzip",
    ".gz": "application/gzip",
    ".zip": "application/zip",
    ".parquet": "application/vnd.apache.parquet",
    ".tsv": "text/tab-separated-values",
    ".csv": "text/csv",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def infer_media_type(path: Path) -> str:
    """Best-effort media type from a filename, handling compound suffixes (e.g. ``.xml.gz``)."""
    name = path.name.lower()
    for suffix in (".xml.gz", ".tsv", ".csv", ".jsonl", ".json", ".parquet", ".zip", ".gz", ".xml", ".txt", ".xlsx"):
        if name.endswith(suffix):
            return _MEDIA_TYPES[suffix]
    return "application/octet-stream"


__all__ = ["infer_media_type"]
