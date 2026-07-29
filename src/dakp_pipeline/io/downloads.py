"""Source acquisition helpers.

Milestone 1 ships only the media-type inference and a clearly-stubbed HTTP downloader.
Real network acquisition (DailyMed/FAERS/Drugs@FDA downloaders with manifests and
checksums) lands in **Milestone 2**; the mock profile never reaches :func:`http_download`.
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


def http_download(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest``.

    Stub: real acquisition (idempotent, manifest/checksum, no destructive stashing) is
    implemented in Milestone 2. Calling this from a non-mock profile in Milestone 1 fails
    loudly rather than silently pretending to download.
    """
    msg = "http_download() is a Milestone-1 stub; real source acquisition lands in Milestone 2. The mock profile must never reach this path."
    raise NotImplementedError(msg)


__all__ = ["http_download", "infer_media_type"]
