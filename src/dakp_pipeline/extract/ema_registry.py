"""EMA centrally-authorised medicines extractor.

Parses the EMA "Medicines" report xlsx (the fixed-name nightly bulk export acquired by
:mod:`dakp_pipeline.sources.ema`) into one normalized interim parquet table
(``data/interim/ema/ema_registry.parquet``) for the assertion-shaping stage.

Design notes:

* The export opens with banner lines ("Content type:" / "Medicine", "Output automatically
  generated from content on www.ema.europa.eu on: <date>") ABOVE the real header row, so the
  parser locates the header row — the row carrying "Name of medicine" and "Medicine status" —
  instead of assuming row 0.
* Only rows with ``Medicine status == "Authorised"`` AND ``Category == "Human"`` survive: every
  other status (Withdrawn / Refused / Suspended / ...) is dropped entirely, and veterinary rows
  carry no usable object (the "Therapeutic area (MeSH)" column is human-only).
* The full free-text "Therapeutic indication" column is kept verbatim in the interim table —
  Phase 2 mines it with the DiseaseNER for ``infores:epar`` edges.
* Multi-valued cells stay semicolon-joined here (``active_substance`` / ``therapeutic_area_mesh``);
  the approved-treats shaper owns the one-edge-per-(substance, MeSH-term) fan-out.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import bind, stats
from dakp_pipeline.paths import Workdir

# --- normalized column contract ---------------------------------------------------

EMA_REGISTRY_COLUMNS = [
    "medicine_name",
    "ema_product_number",
    "category",
    "medicine_status",
    "inn",
    "active_substance",
    "therapeutic_area_mesh",
    "therapeutic_indication",
    "medicine_url",
]

#: Source xlsx column -> normalized interim column. Matched exactly against the located header
#: row (none of these names carries the newline some sibling columns do).
_COLUMN_MAP: dict[str, str] = {
    "Category": "category",
    "Name of medicine": "medicine_name",
    "EMA product number": "ema_product_number",
    "Medicine status": "medicine_status",
    "International non-proprietary name (INN) / common name": "inn",
    "Active substance": "active_substance",
    "Therapeutic area (MeSH)": "therapeutic_area_mesh",
    "Therapeutic indication": "therapeutic_indication",
    "Medicine URL": "medicine_url",
}

#: Header-row markers: the row carrying BOTH is the real header (banner lines above it are not).
_HEADER_MARKERS = frozenset({"Name of medicine", "Medicine status"})

#: Only centrally AUTHORISED, HUMAN medicines produce assertion input (see module docstring).
_STATUS_KEPT = "Authorised"
_CATEGORY_KEPT = "Human"


class EMARegistryExtractor:
    """Parse the EMA medicines xlsx into a normalized interim parquet table."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        log = bind(task_id="extract_ema_registry")

        xlsx_ref = next((ref for ref in inputs if ref.uri.suffix.lower() == ".xlsx"), None)
        if xlsx_ref is None:
            msg = "no EMA medicines xlsx among the inputs"
            raise ValueError(msg)

        frame = parse_ema_registry(xlsx_ref.uri)
        out = wd.interim / "ema" / "ema_registry.parquet"
        rows = schemas.write_parquet(frame.select(EMA_REGISTRY_COLUMNS), out)
        fingerprint = schemas.schema_fingerprint(EMA_REGISTRY_COLUMNS)
        ref = store.register(
            out,
            media_type=schemas.PARQUET_MEDIA_TYPE,
            rows=rows,
            schema_fingerprint=fingerprint,
            inputs=[item.blake3 for item in inputs],
            operation=OperationBlock(name="extract_ema_registry"),
            table=TableBlock(rows=rows, schema_fingerprint=fingerprint),
        )
        stats(log, "extract_ema_registry", rows=rows, outputs=1, source=str(xlsx_ref.uri))
        return [ref]


def parse_ema_registry(path: Path) -> pl.DataFrame:
    """Parse the EMA medicines xlsx at ``path`` into the normalized interim frame (pure).

    Locates the real header row below the banner lines, keeps only Authorised + Human rows, and
    projects the normalized contract columns (all UTF-8 strings, trimmed, sorted by
    ``(ema_product_number, medicine_name)`` for deterministic output).
    """
    raw = pl.read_excel(path, has_header=False)
    header_row = _locate_header_row(raw)
    header = ["" if value is None else str(value).strip() for value in raw.row(header_row)]
    data = raw.slice(header_row + 1)
    data.columns = header

    missing = [source for source in _COLUMN_MAP if source not in data.columns]
    if missing:
        msg = f"EMA medicines table is missing required columns: {missing}"
        raise ValueError(msg)

    return (
        data.filter(
            (pl.col("Category").cast(pl.Utf8).str.strip_chars() == _CATEGORY_KEPT)
            & (pl.col("Medicine status").cast(pl.Utf8).str.strip_chars() == _STATUS_KEPT)
        )
        .select([pl.col(source).cast(pl.Utf8).fill_null("").str.strip_chars().alias(target) for source, target in _COLUMN_MAP.items()])
        .select(EMA_REGISTRY_COLUMNS)
        .sort("ema_product_number", "medicine_name")
    )


def _locate_header_row(raw: pl.DataFrame) -> int:
    """Index of the real header row (banner lines precede it); ValueError when absent."""
    for index in range(raw.height):
        values = {"" if value is None else str(value).strip() for value in raw.row(index)}
        if values >= _HEADER_MARKERS:
            return index
    msg = "no EMA header row (looking for 'Name of medicine' + 'Medicine status') found in the workbook"
    raise ValueError(msg)


extract = EMARegistryExtractor().extract

__all__ = ["EMA_REGISTRY_COLUMNS", "EMARegistryExtractor", "extract", "parse_ema_registry"]
