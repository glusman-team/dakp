"""Drugs@FDA product/application extractor (stub).

Parses the tiny mock Drugs@FDA products fixture (TSV) into a normalized products interim
table. Real extraction (product/application/submission tables, NDA/BLA/ANDA with and
without leading zeroes) lands in **Milestone 3**.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.paths import Workdir

_PRODUCTS_COLUMNS = ["appl_no", "appl_type", "product_no", "drug_name", "active_ingredient", "marketing_status_name"]


class DrugsFDAProductsExtractor:
    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        store = ArtifactStore(Workdir(ctx.workdir))
        refs: list[ArtifactRef] = []
        for ref in inputs:
            if not ref.uri.name.lower().endswith(".tsv") or "drugsfda" not in ref.uri.name.lower():
                continue
            frame = pl.read_csv(ref.uri, separator="\t")
            # Normalize column names to the canonical lowercase contract.
            frame = frame.rename({c: c.strip().lower().replace(" ", "_") for c in frame.columns})
            frame = frame.select([c for c in _PRODUCTS_COLUMNS if c in frame.columns])
            # Normalize ApplNo to digits-only for consistent joins with FAERS nda.
            if "appl_no" in frame.columns:
                frame = frame.with_columns(pl.col("appl_no").cast(pl.Utf8).map_elements(_digits_only, return_dtype=pl.Utf8).alias("appl_no"))
            out = Workdir(ctx.workdir).interim / "drugsfda" / "products.parquet"
            rows_written = schemas.write_parquet(frame, out)
            ref_out = store.register(
                out,
                media_type=schemas.PARQUET_MEDIA_TYPE,
                rows=rows_written,
                schema_fingerprint=schemas.schema_fingerprint(_PRODUCTS_COLUMNS),
                inputs=[ref.blake3],
                operation=OperationBlock(name="extract_drugsfda_products"),
                table=TableBlock(rows=rows_written, schema_fingerprint=schemas.schema_fingerprint(_PRODUCTS_COLUMNS)),
            )
            refs.append(ref_out)
        return refs


def _digits_only(value: str | None) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())


extract = DrugsFDAProductsExtractor().extract

__all__ = ["DrugsFDAProductsExtractor", "extract"]
