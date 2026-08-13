"""Tabular contracts: ordered column lists for every public TSV table, schema
fingerprints, and polars-backed read/write helpers.

Column lists are the public tabular contracts. Tablassert-facing tables are
uncompressed TSV (Tablassert cannot read compressed inputs); interim tables are parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.io.content_hash import hash_bytes

# --- public TSV column contracts -------------------------------------------------

DAILYMED_SPL_DOCUMENTS_COLUMNS = [
    "spl_document_id",
    "spl_set_id",
    "xml_path",
    "release_file",
    "approval_code",
    "approval_type",
    "loinc_code",
    "section_name",
    "section_text",
    "active_ingredient_name",
    "active_ingredient_unii",
]

APPROVED_TREATS_COLUMNS = [
    "subject_text",
    "subject_curie",
    "subject_name",
    "subject_category",
    "predicate",
    "object_text",
    "object_curie",
    "object_name",
    "object_category",
    "approval_ids",
    "supporting_spl_sets",
    "supporting_spl_documents",
    "clinical_approval_status",
    "knowledge_level",
    "agent_type",
    "primary_knowledge_source",
    "upstream_resource_ids",
]

FAERS_APPLIED_TO_TREAT_COLUMNS = [
    "subject_text",
    "subject_curie",
    "subject_name",
    "subject_category",
    "predicate",
    "object_text",
    "object_curie",
    "object_name",
    "object_category",
    "case_count",
    "clinical_approval_status",
    "knowledge_level",
    "agent_type",
    "primary_knowledge_source",
    "upstream_resource_ids",
]

CONTRAINDICATION_COLUMNS = [
    "subject_text",
    "subject_curie",
    "subject_name",
    "subject_category",
    "predicate",
    "object_text",
    "object_curie",
    "object_name",
    "object_category",
    "disease_context_text",
    "evidence_text",
    "supporting_spl_sets",
    "supporting_spl_documents",
    "source_score",
    "knowledge_level",
    "agent_type",
    "primary_knowledge_source",
    "upstream_resource_ids",
]

FAERS_CASES_COLUMNS = [
    "quarter",
    "primaryid",
    "caseid",
    "source",
    "occp_cod",
    "reporter_country",
    "drugname",
    "ingredient",
    "nda",
    "indication",
    "effects",
]

# Single registry: assertion table name -> ordered columns.
ASSERTION_TABLES: dict[str, list[str]] = {
    "approved_treats_assertions": APPROVED_TREATS_COLUMNS,
    "faers_applied_to_treat_assertions": FAERS_APPLIED_TO_TREAT_COLUMNS,
    "contraindication_assertions": CONTRAINDICATION_COLUMNS,
}


def schema_fingerprint(columns: list[str]) -> str:
    """Deterministic ``b3:<hex>`` fingerprint of an ordered column list.

    Captured in artifact manifests so schema drift between runs is detectable by hash.
    """
    return hash_bytes("\t".join(columns).encode("utf-8"))


def columns_for(table: str) -> list[str]:
    """Return the ordered columns for a named assertion table (raises KeyError if unknown)."""
    if table not in ASSERTION_TABLES:
        msg = f"unknown assertion table {table!r}; expected one of: {sorted(ASSERTION_TABLES)}"
        raise KeyError(msg)
    return list(ASSERTION_TABLES[table])


# --- polars-backed I/O helpers ---------------------------------------------------

TSV_MEDIA_TYPE = "text/tab-separated-values"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


def write_tsv(frame: pl.DataFrame, path: Path) -> int:
    """Write an uncompressed TSV with a header row (Tablassert-readable). Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path, separator="\t")
    return frame.height


def write_parquet(frame: pl.DataFrame, path: Path) -> int:
    """Write a parquet interim table. Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame.height


def read_table(path: Path) -> pl.DataFrame:
    """Read a table by suffix: parquet -> :func:`pl.read_parquet`, else TSV/CSV."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, separator="\t")


__all__ = [
    "APPROVED_TREATS_COLUMNS",
    "ASSERTION_TABLES",
    "CONTRAINDICATION_COLUMNS",
    "DAILYMED_SPL_DOCUMENTS_COLUMNS",
    "FAERS_APPLIED_TO_TREAT_COLUMNS",
    "FAERS_CASES_COLUMNS",
    "PARQUET_MEDIA_TYPE",
    "TSV_MEDIA_TYPE",
    "columns_for",
    "read_table",
    "schema_fingerprint",
    "write_parquet",
    "write_tsv",
]
