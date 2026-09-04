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
    "FDA_regulatory_approvals",
    "supporting_spl_sets",
    "supporting_spl_documents",
    "supporting_spl_evidence",
    "clinical_approval_status",
    "knowledge_level",
    "agent_type",
    "primary_knowledge_source",
    "upstream_resource_ids",
    "edge_evidence",
    "supporting_faers_records",
    "supporting_faers_urls",
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
    "number_of_cases",
    "case_ids",
    "clinical_approval_status",
    "knowledge_level",
    "agent_type",
    "primary_knowledge_source",
    "upstream_resource_ids",
    "FDA_regulatory_approvals",
    "edge_evidence",
    "supporting_faers_records",
    "supporting_faers_urls",
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
    "supporting_spl_evidence",
    "source_score",
    "knowledge_level",
    "agent_type",
    "primary_knowledge_source",
    "upstream_resource_ids",
    "FDA_regulatory_approvals",
    "edge_evidence",
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


def coerce_count_column(frame: pl.DataFrame, column: str = "number_of_cases") -> pl.DataFrame:
    """Rewrite float-formatted whole-number cells (``"55.0"``) as integer strings (``"55"``).

    Tablassert claims ``number_of_cases`` onto the integer-ranged Biolink slot only when the TSV
    cell parses as an integer; a float-formatted cell fails that coercion and the count falls out
    of the edge into a ``has_supporting_studies`` study description (the v1.4.0 release shipped
    three ``applied_to_treat`` edges whose case counts survived only as ``number_of_cases=55.0``
    description text). Applied to every assertion table in
    :func:`dakp_pipeline.assertions.evidence.write_assertion_table`, i.e. before any Tablassert
    handoff. A non-empty cell that is not a finite whole number is a shaper data bug and raises.
    """
    if column not in frame.columns or frame.is_empty():
        return frame
    cells = frame.get_column(column).cast(pl.Utf8).str.strip_chars()
    values = cells.cast(pl.Float64, strict=False)  # null where the cell is not numeric
    ok = (((values.is_finite()) & (values == values.floor())) | (cells == "")).fill_null(False)
    if not ok.all():
        offenders = cells.filter(~ok).unique().sort().head(5).to_list()
        msg = f"column {column!r} must hold empty or whole-number cells; offenders: {offenders}"
        raise ValueError(msg)
    expr = pl.when(pl.col(column).cast(pl.Utf8).str.strip_chars() == "").then(pl.lit("")).otherwise(values.cast(pl.Int64).cast(pl.Utf8)).alias(column)
    return frame.with_columns(expr)


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
    "coerce_count_column",
    "columns_for",
    "read_table",
    "schema_fingerprint",
    "write_parquet",
    "write_tsv",
]
