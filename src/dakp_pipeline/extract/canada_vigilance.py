"""Canada Vigilance extraction.

Extracts the drug-indication observations out of Health Canada's Canada Vigilance data extract
ZIP into the normalized interim table the observed-use shaper consumes
(``data/interim/canada_vigilance/cv_indications.parquet``).

The extract is a series of ``$``-delimited, quote-enclosed ASCII files with NO header row
(the data-structures page documents the column order). Only three of the thirteen members are
needed, joined on the published keys:

* ``report_drug_indication.txt`` (REPORT_DRUG_ID, REPORT_ID, DRUG_PRODUCT_ID, DRUGNAME,
  INDICATION_NAME_ENG, INDICATION_NAME_FR) — the per-report indication text;
* ``report_drug.txt`` (REPORT_DRUG_ID, ... DRUGINVOLV_ENG ...) — the health-product role; only
  ``Suspect`` drugs contribute (a concomitant drug's indication is not an observed use of that
  drug, mirroring the FAERS suspect-drug discipline);
* ``drug_product_ingredients.txt`` (DRUG_PRODUCT_ID, ..., ACTIVE_INGREDIENT_NAME) — the active
  ingredient fan-out (a product's ingredient list resolves the subject, like the EMA registry
  join does for SmPC rows; ``DRUGNAME`` remains the fallback subject for products the
  ingredients member does not cover).

Indication text is kept RAW: the observed-use shaper owns normalization (``defaersify``), the
non-disease stop-list, and object resolution, exactly as it does for FAERS ``indi_pt`` text.
Row identity: one output row per ``(REPORT_DRUG_ID, INDICATION_NAME_ENG, ingredient)``
observation with ``source_record_id = cv:<report_id>:<report_drug_id>:<ingredient-or-brand>``.
The French indication column is intentionally unused (English-only edges; the shaper's stop
list and disease map are English).

Outputs (under ``interim/canada_vigilance/``):

* ``cv_indications.parquet`` — the joined observation rows (returned first).
* ``cv_warnings.parquet`` — parse/coverage warnings (missing members, empty extracts).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import bind, stats
from dakp_pipeline.paths import Workdir

CV_INDICATIONS_COLUMNS: list[str] = [
    "source_record_id",
    "report_id",
    "report_drug_id",
    "drug_product_id",
    "drugname",
    "ingredient",
    "indication",
    "source_file",
]
CV_WARNINGS_COLUMNS: list[str] = ["file", "code", "message", "count"]

# Member basenames inside the ZIP (lowercase in the published extract; matched case-insensitively).
_MEMBER_INDICATIONS = "report_drug_indication.txt"
_MEMBER_REPORT_DRUG = "report_drug.txt"
_MEMBER_INGREDIENTS = "drug_product_ingredients.txt"
#: Only ``Suspect`` health products contribute observed-use rows.
_ROLE_SUSPECT = "Suspect"

#: Narration prefix for every log line this extractor emits (one stat per line).
_EVENT = "extract_canada_vigilance"


class CanadaVigilanceExtractor:
    """Parse the Canada Vigilance data-extract ZIP into the normalized indication table."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        log = bind(task_id=_EVENT)

        zip_ref = next((ref for ref in inputs if ref.uri.suffix.lower() == ".zip"), None)
        if zip_ref is None:
            msg = "no Canada Vigilance data-extract ZIP among the inputs"
            raise ValueError(msg)

        rows, warnings = parse_extract(zip_ref.uri)

        operation = OperationBlock(name=_EVENT)
        input_ids = [ref.blake3 for ref in inputs]
        interim_dir = wd.interim / "canada_vigilance"
        indications_fp = schemas.schema_fingerprint(CV_INDICATIONS_COLUMNS)
        warnings_fp = schemas.schema_fingerprint(CV_WARNINGS_COLUMNS)
        refs = [
            _write_parquet(
                rows, CV_INDICATIONS_COLUMNS, interim_dir / "cv_indications.parquet", store, operation, indications_fp, len(warnings), input_ids
            ),
            _write_parquet(
                warnings, CV_WARNINGS_COLUMNS, interim_dir / "cv_warnings.parquet", store, operation, warnings_fp, len(warnings), input_ids
            ),
        ]
        stats(log, _EVENT, rows=len(rows), warnings=len(warnings), outputs=len(refs))
        return refs


@dataclass
class _Warnings:
    """Accumulates warning rows (dedup + counted), like the FAERS extractor's audit."""

    items: list[dict[str, str]] = field(default_factory=list)

    def add(self, file: str, code: str, message: str) -> None:
        self.items.append({"file": file, "code": code, "message": message, "count": "1"})

    def sorted(self) -> list[dict[str, str]]:
        return sorted(self.items, key=lambda w: (w["file"], w["code"], w["message"]))


def parse_extract(zip_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse the extract ZIP into observation rows + warnings (pure; deterministic order)."""
    warnings = _Warnings()
    members = _members(zip_path, warnings)
    if members is None:
        return [], warnings.sorted()

    indications = _read_member(
        members[_MEMBER_INDICATIONS],
        _MEMBER_INDICATIONS,
        ("report_drug_id", "report_id", "drug_product_id", "drugname", "indication"),
        (0, 1, 2, 3, 4),
        warnings,
    )
    roles = _read_member(members[_MEMBER_REPORT_DRUG], _MEMBER_REPORT_DRUG, ("report_drug_id", "role"), (0, 4), warnings)
    ingredients = _read_member(members[_MEMBER_INGREDIENTS], _MEMBER_INGREDIENTS, ("drug_product_id", "ingredient"), (1, 4), warnings)
    if indications is None or roles is None or ingredients is None:
        return [], warnings.sorted()

    suspects = roles.filter(pl.col("role") == _ROLE_SUSPECT)
    joined = (
        indications.join(suspects, on="report_drug_id", how="inner")
        .join(ingredients.filter(pl.col("ingredient") != ""), on="drug_product_id", how="left")
        .select("report_id", "report_drug_id", "drug_product_id", "drugname", "indication", pl.col("ingredient").fill_null(""))
        .filter((pl.col("indication") != "") & (pl.col("drugname") != ""))
        .unique()
        .sort("report_id", "report_drug_id", "indication", "ingredient")
    )
    unknown_products = joined.filter(pl.col("ingredient") == "").select("drug_product_id").unique().height if joined.height else 0
    if unknown_products:
        warnings.add(
            _MEMBER_INGREDIENTS,
            "unknown_drug_product",
            f"{unknown_products} drug_product_id(s) absent from the ingredients member (subject falls back to the brand name)",
        )

    rows = [
        {
            "source_record_id": f"cv:{rec['report_id']}:{rec['report_drug_id']}:{rec['ingredient'] or rec['drugname']}",
            "report_id": str(rec["report_id"]),
            "report_drug_id": str(rec["report_drug_id"]),
            "drug_product_id": str(rec["drug_product_id"]),
            "drugname": str(rec["drugname"]),
            "ingredient": str(rec["ingredient"]),
            "indication": str(rec["indication"]),
            "source_file": _MEMBER_INDICATIONS,
        }
        for rec in joined.iter_rows(named=True)
    ]
    return rows, warnings.sorted()


def _members(zip_path: Path, warnings: _Warnings) -> dict[str, IO[bytes]] | None:
    """Open the extract ZIP and map the three needed basenames to handles (case-insensitive)."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        warnings.add("", "zip_unreadable", f"{type(exc).__name__}: {exc}")
        return None
    wanted = {_MEMBER_INDICATIONS, _MEMBER_REPORT_DRUG, _MEMBER_INGREDIENTS}
    found: dict[str, IO[bytes]] = {}
    for name in archive.namelist():
        base = Path(name).name.lower()
        if base in wanted:
            found[base] = archive.open(name)
    missing = sorted(wanted - set(found))
    if missing:
        warnings.add("", "missing_member", f"data-extract members absent: {', '.join(missing)}")
        return None
    return found


def _read_member(handle: IO[bytes], file: str, names: tuple[str, ...], indices: tuple[int, ...], warnings: _Warnings) -> pl.DataFrame | None:
    """Read one extract member with explicit (no header) columns; a parse failure is a warning.

    ``indices`` are the positional projections from the published column order; ``names`` the
    normalized names. Every column stays text (``infer_schema_length=0``): ids are joined as
    strings, never coerced to numbers.
    """
    try:
        # Positional projection AFTER the read (``new_columns`` applies to the pre-projection
        # schema): the published column order is the contract, ids stay text.
        frame = pl.read_csv(handle, has_header=False, separator="$", quote_char='"', infer_schema_length=0).select(
            [pl.nth(index).alias(name) for index, name in zip(indices, names, strict=True)]
        )
    except Exception as exc:
        warnings.add(file, "parse_failed", f"{type(exc).__name__}: {exc}")
        return None
    return frame


def _write_parquet(
    rows: list[dict[str, str]],
    columns: list[str],
    out: Path,
    store: ArtifactStore,
    operation: OperationBlock,
    fingerprint: str,
    warnings: int,
    inputs: list[str],
) -> ArtifactRef:
    frame = pl.DataFrame(rows, schema=dict.fromkeys(columns, pl.Utf8)) if rows else pl.DataFrame(schema=dict.fromkeys(columns, pl.Utf8))
    rows_written = schemas.write_parquet(frame, out)
    return store.register(
        out,
        media_type=schemas.PARQUET_MEDIA_TYPE,
        rows=rows_written,
        schema_fingerprint=fingerprint,
        inputs=inputs,
        operation=operation,
        table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint, warnings=warnings),
    )


extract = CanadaVigilanceExtractor().extract

__all__ = ["CV_INDICATIONS_COLUMNS", "CV_WARNINGS_COLUMNS", "CanadaVigilanceExtractor", "extract", "parse_extract"]
