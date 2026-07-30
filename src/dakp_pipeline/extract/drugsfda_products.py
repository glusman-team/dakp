"""Drugs@FDA product/application/submission extractor (Milestone 3).

Parses the FDA Drugs@FDA tab-delimited data files (the ``media/89850`` ZIP contents —
``Products.txt`` / ``Applications.txt`` / ``Submissions.txt`` — or their fixture mirrors)
into normalized interim parquet tables, lookup tables, and one uncompressed TSV
source-section table (``data/tabular/drugsfda_products.tsv``) for Tablassert handoff.

Application-number normalization ports the semantics of the legacy
``FAERS/bin/drug2indi.pl readNDAproducts`` (``s/^(NDA|BLA|ANDA)0*(.+)/``): the raw
``APPLICATIONNUMBER`` is preserved *and* both normalized forms are kept — digits with
leading zeroes preserved (e.g. ``012345``) and leading zeroes stripped (e.g. ``12345``) —
so NDA/BLA/ANDA variants join robustly with FAERS ``nda`` values regardless of padding.

Design notes:

* All source columns are read as UTF-8 strings (``infer_schema_length=0``) so leading
  zeroes and the ``NDA/BLA/ANDA`` prefixes survive polars type inference.
* Column mapping is name-and-alias based (``ApplicationNumber`` vs ``ApplNo``), so the
  parser tolerates the real Drugs@FDA schema, the legacy NDC ``product.txt`` schema, and
  the fixture subset alike.
* Parsing is pure functions over frames/paths wherever possible (PLAN.md extraction
  requirement), making file inputs easy to monkeypatch in tests.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.paths import Workdir
from dakp_pipeline.workers import go_runner

# --- normalized column contracts ------------------------------------------------

PRODUCTS_COLUMNS = [
    "source_record_id",
    "source_file",
    "appl_no_raw",
    "appl_type",
    "appl_no",
    "appl_no_stripped",
    "product_no",
    "drug_name",
    "active_ingredient",
    "form",
    "route",
    "strength",
    "reference_drug",
    "reference_standard",
    "product_ndc",
    "marketing_status_name",
]

APPLICATIONS_COLUMNS = [
    "source_record_id",
    "source_file",
    "appl_no_raw",
    "appl_type",
    "appl_no",
    "appl_no_stripped",
    "sponsor_name",
    "common_or_original_name",
    "submission_classification",
    "orphan_status",
]

SUBMISSIONS_COLUMNS = [
    "source_record_id",
    "source_file",
    "appl_no_raw",
    "appl_type",
    "appl_no",
    "appl_no_stripped",
    "submission_type",
    "submission_no",
    "submission_status",
    "submission_status_date",
    "submission_notes",
]

LOOKUPS_COLUMNS = ["lookup_type", "term", "appl_no", "appl_no_stripped", "appl_type"]

# Source column aliases (normalized, space/underscore-insensitive) -> canonical field.
# Multiple source spellings map to one canonical column.
_PRODUCT_FIELD_ALIASES: dict[str, list[str]] = {
    "appl_no_raw": ["applicationnumber"],
    "appl_type": ["appltype", "applicationtype"],
    "appl_no": ["applno"],
    "product_no": ["productno"],
    "drug_name": ["drugname", "proprietaryname"],
    "active_ingredient": ["activeingredient", "nonproprietaryname"],
    "form": ["form", "dosageformname"],
    "route": ["route", "routename"],
    "strength": ["strength"],
    "reference_drug": ["referencedrug"],
    "reference_standard": ["referencestandard"],
    "product_ndc": ["productndc"],
    "marketing_status_name": ["marketingstatusname", "marketingstatusdescription"],
}
_APPLICATION_FIELD_ALIASES: dict[str, list[str]] = {
    "appl_no_raw": ["applicationnumber"],
    "appl_type": ["appltype"],
    "appl_no": ["applno"],
    "sponsor_name": ["sponsorname", "labelername"],
    "common_or_original_name": ["commonororiginalname"],
    "submission_classification": ["submissionclassification"],
    "orphan_status": ["orphanstatus"],
}
_SUBMISSION_FIELD_ALIASES: dict[str, list[str]] = {
    "appl_no_raw": ["applicationnumber"],
    "appl_type": ["appltype"],
    "appl_no": ["applno"],
    "submission_type": ["submissiontype"],
    "submission_no": ["submissionno"],
    "submission_status": ["submissionstatus"],
    "submission_status_date": ["submissionstatusdate"],
    "submission_notes": ["submissionspublicnotes"],
}

# (appl_type, digits_with_zeroes) extraction from a combined APPLICATIONNUMBER.
_APPL_PREFIX_RE = re.compile(r"^(NDA|BLA|ANDA)\s*(\d+)", re.IGNORECASE)

# Multi-ingredient separator in Drugs@FDA ActiveIngredient (e.g. "EZETIMIBE; SIMVASTATIN").
_INGREDIENT_SEPARATOR = ";"


class DrugsFDAProductsExtractor:
    """Parse Drugs@FDA tab-delimited tables into normalized parquet + a Tablassert TSV."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        if go_runner.should_use_go(ctx):
            return self._extract_via_go(inputs, ctx)
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        log = bind(task_id="extract_drugsfda_products")
        warnings: list[dict[str, str]] = []
        refs: list[ArtifactRef] = []

        tables = _collect_tables(inputs)
        products = tables.get("products")
        applications = tables.get("applications")
        submissions = tables.get("submissions")
        if not tables:
            warnings.append({"table": "*", "message": "no Drugs@FDA tables recognized in inputs"})

        # appl_no_stripped -> appl_type map (products first, then applications) so
        # submissions — which carry only ApplNo in the real schema — inherit appl_type.
        appl_type_map: dict[str, str] = {}

        # 1. Products (required core output) -----------------------------------
        products_frame: pl.DataFrame | None = None
        if products is not None:
            frame, source_name, source_blake3 = products
            products_frame, p_warnings = _build_products(frame, source_name)
            warnings.extend(p_warnings)
            _fill_appl_type_map(appl_type_map, products_frame)
            refs.append(self._register_parquet(wd, store, products_frame, "products", PRODUCTS_COLUMNS, [source_blake3], len(p_warnings)))
            # Uncompressed source-section TSV for Tablassert handoff (PLAN.md: Tablassert
            # cannot read compressed inputs).
            tsv_out = wd.tabular / "drugsfda_products.tsv"
            schemas.write_tsv(products_frame.select(PRODUCTS_COLUMNS), tsv_out)
            refs.append(self._register_tsv(store, tsv_out, PRODUCTS_COLUMNS, [source_blake3], len(p_warnings)))
        else:
            warnings.append({"table": "products", "message": "no Products table found in inputs"})

        # 2. Applications (optional) -------------------------------------------
        if applications is not None:
            frame, source_name, source_blake3 = applications
            appl_frame, a_warnings = _build_applications(frame, source_name)
            warnings.extend(a_warnings)
            _fill_appl_type_map(appl_type_map, appl_frame)
            refs.append(self._register_parquet(wd, store, appl_frame, "applications", APPLICATIONS_COLUMNS, [source_blake3], len(a_warnings)))
        else:
            warnings.append({"table": "applications", "message": "no Applications table found in inputs"})

        # 3. Submissions (optional; inherit appl_type via the map) --------------
        if submissions is not None:
            frame, source_name, source_blake3 = submissions
            sub_frame, s_warnings = _build_submissions(frame, source_name, appl_type_map)
            warnings.extend(s_warnings)
            refs.append(self._register_parquet(wd, store, sub_frame, "submissions", SUBMISSIONS_COLUMNS, [source_blake3], len(s_warnings)))
        else:
            warnings.append({"table": "submissions", "message": "no Submissions table found in inputs"})

        # 4. Lookup tables (name/ingredient/ndc/marketing-status -> appl_no) ---
        if products_frame is not None:
            lookups_frame = _build_lookups(products_frame)
            lookups_blake = _input_blake3(products, inputs)
            refs.append(self._register_parquet(wd, store, lookups_frame, "lookups", LOOKUPS_COLUMNS, lookups_blake, 0))

        # 5. Parse warnings (deterministic provenance record) ------------------
        warnings_out = wd.interim / "drugsfda" / "extract_warnings.jsonl"
        _write_warnings_jsonl(warnings_out, warnings)
        refs.append(
            store.register(
                warnings_out,
                media_type="application/x-ndjson",
                rows=len(warnings),
                inputs=[ref.blake3 for ref in inputs],
                operation=OperationBlock(name="extract_drugsfda_warnings"),
                table=TableBlock(rows=len(warnings)),
            )
        )

        rows_total = products_frame.height if products_frame is not None else 0
        log.info("extracted Drugs@FDA", products=rows_total, warnings=len(warnings), outputs=len(refs))
        return refs

    # -- registration helpers -------------------------------------------------
    def _register_parquet(
        self, wd: Workdir, store: ArtifactStore, frame: pl.DataFrame, table_name: str, columns: list[str], inputs: list[str], warning_count: int
    ) -> ArtifactRef:
        out = wd.interim / "drugsfda" / f"{table_name}.parquet"
        rows = schemas.write_parquet(frame.select(columns), out)
        fingerprint = schemas.schema_fingerprint(columns)
        return store.register(
            out,
            media_type=schemas.PARQUET_MEDIA_TYPE,
            rows=rows,
            schema_fingerprint=fingerprint,
            inputs=inputs,
            operation=OperationBlock(name=f"extract_drugsfda_{table_name}"),
            table=TableBlock(rows=rows, schema_fingerprint=fingerprint, warnings=warning_count),
        )

    def _register_tsv(self, store: ArtifactStore, path: Path, columns: list[str], inputs: list[str], warning_count: int) -> ArtifactRef:
        rows = _tsv_row_count(path)
        fingerprint = schemas.schema_fingerprint(columns)
        return store.register(
            path,
            media_type=schemas.TSV_MEDIA_TYPE,
            rows=rows,
            schema_fingerprint=fingerprint,
            inputs=inputs,
            operation=OperationBlock(name="extract_drugsfda_products_section"),
            table=TableBlock(rows=rows, schema_fingerprint=fingerprint, warnings=warning_count),
        )

    def _extract_via_go(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        """Delegate parsing to the Go ``drugsfda`` worker, repackaging its TSVs identically.

        Go reads the tab-delimited tables and writes products/applications/submissions/lookups
        as uncompressed TSV (byte-for-byte parity; see ``go/internal/drugsfda``). We read them
        back through the SAME parquet/TSV registration as the Python path so the returned refs
        (names, order, schemas) are unchanged.
        """
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        log = bind(task_id="extract_drugsfda_products")
        input_ids = [ref.blake3 for ref in inputs]

        with tempfile.TemporaryDirectory() as scratch:
            stage = Path(scratch)
            in_dir = go_runner.stage_inputs(inputs, stage / "in")
            out_dir = stage / "out"
            out_dir.mkdir()
            result = go_runner.get_runner().run_table("drugsfda", in_dir, out_dir)
            frames = {
                name: go_runner.read_go_tsv(out_dir / f"{name}.tsv")
                for name in ("drugsfda_products", "drugsfda_applications", "drugsfda_submissions", "drugsfda_lookups")
                if (out_dir / f"{name}.tsv").exists()
            }
        warnings = go_runner.go_warnings(result)
        refs: list[ArtifactRef] = []

        products = frames.get("drugsfda_products")
        if products is not None:
            refs.append(self._register_parquet(wd, store, products, "products", PRODUCTS_COLUMNS, input_ids, warnings))
            tsv_out = wd.tabular / "drugsfda_products.tsv"
            schemas.write_tsv(products.select(PRODUCTS_COLUMNS), tsv_out)
            refs.append(self._register_tsv(store, tsv_out, PRODUCTS_COLUMNS, input_ids, warnings))
        applications = frames.get("drugsfda_applications")
        if applications is not None:
            refs.append(self._register_parquet(wd, store, applications, "applications", APPLICATIONS_COLUMNS, input_ids, warnings))
        submissions = frames.get("drugsfda_submissions")
        if submissions is not None:
            refs.append(self._register_parquet(wd, store, submissions, "submissions", SUBMISSIONS_COLUMNS, input_ids, warnings))
        lookups = frames.get("drugsfda_lookups")
        if lookups is not None:
            refs.append(self._register_parquet(wd, store, lookups, "lookups", LOOKUPS_COLUMNS, input_ids, 0))

        # Parse-warning provenance record (Go records warnings in its slog stream; the JSONL is
        # kept empty so the ref set matches the Python path).
        warnings_out = wd.interim / "drugsfda" / "extract_warnings.jsonl"
        _write_warnings_jsonl(warnings_out, [])
        refs.append(
            store.register(
                warnings_out,
                media_type="application/x-ndjson",
                rows=0,
                inputs=input_ids,
                operation=OperationBlock(name="extract_drugsfda_warnings"),
                table=TableBlock(rows=0),
            )
        )
        rows_total = products.height if products is not None else 0
        log.info("extracted Drugs@FDA via Go worker", products=rows_total, warnings=warnings, outputs=len(refs))
        return refs


extract = DrugsFDAProductsExtractor().extract

__all__ = ["APPLICATIONS_COLUMNS", "LOOKUPS_COLUMNS", "PRODUCTS_COLUMNS", "SUBMISSIONS_COLUMNS", "DrugsFDAProductsExtractor", "extract"]


# --- input collection (ZIP members or loose TSV/TXT) ----------------------------


def _collect_tables(inputs: list[ArtifactRef]) -> dict[str, tuple[pl.DataFrame, str, str]]:
    """Return ``{table_key: (frame, source_name, source_blake3)}`` for recognized tables."""
    collected: dict[str, tuple[pl.DataFrame, str, str]] = {}
    for ref in inputs:
        name = ref.uri.name.lower()
        if name.endswith(".zip"):
            for key, frame, source_name in _read_zip_tables(ref.uri):
                collected[key] = (frame, source_name, ref.blake3)
            continue
        key = _table_key(ref.uri.name)
        if key is not None:
            collected[key] = (_read_tsv(ref.uri), ref.uri.name, ref.blake3)
    return collected


def _read_zip_tables(path: Path) -> list[tuple[str, pl.DataFrame, str]]:
    out: list[tuple[str, pl.DataFrame, str]] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            key = _table_key(info.filename)
            if key is None:
                continue
            with zf.open(info) as handle:
                out.append((key, _read_tsv_bytes(handle.read()), info.filename))
    return out


def _table_key(filename: str) -> str | None:
    """Classify a Drugs@FDA table filename into ``products``/``applications``/``submissions``.

    Matches by stem suffix so it accepts both the real files (``Products.txt``) and the
    fixture mirrors (``drugsfda_products.tsv``) while rejecting sub-tables like
    ``SubmissionPropertyType.txt``.
    """
    stem = Path(filename).stem.lower()
    if stem.endswith("products") or stem == "product":
        return "products"
    if stem.endswith("applications") or stem == "application":
        return "applications"
    if stem.endswith("submissions") or stem == "submission":
        return "submissions"
    return None


def _read_tsv(path: Path) -> pl.DataFrame:
    return _read_tsv_bytes(path.read_bytes())


def _read_tsv_bytes(data: bytes) -> pl.DataFrame:
    # infer_schema_length=0 -> every column is Utf8, preserving leading zeroes and
    # NDA/BLA/ANDA prefixes that integer inference would silently strip.
    return pl.read_csv(io.BytesIO(data), separator="\t", infer_schema_length=0)


# --- column normalization -------------------------------------------------------


def _norm_key(column: str) -> str:
    """Case/space/underscore-insensitive key used to match source columns to aliases."""
    return column.strip().lower().lstrip("\ufeff").replace(" ", "").replace("_", "")


def _rename_to_canonical(frame: pl.DataFrame, field_aliases: dict[str, list[str]]) -> pl.DataFrame:
    """Rename recognized source columns to their canonical field names (others untouched)."""
    src_to_orig: dict[str, str] = {}
    for column in frame.columns:
        src_to_orig.setdefault(_norm_key(column), column)
    rename: dict[str, str] = {}
    for canonical, aliases in field_aliases.items():
        if canonical in rename:
            continue
        for alias in aliases:
            orig = src_to_orig.get(_norm_key(alias))
            if orig is not None and orig not in rename.values():
                rename[orig] = canonical
                break
    return frame.rename(rename) if rename else frame


def _field(rec: dict[str, Any], name: str) -> str:
    value = rec.get(name)
    return "" if value is None else str(value)


# --- application-number normalization (ports legacy readNDAproducts) ------------


def _digits_only(value: str) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())


def _parse_combined(value: str) -> tuple[str, str]:
    """Split a combined APPLICATIONNUMBER into ``(appl_type, digits_with_zeroes)``."""
    text = (value or "").strip()
    if not text:
        return ("", "")
    match = _APPL_PREFIX_RE.match(text)
    if match:
        return (match.group(1).upper(), match.group(2))
    return ("", _digits_only(text))


def _normalize_appl_fields(appl_no_raw_src: str, appl_type_src: str, appl_no_src: str) -> tuple[str, str, str, str]:
    """Return ``(appl_no_raw, appl_type, appl_no, appl_no_stripped)``.

    Handles both NDC-style combined ``APPLICATIONNUMBER`` (``NDA012345``) and Drugs@FDA
    split ``ApplType``+``ApplNo`` (``NDA``, ``012345``). ``appl_no`` keeps leading zeroes;
    ``appl_no_stripped`` removes them — mirroring legacy ``s/^(NDA|BLA|ANDA)0*(.+)/``.
    """
    raw_src = (appl_no_raw_src or "").strip()
    atype = (appl_type_src or "").strip().upper()
    ano = (appl_no_src or "").strip()

    if raw_src:
        prefix, digits = _parse_combined(raw_src)
        if prefix and not atype:
            atype = prefix
        if not digits:
            digits = _digits_only(ano)
    else:
        digits = _digits_only(ano)

    appl_no = digits
    stripped = appl_no.lstrip("0")
    appl_no_stripped = stripped if stripped else appl_no  # keep all-zero/empty as-is
    appl_no_raw = f"{atype}{appl_no}" if atype else appl_no
    return appl_no_raw, atype, appl_no, appl_no_stripped


# --- per-table builders ---------------------------------------------------------


def _build_products(frame: pl.DataFrame, source_file: str) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    norm = _rename_to_canonical(frame, _PRODUCT_FIELD_ALIASES)
    rows: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for index, rec in enumerate(norm.iter_rows(named=True), start=2):  # row 1 is the header
        appl_no_raw, appl_type, appl_no, appl_no_stripped = _normalize_appl_fields(
            _field(rec, "appl_no_raw"), _field(rec, "appl_type"), _field(rec, "appl_no")
        )
        if not appl_no:
            warnings.append({"table": "products", "source_file": source_file, "message": f"row {index}: missing application number"})
        product_no = _field(rec, "product_no").strip()
        product_ndc = _field(rec, "product_ndc").strip()
        rows.append(
            {
                "source_record_id": _product_record_id(appl_type, appl_no_stripped, product_no, product_ndc, index),
                "source_file": source_file,
                "appl_no_raw": appl_no_raw,
                "appl_type": appl_type,
                "appl_no": appl_no,
                "appl_no_stripped": appl_no_stripped,
                "product_no": product_no,
                "drug_name": _field(rec, "drug_name").strip(),
                "active_ingredient": _field(rec, "active_ingredient").strip(),
                "form": _field(rec, "form").strip(),
                "route": _field(rec, "route").strip(),
                "strength": _field(rec, "strength").strip(),
                "reference_drug": _field(rec, "reference_drug").strip(),
                "reference_standard": _field(rec, "reference_standard").strip(),
                "product_ndc": product_ndc,
                "marketing_status_name": _field(rec, "marketing_status_name").strip(),
            }
        )
    return _frame_of(rows, PRODUCTS_COLUMNS), warnings


def _build_applications(frame: pl.DataFrame, source_file: str) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    norm = _rename_to_canonical(frame, _APPLICATION_FIELD_ALIASES)
    rows: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for index, rec in enumerate(norm.iter_rows(named=True), start=2):
        appl_no_raw, appl_type, appl_no, appl_no_stripped = _normalize_appl_fields(
            _field(rec, "appl_no_raw"), _field(rec, "appl_type"), _field(rec, "appl_no")
        )
        if not appl_no:
            warnings.append({"table": "applications", "source_file": source_file, "message": f"row {index}: missing application number"})
        rows.append(
            {
                "source_record_id": _record_id("application", appl_type, appl_no_stripped, index),
                "source_file": source_file,
                "appl_no_raw": appl_no_raw,
                "appl_type": appl_type,
                "appl_no": appl_no,
                "appl_no_stripped": appl_no_stripped,
                "sponsor_name": _field(rec, "sponsor_name").strip(),
                "common_or_original_name": _field(rec, "common_or_original_name").strip(),
                "submission_classification": _field(rec, "submission_classification").strip(),
                "orphan_status": _field(rec, "orphan_status").strip(),
            }
        )
    return _frame_of(rows, APPLICATIONS_COLUMNS), warnings


def _build_submissions(frame: pl.DataFrame, source_file: str, appl_type_map: dict[str, str]) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    norm = _rename_to_canonical(frame, _SUBMISSION_FIELD_ALIASES)
    rows: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for index, rec in enumerate(norm.iter_rows(named=True), start=2):
        appl_no_raw_src = _field(rec, "appl_no_raw")
        appl_type_src = _field(rec, "appl_type")
        appl_no_src = _field(rec, "appl_no")
        appl_no_raw, appl_type, appl_no, appl_no_stripped = _normalize_appl_fields(appl_no_raw_src, appl_type_src, appl_no_src)
        if not appl_no:
            warnings.append({"table": "submissions", "source_file": source_file, "message": f"row {index}: missing application number"})
        # Submissions.txt carries no ApplType in the real schema: inherit from products/applications.
        if not appl_type and appl_no_stripped:
            appl_type = appl_type_map.get(appl_no_stripped, "")
            if appl_type:
                appl_no_raw = f"{appl_type}{appl_no}" if appl_no else appl_no_raw
        rows.append(
            {
                "source_record_id": _record_id("submission", appl_type, appl_no_stripped, index, _field(rec, "submission_no").strip()),
                "source_file": source_file,
                "appl_no_raw": appl_no_raw,
                "appl_type": appl_type,
                "appl_no": appl_no,
                "appl_no_stripped": appl_no_stripped,
                "submission_type": _field(rec, "submission_type").strip(),
                "submission_no": _field(rec, "submission_no").strip(),
                "submission_status": _field(rec, "submission_status").strip(),
                "submission_status_date": _field(rec, "submission_status_date").strip(),
                "submission_notes": _field(rec, "submission_notes").strip(),
            }
        )
    return _frame_of(rows, SUBMISSIONS_COLUMNS), warnings


def _build_lookups(products_frame: pl.DataFrame) -> pl.DataFrame:
    """Build name/ingredient/ndc/marketing-status -> appl_no lookup rows (deduplicated)."""
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, str]] = []
    for rec in products_frame.select(LOOKUPS_SOURCE_COLUMNS).iter_rows(named=True):
        appl_no = _field(rec, "appl_no")
        appl_no_stripped = _field(rec, "appl_no_stripped")
        appl_type = _field(rec, "appl_type")
        if not appl_no_stripped:
            continue
        candidates: list[tuple[str, str]] = []
        drug_name = _field(rec, "drug_name")
        if drug_name:
            candidates.append(("proprietary_name", drug_name))
        active_ingredient = _field(rec, "active_ingredient")
        if active_ingredient:
            candidates.append(("nonproprietary_name", active_ingredient))
            for part in active_ingredient.split(_INGREDIENT_SEPARATOR):
                ingredient = part.strip()
                if ingredient:
                    candidates.append(("ingredient", ingredient))
        product_ndc = _field(rec, "product_ndc")
        if product_ndc:
            candidates.append(("product_ndc", product_ndc))
        marketing_status = _field(rec, "marketing_status_name")
        if marketing_status:
            candidates.append(("marketing_status", marketing_status))
        for lookup_type, term in candidates:
            key = (lookup_type, term.casefold(), appl_no_stripped)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"lookup_type": lookup_type, "term": term, "appl_no": appl_no, "appl_no_stripped": appl_no_stripped, "appl_type": appl_type})
    return _frame_of(rows, LOOKUPS_COLUMNS)


LOOKUPS_SOURCE_COLUMNS = ["drug_name", "active_ingredient", "product_ndc", "marketing_status_name", "appl_no", "appl_no_stripped", "appl_type"]


def _fill_appl_type_map(appl_type_map: dict[str, str], frame: pl.DataFrame) -> None:
    for rec in frame.select(["appl_no_stripped", "appl_type"]).iter_rows(named=True):
        appl_no_stripped = _field(rec, "appl_no_stripped")
        appl_type = _field(rec, "appl_type")
        if appl_no_stripped and appl_type:
            appl_type_map.setdefault(appl_no_stripped, appl_type)


# --- record ids + small helpers -------------------------------------------------


def _product_record_id(appl_type: str, appl_no_stripped: str, product_no: str, product_ndc: str, row_index: int) -> str:
    if appl_no_stripped:
        return f"drugsfda:product:{appl_type}{appl_no_stripped}:{product_no or 'NA'}"
    if product_ndc:
        return f"drugsfda:product:ndc:{product_ndc}"
    return f"drugsfda:product:row:{row_index}"


def _record_id(kind: str, appl_type: str, appl_no_stripped: str, row_index: int, suffix: str = "") -> str:
    if appl_no_stripped:
        base = f"drugsfda:{kind}:{appl_type}{appl_no_stripped}"
        return f"{base}:{suffix}" if suffix else base
    return f"drugsfda:{kind}:row:{row_index}"


def _frame_of(rows: list[dict[str, str]], columns: list[str]) -> pl.DataFrame:
    schema = dict.fromkeys(columns, pl.Utf8)
    return pl.DataFrame(rows, schema=schema)


def _input_blake3(matched: tuple[pl.DataFrame, str, str] | None, inputs: list[ArtifactRef]) -> list[str]:
    if matched is None:
        return [ref.blake3 for ref in inputs]
    return [matched[2]]


def _tsv_row_count(path: Path) -> int:
    # Header + N data rows; subtract the header. Counts lines (TSV cells never contain
    # raw newlines in Drugs@FDA sources because fields are tab-delimited and unquoted).
    with path.open("rb") as handle:
        count = sum(1 for line in handle if line.strip())
    return max(count - 1, 0)


def _write_warnings_jsonl(path: Path, warnings: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for warning in warnings:
            handle.write(json.dumps(warning, sort_keys=True))
            handle.write("\n")
