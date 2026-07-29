"""MEDI / matrix contraindication extractor.

Parses the MEDI contraindication list (TSV fixture or real ``.xlsx`` release asset) into:

1. an interim parquet table ``data/interim/medi/contraindications.parquet`` preserving
   the original contraindication string, active ingredient, the *provided* normalized
   drug/disease id+label, ``medi_version``, and sheet/row provenance with a stable
   ``source_record_id``; and
2. an **uncompressed** Tablassert-facing source-section TSV
   ``data/tabular/medi/contraindications_sections.tsv``.

Also implements the DailyMed-support scoring concept from the legacy
``matrix/bin/studyContraindications.py`` — a lexical word-overlap score between a
contraindication text and a DailyMed contraindication section text. The scoring is a set
of **pure functions** (text in, scores out), fully decoupled from DailyMed extraction;
the extractor populates ``source_score`` only when DailyMed section texts are supplied via
``ctx.params["dailymed_contraindication_sections"]`` (a ``list[str]`` or ``dict[str,str]``
of section id -> text).

Reading is stdlib-first: ``csv`` for TSV and ``zipfile`` + ``xml.etree`` for xlsx (no
``openpyxl`` dependency). Header names are normalized so the snake_case TSV fixture and the
space-bearing xlsx headers map to the same canonical fields.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import digest_dirname, hash_bytes
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.paths import Workdir

# --- public column contracts ----------------------------------------------------

# Raw-normalized interim table (parquet). Full provenance; faithful to the source row.
MEDI_CONTRAINDICATIONS_COLUMNS = [
    "source_record_id",
    "medi_version",
    "source_sheet",
    "source_row",
    "source_file",
    "active_ingredient",
    "contraindication_text",
    "disease_contraindicated",
    "normalized_drug_id",
    "normalized_drug_label",
    "normalized_disease_id",
    "normalized_disease_label",
    "source_score",
]

# Tablassert-facing source-section projection (uncompressed TSV). Drops source_file
# (file-level provenance lives in the manifest, not the per-row table).
MEDI_CONTRAINDICATION_SECTIONS_COLUMNS = [
    "source_record_id",
    "medi_version",
    "source_sheet",
    "source_row",
    "active_ingredient",
    "contraindication_text",
    "disease_contraindicated",
    "normalized_drug_id",
    "normalized_drug_label",
    "normalized_disease_id",
    "normalized_disease_label",
    "source_score",
]

# Canonical data fields populated from source headers (everything except provenance/score).
_DATA_FIELDS = (
    "active_ingredient",
    "contraindication_text",
    "disease_contraindicated",
    "normalized_drug_id",
    "normalized_drug_label",
    "normalized_disease_id",
    "normalized_disease_label",
)

# Normalized source header -> canonical field. Covers the real xlsx headers
# ("active ingredient", "final normalized drug id", ...) and the snake_case fixture.
_FIELD_MAP: dict[str, str] = {
    "active_ingredient": "active_ingredient",
    "contraindications": "contraindication_text",
    "disease_contraindicated": "disease_contraindicated",
    "final_normalized_drug_id": "normalized_drug_id",
    "final_normalized_drug_label": "normalized_drug_label",
    "final_normalized_disease_id": "normalized_disease_id",
    "final_normalized_disease_label": "normalized_disease_label",
}

_FIXTURE_DEFAULT_VERSION = "MEDI-0.0-mock"


# --- DailyMed-support scoring (pure; legacy studyContraindications concept) -------

_TAG_RE = re.compile(r"<[^>]+>")
_LT_RE = re.compile(r"<")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
# Header normalizer keeps underscores (canonical field names are snake_case).
_HEADER_NONWORD_RE = re.compile(r"[^a-z0-9_]+")


def words_in_text(text: str) -> list[str]:
    """Tokenize ``text`` into lowercased alphanumeric word tokens.

    Mirrors the legacy ``wordsInText``: strip HTML tags, lowercase, turn a stray ``<``
    into ``lt``, drop non-alphanumeric runs, collapse whitespace, split on space.
    """
    if not text:
        return []
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = cleaned.lower()
    cleaned = _LT_RE.sub("lt", cleaned)
    cleaned = _NONWORD_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned.split(" ") if cleaned else []


def support_score(contraindication_text: str, section_text: str) -> float:
    """Lexical word-overlap support score in ``[0.0, 1.0]``.

    Fraction of the contraindication's word tokens (with repeats, as in the legacy
    scorer) that appear anywhere in ``section_text``. Empty contraindication or section
    -> ``0.0`` (never divides by zero).
    """
    ci_words = words_in_text(contraindication_text)
    if not ci_words:
        return 0.0
    section_words = set(words_in_text(section_text))
    if not section_words:
        return 0.0
    hits = sum(1 for word in ci_words if word in section_words)
    return hits / len(ci_words)


def best_support_score(contraindication_text: str, section_texts: Sequence[str]) -> tuple[float, int]:
    """Return ``(best_score, index)`` of the most-supportive section text.

    Ties resolve to the first occurrence (deterministic). Returns ``(0.0, -1)`` when no
    section texts are supplied.
    """
    best_score = 0.0
    best_idx = -1
    for idx, section in enumerate(section_texts):
        score = support_score(contraindication_text, section)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_score, best_idx


def rank_sections(contraindication_text: str, section_texts: Sequence[str]) -> list[tuple[float, int]]:
    """All ``(score, index)`` pairs sorted by score desc, then index asc (deterministic)."""
    scored = [(support_score(contraindication_text, section), idx) for idx, section in enumerate(section_texts)]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored


# --- xlsx reader (stdlib zipfile + ElementTree; no openpyxl) ---------------------

_XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_OFFICE_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


@dataclass(frozen=True)
class SheetRows:
    """One worksheet parsed as a grid of string cells."""

    name: str
    rows: list[list[str]]


def read_xlsx_tables(path: Path) -> list[SheetRows]:
    """Read every worksheet of an ``.xlsx`` as a grid of string rows (stdlib only).

    Shared strings, inline strings, and sparse columns (column letters from the cell
    reference) are resolved. Empty cells in sparse rows are filled with ``""``.
    """
    sheets: list[SheetRows] = []
    with zipfile.ZipFile(path) as archive:
        shared = _read_shared_strings(archive)
        for name, target in _read_sheet_targets(archive):
            rows = _read_worksheet(archive, target, shared)
            sheets.append(SheetRows(name=name, rows=rows))
    return sheets


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in si.iter(f"{_XML_NS}t")) for si in root.findall(f"{_XML_NS}si")]


def _read_sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Ordered ``(sheet_name, worksheet_xml_path)`` from workbook.xml + its rels."""
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {rel.get("Id"): rel.get("Target") for rel in rels.findall(f"{_REL_NS}Relationship")}
    targets: list[tuple[str, str]] = []
    for sheet in workbook.iter(f"{_XML_NS}sheet"):
        name = sheet.get("name") or "Sheet1"
        rid = sheet.get(_OFFICE_REL_NS) or ""
        raw_target = rid_to_target.get(rid) or "worksheets/sheet1.xml"
        target = raw_target if raw_target.startswith("xl/") else f"xl/{raw_target.lstrip('/')}"
        targets.append((name, target))
    return targets


def _read_worksheet(archive: zipfile.ZipFile, target: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(target))
    sheet_data = root.find(f"{_XML_NS}sheetData")
    if sheet_data is None:
        return []
    rows: list[list[str]] = []
    for row_el in sheet_data.findall(f"{_XML_NS}row"):
        cells: list[str] = []
        for cell in row_el.findall(f"{_XML_NS}c"):
            col = _col_index(cell.get("r") or "")
            while len(cells) <= col:
                cells.append("")
            cells[col] = _cell_value(cell, shared)
        if cells:
            rows.append(cells)
    return rows


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.get("t") or ""
    value_el = cell.find(f"{_XML_NS}v")
    inline_el = cell.find(f"{_XML_NS}is")
    if cell_type == "s" and value_el is not None:
        idx = int(value_el.text or "0")
        return shared[idx] if 0 <= idx < len(shared) else ""
    if cell_type == "inlineStr" and inline_el is not None:
        return "".join(node.text or "" for node in inline_el.iter(f"{_XML_NS}t"))
    if value_el is not None:
        return value_el.text or ""
    return ""


def _col_index(ref: str) -> int:
    """Convert the column letters of an A1 reference to a 0-based index (``"AB12" -> 27``)."""
    letters = ""
    for ch in ref:
        if ch.isalpha():
            letters += ch
        else:
            break
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(index - 1, 0)


# --- source row iteration -------------------------------------------------------


def iter_medi_rows(path: Path) -> Iterator[tuple[str, int, dict[str, str]]]:
    """Yield ``(sheet, row_number, {canonical_field: value})`` for each MEDI data row.

    Handles both the TSV fixture and the real ``.xlsx`` asset. Non-MEDI sheets (no
    ``active_ingredient`` + ``contraindications`` columns) are skipped. ``row_number`` is
    1-based and counts the header row as line 1, so data rows start at 2.
    """
    for sheet_name, grid in _read_sheets(path):
        if not grid:
            continue
        col_fields = [_FIELD_MAP.get(_normalize_header(header)) or "" for header in grid[0]]
        if "active_ingredient" not in col_fields or "contraindication_text" not in col_fields:
            continue  # not a MEDI sheet
        for offset, raw_row in enumerate(grid[1:], start=2):
            if not any((cell or "").strip() for cell in raw_row):
                continue  # blank line
            padded = (list(raw_row) + [""] * len(col_fields))[: len(col_fields)]
            record: dict[str, str] = dict.fromkeys(_DATA_FIELDS, "")
            for field, value in zip(col_fields, padded, strict=True):
                if field:
                    record[field] = (value or "").strip()
            yield sheet_name, offset, record


def _read_sheets(path: Path) -> list[tuple[str, list[list[str]]]]:
    if path.suffix.lower() == ".xlsx":
        return [(sheet.name, sheet.rows) for sheet in read_xlsx_tables(path)]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return [("tsv", rows)]


def _normalize_header(header: str) -> str:
    """Lowercase, spaces -> underscores, drop non ``[a-z0-9_]`` (keeps snake_case)."""
    return _HEADER_NONWORD_RE.sub("", header.strip().lower().replace(" ", "_"))


def source_record_id(source_blake3: str, sheet: str, row: int) -> str:
    """Stable per-row id derived from the source content hash + sheet + row position.

    Content-aware (via the source BLAKE3) and position-aware, so the same source bytes
    always yield the same ids across runs while distinct sources never collide.
    """
    raw = f"{source_blake3}\t{sheet}\t{row}".encode()
    return f"medi:{digest_dirname(hash_bytes(raw))[:12]}"


# --- extractor ------------------------------------------------------------------


class MEDIContraindicationExtractor:
    """Parse MEDI contraindication rows into interim parquet + a Tablassert-facing TSV."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        store = ArtifactStore(Workdir(ctx.workdir))
        version = _resolve_version(ctx)
        section_texts = _resolve_section_texts(ctx.params.get("dailymed_contraindication_sections"))

        rows: list[dict[str, str]] = []
        warnings = 0
        input_ids: list[str] = []
        for ref in inputs:
            if not _looks_like_medi(ref):
                continue
            input_ids.append(ref.blake3)
            for sheet, row_number, record in iter_medi_rows(ref.uri):
                if not record["active_ingredient"] or not record["contraindication_text"]:
                    # Lossy row: keep it (no silent data loss) but flag for QA.
                    warnings += 1
                row = dict.fromkeys(MEDI_CONTRAINDICATIONS_COLUMNS, "")
                row.update(record)
                row["source_record_id"] = source_record_id(ref.blake3, sheet, row_number)
                row["medi_version"] = version
                row["source_sheet"] = sheet
                row["source_row"] = str(row_number)
                row["source_file"] = ref.uri.name
                row["source_score"] = _score(record["contraindication_text"], section_texts)
                rows.append(row)

        if not rows:
            return []

        parquet_ref = self._write_parquet(rows, warnings, input_ids, ctx, store)
        tsv_ref = self._write_sections_tsv(rows, warnings, parquet_ref.blake3, ctx, store)
        return [parquet_ref, tsv_ref]

    # -- outputs ---------------------------------------------------------------
    def _write_parquet(self, rows: list[dict[str, str]], warnings: int, input_ids: list[str], ctx: TaskContext, store: ArtifactStore) -> ArtifactRef:
        frame = pl.DataFrame(rows, schema=dict.fromkeys(MEDI_CONTRAINDICATIONS_COLUMNS, pl.Utf8))
        out = Workdir(ctx.workdir).interim / "medi" / "contraindications.parquet"
        rows_written = schemas.write_parquet(frame, out)
        fingerprint = schemas.schema_fingerprint(MEDI_CONTRAINDICATIONS_COLUMNS)
        return store.register(
            out,
            media_type=schemas.PARQUET_MEDIA_TYPE,
            rows=rows_written,
            schema_fingerprint=fingerprint,
            inputs=input_ids,
            operation=OperationBlock(name="extract_medi_contraindications"),
            table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint, warnings=warnings),
        )

    def _write_sections_tsv(self, rows: list[dict[str, str]], warnings: int, parquet_id: str, ctx: TaskContext, store: ArtifactStore) -> ArtifactRef:
        frame = pl.DataFrame(rows, schema=dict.fromkeys(MEDI_CONTRAINDICATIONS_COLUMNS, pl.Utf8)).select(MEDI_CONTRAINDICATION_SECTIONS_COLUMNS)
        out = Workdir(ctx.workdir).tabular / "medi" / "contraindications_sections.tsv"
        rows_written = schemas.write_tsv(frame, out)
        fingerprint = schemas.schema_fingerprint(MEDI_CONTRAINDICATION_SECTIONS_COLUMNS)
        return store.register(
            out,
            media_type=schemas.TSV_MEDIA_TYPE,
            rows=rows_written,
            schema_fingerprint=fingerprint,
            inputs=[parquet_id],
            operation=OperationBlock(name="extract_medi_contraindication_sections"),
            table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint, warnings=warnings),
        )


def _looks_like_medi(ref: ArtifactRef) -> bool:
    name = ref.uri.name.lower()
    if not name.endswith((".tsv", ".csv", ".xlsx")):
        return False
    return "medi" in name or "contraindication" in name


def _resolve_version(ctx: TaskContext) -> str:
    value = ctx.params.get("medi_version")
    if value is not None:
        return str(value)
    return _FIXTURE_DEFAULT_VERSION if ctx.profile == "mock" else "unknown"


def _resolve_section_texts(raw: object) -> list[str]:
    """Accept ``list[str]`` or ``dict[id, text]`` DailyMed section inputs; else no scoring."""
    if isinstance(raw, dict):
        return [str(v) for v in raw.values()]
    if isinstance(raw, list | tuple):
        return [str(v) for v in raw]
    return []


def _score(contraindication_text: str, section_texts: Sequence[str]) -> str:
    if not section_texts:
        return ""
    score, _ = best_support_score(contraindication_text, section_texts)
    return f"{score:.4f}"


extract = MEDIContraindicationExtractor().extract

__all__ = [
    "MEDI_CONTRAINDICATIONS_COLUMNS",
    "MEDI_CONTRAINDICATION_SECTIONS_COLUMNS",
    "MEDIContraindicationExtractor",
    "best_support_score",
    "extract",
    "rank_sections",
    "read_xlsx_tables",
    "source_record_id",
    "support_score",
    "words_in_text",
]
