"""Shared evidence helpers for the assertion-aggregation stage.

Pure, testable building blocks used by every assertion shaper:

* **NDA join-key normalization** — FAERS stores zero-stripped application numbers
  (``12345``) while DailyMed/Drugs@FDA store the padded FDA form (``012345``); every
  cross-source NDA join goes through :func:`normalize_nda`.
* **Provenance column assembly** — deduplicated, deterministically sorted, pipe-joined
  lists (the Translator list-encoding convention) for ``approval_ids``,
  ``supporting_spl_sets`` and ``supporting_spl_documents``.
* **SPL-support joining** — index DailyMed SPL approvals/ingredients/sections and
  Drugs@FDA application→ingredient lookups so shapers can ask "which SPL sets support
  this approval?" without re-scanning frames.
* **Table resolution + output writing** — find interim parquet tables among
  ``inputs`` and register the uncompressed assertion TSV.

Text-first by design: these helpers surface source text and only populate CURIEs where a
source already provides an id (DailyMed UNII). Canonical CURIE
mapping is a later milestone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.paths import Workdir

# LOINC section codes DAKP consumes for SPL support (mirrors extract.spl_xml.SECTION_CODE_NAMES).
INDICATION_LOINC = "34067-9"  # indications_and_usage
CONTRAINDICATION_LOINC = "34070-3"  # contraindications

_NDA_DIGITS_RE = re.compile(r"[^0-9]+")


# --- NDA join-key normalization -------------------------------------------------


def normalize_nda(value: Any) -> str:
    """Normalize an FDA application number to a stable join key: digits, no leading zeros.

    FAERS reports ``12345`` while DailyMed/Drugs@FDA report ``012345``; both normalize to
    ``"12345"`` so cross-source NDA joins line up. Non-numeric input and empty values
    normalize to ``""`` (never a join key).
    """
    digits = _NDA_DIGITS_RE.sub("", "" if value is None else str(value))
    return digits.lstrip("0")


# --- provenance column assembly -------------------------------------------------


def merge_unique(*value_lists: Iterable[Any]) -> list[str]:
    """Flatten, stringify, drop empties, and return a deterministically sorted unique list."""
    seen: set[str] = set()
    for values in value_lists:
        for value in values:
            text = "" if value is None else str(value).strip()
            if text:
                seen.add(text)
    return sorted(seen)


def sorted_pipe(values: Iterable[Any]) -> str:
    """Deduplicated, sorted, ``|``-joined encoding of ``values`` (Translator list convention)."""
    return "|".join(merge_unique(values))


# --- input table resolution -----------------------------------------------------

_TableIndex = dict[str, list[ArtifactRef]]


def _table_index(inputs: Iterable[ArtifactRef]) -> _TableIndex:
    """Index refs by basename, preserving input order for duplicate names."""
    index: _TableIndex = {}
    for ref in inputs:
        index.setdefault(ref.uri.name, []).append(ref)
    return index


def _read_indexed_table(index: _TableIndex, filename: str) -> pl.DataFrame | None:
    """Read the first readable table matching ``filename`` from a prebuilt input index."""
    for ref in index.get(filename, []):
        try:
            return schemas.read_table(ref.uri)
        except Exception as exc:
            logger.warning("skipping unreadable input {} ({})", ref.uri, exc)
    return None


def find_table(inputs: Iterable[ArtifactRef], filename: str) -> pl.DataFrame | None:
    """Read the first parquet/tsv input whose ``uri.name`` equals ``filename``."""
    return _read_indexed_table(_table_index(inputs), filename)


def find_faers_cases(inputs: Iterable[ArtifactRef], columns: tuple[str, ...] | None = None) -> pl.DataFrame | None:
    """Resolve the FAERS case table: the global ``cases.parquet`` (preferred) or ``faers_cases.tsv``.

    The extractor emits a global ``cases.parquet`` first plus per-quarter ``cases.parquet``
    partitions; the global one (no ``quarter=`` in its path) is the deduped case join the
    shapers want. Falls back to the public ``faers_cases.tsv`` projection. Returns ``None``
    when no FAERS case table is present (in which case the approved-treats shaper degrades to
    its DailyMed fallback candidate path).

    ``columns`` optionally projects the read to a subset of columns — the production case
    table is tens of millions of rows wide, so shapers that need only a few columns
    (observed-uses: drugname/indication/primaryid) skip the rest (a cheap schema peek
    decides which requested columns actually exist, so a primaryid-less table still loads).
    """
    refs = list(inputs)
    global_cases: ArtifactRef | None = None
    any_cases: ArtifactRef | None = None
    tsv_cases: ArtifactRef | None = None
    for ref in refs:
        if ref.uri.name == "cases.parquet":
            any_cases = any_cases or ref
            if "quarter=" not in str(ref.uri):
                global_cases = global_cases or ref
        elif ref.uri.name == "faers_cases.tsv":
            tsv_cases = tsv_cases or ref

    chosen = global_cases or any_cases or tsv_cases
    if chosen is None:
        return None
    frame = _read_case_table(chosen.uri, columns)
    if "drugname" not in frame.columns or "indication" not in frame.columns:
        return None
    return frame


def _read_case_table(path: Path, columns: tuple[str, ...] | None) -> pl.DataFrame:
    """Read the case table, projecting to ``columns`` when given (absent columns are skipped)."""
    if columns is None:
        return schemas.read_table(path)
    available = _schema_columns(path)
    keep = [c for c in columns if c in available]
    if not keep:
        return pl.DataFrame()
    if path.suffix == ".parquet":
        return pl.read_parquet(path, columns=keep)
    return pl.read_csv(path, separator="\t", columns=keep)


def _schema_columns(path: Path) -> set[str]:
    """Column names of a table WITHOUT loading its data (parquet metadata / CSV header scan)."""
    if path.suffix == ".parquet":
        return set(pl.read_parquet_schema(path))
    return set(pl.scan_csv(path, separator="\t").collect_schema().names())


# --- DailyMed SPL support index -------------------------------------------------


@dataclass(frozen=True)
class DailyMedEvidence:
    """Indexed DailyMed SPL evidence for assertion joining.

    All keys are normalized: ``approval_sets``/``approval_display`` are keyed by
    :func:`normalize_nda`; section maps are keyed by ``spl_set_id``.
    """

    approval_sets: dict[str, set[str]] = field(default_factory=dict)  # norm_nda -> {spl_set_id}
    approval_display: dict[str, str] = field(default_factory=dict)  # norm_nda -> display approval id
    set_ingredient: dict[str, tuple[str, str]] = field(default_factory=dict)  # spl_set_id -> first active (name, unii)
    active_ingredients_by_set: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> all active (name, unii)
    indication_docs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> [(doc_id, text)]
    contraindication_docs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> [(doc_id, text)]

    def indication_support(self, norm_nda: str) -> tuple[list[str], list[str]]:
        """Sorted ``(spl_set_ids, spl_document_ids)`` supporting an approval via indication sections.

        A set supports the approval when it bears that approval AND has an indications-and-usage
        section. Empty lists when the approval has no supporting indication section.
        """
        sets = [s for s in self.approval_sets.get(norm_nda, set()) if self.indication_docs.get(s)]
        docs = [doc for s in sets for doc, _text in self.indication_docs[s]]
        return merge_unique(sets), merge_unique(docs)

    def contraindication_sets_for_drug(self, drug_text: str) -> list[str]:
        """Sorted SPL set ids whose active ingredient matches ``drug_text`` and that have a
        contraindication section (the "first-scope" contraindication-support link)."""
        drug = drug_text.strip().lower()
        if not drug:
            return []
        return merge_unique(
            set_id for set_id, (name, _unii) in self.set_ingredient.items() if name.strip().lower() == drug and self.contraindication_docs.get(set_id)
        )


def build_dailymed_evidence(inputs: Iterable[ArtifactRef]) -> DailyMedEvidence:
    """Index DailyMed SPL approvals, active ingredients, and indication/contraindication sections.

    Reads the normalized interim tables (``spl_approvals``/``spl_ingredients``/``spl_sections``)
    emitted by :mod:`dakp_pipeline.extract.spl_xml`. Missing tables yield empty indexes so the
    shapers degrade gracefully rather than failing.
    """
    evidence = DailyMedEvidence()
    index = _table_index(inputs)

    approvals = _read_indexed_table(index, "spl_approvals.parquet")
    if approvals is not None:
        for rec in approvals.iter_rows(named=True):
            norm = normalize_nda(rec.get("approval_id") or rec.get("approval_code"))
            set_id = str(rec.get("spl_set_id") or "").strip()
            if not norm or not set_id:
                continue
            evidence.approval_sets.setdefault(norm, set()).add(set_id)
            evidence.approval_display.setdefault(norm, str(rec.get("approval_id") or rec.get("approval_code") or "").strip())

    ingredients = _read_indexed_table(index, "spl_ingredients.parquet")
    if ingredients is not None:
        seen_ingredients: set[tuple[str, str, str]] = set()
        for rec in ingredients.iter_rows(named=True):
            if str(rec.get("role") or "").strip().lower() != "active":
                continue
            set_id = str(rec.get("spl_set_id") or "").strip()
            name = str(rec.get("ingredient_name") or "").strip()
            unii = str(rec.get("ingredient_unii") or "").strip()
            if not set_id or not name:
                continue
            key = (set_id, name.lower(), unii)
            if key in seen_ingredients:
                continue
            seen_ingredients.add(key)
            evidence.active_ingredients_by_set.setdefault(set_id, []).append((name, unii))
            evidence.set_ingredient.setdefault(set_id, (name, unii))
        for pairs in evidence.active_ingredients_by_set.values():
            pairs.sort()

    sections = _read_indexed_table(index, "spl_sections.parquet")
    if sections is not None:
        for rec in sections.iter_rows(named=True):
            set_id = str(rec.get("spl_set_id") or "").strip()
            doc_id = str(rec.get("spl_document_id") or "").strip()
            text = str(rec.get("clean_text") or rec.get("raw_text") or "").strip()
            loinc = str(rec.get("loinc_code") or "").strip()
            if not set_id:
                continue
            if loinc == INDICATION_LOINC:
                evidence.indication_docs.setdefault(set_id, []).append((doc_id, text))
            elif loinc == CONTRAINDICATION_LOINC:
                evidence.contraindication_docs.setdefault(set_id, []).append((doc_id, text))

    return evidence


# --- Drugs@FDA application -> ingredient index ----------------------------------


def build_drugsfda_ingredient_map(inputs: Iterable[ArtifactRef]) -> dict[str, set[str]]:
    """Map normalized FDA application number -> {active ingredient} from Drugs@FDA products.

    Confirms an NDA is a real FDA application and yields the approved ingredient(s). Keyed by
    :func:`normalize_nda` so it joins against FAERS (stripped) and DailyMed (padded) NDAs alike.
    """
    mapping: dict[str, set[str]] = {}
    products = find_table(inputs, "products.parquet")
    if products is None:
        return mapping
    for rec in products.iter_rows(named=True):
        norm = normalize_nda(rec.get("appl_no_stripped") or rec.get("appl_no"))
        ingredient = str(rec.get("active_ingredient") or "").strip()
        if norm and ingredient:
            mapping.setdefault(norm, set()).add(ingredient)
    return mapping


# --- assertion output registration ----------------------------------------------


def write_assertion_table(
    table: str, rows: list[dict[str, str]], inputs: list[ArtifactRef], ctx: TaskContext, *, operation: str
) -> list[ArtifactRef]:
    """Write ``rows`` as the uncompressed TSV for ``table`` and register it with provenance.

    Columns are fixed by :data:`dakp_pipeline.io.schemas.ASSERTION_TABLES`; row ordering is the
    caller's responsibility (shapers sort deterministically before calling). Returns ``[]`` only
    when the table contract is unknown (cannot happen for the three assertion tables).
    """
    columns = schemas.columns_for(table)
    frame = pl.DataFrame(rows, schema=columns) if rows else pl.DataFrame(schema=columns)
    out = Workdir(ctx.workdir).tabular / f"{table}.tsv"
    rows_written = schemas.write_tsv(frame, out)
    fingerprint = schemas.schema_fingerprint(columns)
    store = ArtifactStore(Workdir(ctx.workdir))
    ref = store.register(
        out,
        media_type=schemas.TSV_MEDIA_TYPE,
        rows=rows_written,
        schema_fingerprint=fingerprint,
        inputs=[ref.blake3 for ref in inputs],
        operation=OperationBlock(name=operation),
        table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint),
    )
    logger.info("wrote assertion table {} ({} rows)", table, rows_written)
    return [ref]


__all__ = [
    "CONTRAINDICATION_LOINC",
    "INDICATION_LOINC",
    "DailyMedEvidence",
    "build_dailymed_evidence",
    "build_drugsfda_ingredient_map",
    "find_faers_cases",
    "find_table",
    "merge_unique",
    "normalize_nda",
    "sorted_pipe",
    "write_assertion_table",
]
