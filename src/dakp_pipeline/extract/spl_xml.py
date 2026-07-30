"""DailyMed SPL XML extractor — streaming, gzip-aware, namespace-robust.

Parses gzipped (or plain) SPL XML batches with :func:`xml.etree.ElementTree.iterparse`
(streaming, constant-memory per document) and emits the normalized DailyMed interim
tables under ``data/interim/dailymed/``:

* ``spl_documents.parquet``  — the locked public contract
  (:data:`dakp_pipeline.io.schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS`): one row per
  document-section with the document's first active ingredient and first approval
  denormalized onto each section row. Returned **first** so downstream assertion shapers
  that resolve "the dailymed parquet" keep working unchanged.
* ``spl_sets.parquet``       — one row per distinct SPL set id.
* ``spl_approvals.parquet``  — one row per approval (id/code/type) per set.
* ``spl_ingredients.parquet``— active + inactive ingredients, each with UNII + role.
* ``spl_sections.parquet``   — the proper per-section table: LOINC code, title, raw *and*
  cleaned section text, with a stable ``source_record_id``.

Plus an **uncompressed** ``data/tabular/dailymed_spl_sections.tsv`` (the section table in
Tablassert-readable form).

Field semantics are ported from ``ref/legacy/DailyMed/bin/parseXML-xtree.py`` (HL7 v3 SPL) but as
clean, typed Python. The extractor handles two shapes:

* **mock** — the namespace-free simplified fixture (direct ``<setId>`` /
  ``<activeIngredient>`` / ``<section loinc=...>`` children).
* **HL7 v3** — real DailyMed SPL (``urn:hl7-org:v3``): set id from ``<setId root=>``,
  approvals from ``subjectOf/approval`` (NDA ids under OID ``2.16.840.1.113883.3.150``;
  application-type codes under ``2.16.840.1.113883.3.26.1.1``), active/inactive
  ingredients from ``activeMoiety`` / ``inactiveIngredient`` subtrees, and LOINC-coded
  sections nested under ``component/section``.

Both paths share one ``source_record_id`` derivation (BLAKE3 over source hash + document
id + local key) so re-runs are byte-stable and joins across tables are lossless.
"""

from __future__ import annotations

import gzip
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_bytes
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.paths import Workdir
from dakp_pipeline.workers import go_runner

# LOINC section codes DAKP consumes, with stable output names (PLAN.md "Sharded DailyMed
# extraction sketch"). Codes absent here are still extracted; the name just falls back
# to the XML ``name`` attribute or the LOINC code itself.
SECTION_CODE_NAMES = {
    "34067-9": "indications_and_usage",
    "34070-3": "contraindications",
    "34066-1": "boxed_warning",
    "42229-5": "warnings_and_precautions",
}

# Approval OID roots / code systems ported from the legacy parser.
_NDA_OID = "2.16.840.1.113883.3.150"  # subjectOf/approval/id[@root] -> extension (NDA id)
_APPL_CODE_SYSTEM = "2.16.840.1.113883.3.26.1.1"  # subjectOf/approval/code[@codeSystem] -> application type
_HL7V3_NS = "{urn:hl7-org:v3}"

# Column contracts for the new normalized tables (spl_documents stays in io.schemas).
SPL_SETS_COLUMNS: list[str] = ["source_record_id", "spl_set_id", "release_file", "xml_path"]
SPL_APPROVALS_COLUMNS: list[str] = ["source_record_id", "spl_set_id", "approval_id", "approval_code", "approval_type", "release_file", "xml_path"]
SPL_INGREDIENTS_COLUMNS: list[str] = ["source_record_id", "spl_set_id", "ingredient_name", "ingredient_unii", "role", "release_file", "xml_path"]
SPL_SECTIONS_COLUMNS: list[str] = [
    "source_record_id",
    "spl_document_id",
    "spl_set_id",
    "loinc_code",
    "section_name",
    "section_title",
    "raw_text",
    "clean_text",
    "release_file",
    "xml_path",
]

# The five normalized tables the Go ``dailymed`` worker writes as uncompressed TSV (same
# column contracts as the parquet tables above; see go/internal/dailymed).
_GO_TABLES: tuple[str, ...] = ("spl_documents", "spl_sets", "spl_approvals", "spl_ingredients", "spl_sections")


@dataclass
class ApprovalRecord:
    approval_id: str
    code: str
    type: str


@dataclass
class IngredientRecord:
    name: str
    unii: str
    role: str  # "active" | "inactive"


@dataclass
class SectionRecord:
    loinc: str
    name: str
    title: str
    raw_text: str
    clean_text: str


@dataclass
class DocumentRecord:
    """Fully parsed SPL document, before flattening into normalized tables."""

    set_id: str
    approvals: list[ApprovalRecord] = field(default_factory=list)
    ingredients: list[IngredientRecord] = field(default_factory=list)
    sections: list[SectionRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SPLXMLExtractor:
    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        if go_runner.should_use_go(ctx):
            return self._extract_via_go(inputs, ctx)
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        interim_dir = wd.interim / "dailymed"
        interim_dir.mkdir(parents=True, exist_ok=True)
        log = bind(task_id="extract_dailymed_spl")

        doc_rows: list[dict[str, str]] = []
        set_rows: list[dict[str, str]] = []
        approval_rows: list[dict[str, str]] = []
        ingredient_rows: list[dict[str, str]] = []
        section_rows: list[dict[str, str]] = []
        total_warnings = 0
        input_ids: list[str] = []

        for ref in inputs:
            if not _looks_like_spl(ref.uri):
                log.debug("skipping non-SPL artifact", uri=str(ref.uri))
                continue
            input_ids.append(ref.blake3)
            release_file = ref.uri.name
            for record, warning_count in _iter_document_records(ref.uri, source_artifact_id=ref.blake3, release_file=release_file):
                total_warnings += warning_count
                doc_rows.extend(_document_rows(record, release_file))
                set_rows.extend(_set_rows(record, ref.blake3, release_file))
                approval_rows.extend(_approval_rows(record, ref.blake3, release_file))
                ingredient_rows.extend(_ingredient_rows(record, ref.blake3, release_file))
                section_rows.extend(_section_rows(record, ref.blake3, release_file))

        operation = OperationBlock(name="extract_dailymed_spl")
        docs_fp = schemas.schema_fingerprint(schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS)
        sets_fp = schemas.schema_fingerprint(SPL_SETS_COLUMNS)
        approvals_fp = schemas.schema_fingerprint(SPL_APPROVALS_COLUMNS)
        ingredients_fp = schemas.schema_fingerprint(SPL_INGREDIENTS_COLUMNS)
        sections_fp = schemas.schema_fingerprint(SPL_SECTIONS_COLUMNS)

        refs: list[ArtifactRef] = []
        # spl_documents is registered FIRST: downstream shapers resolve "the dailymed
        # parquet" via the first matching ref, and this is the locked public contract.
        refs.append(
            _write_parquet(
                doc_rows,
                schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS,
                interim_dir / "spl_documents.parquet",
                store,
                operation,
                docs_fp,
                total_warnings,
                input_ids,
            )
        )
        refs.append(
            _write_parquet(set_rows, SPL_SETS_COLUMNS, interim_dir / "spl_sets.parquet", store, operation, sets_fp, total_warnings, input_ids)
        )
        refs.append(
            _write_parquet(
                approval_rows, SPL_APPROVALS_COLUMNS, interim_dir / "spl_approvals.parquet", store, operation, approvals_fp, total_warnings, input_ids
            )
        )
        refs.append(
            _write_parquet(
                ingredient_rows,
                SPL_INGREDIENTS_COLUMNS,
                interim_dir / "spl_ingredients.parquet",
                store,
                operation,
                ingredients_fp,
                total_warnings,
                input_ids,
            )
        )
        refs.append(
            _write_parquet(
                section_rows, SPL_SECTIONS_COLUMNS, interim_dir / "spl_sections.parquet", store, operation, sections_fp, total_warnings, input_ids
            )
        )
        # Uncompressed TSV of the section table for Tablassert handoff.
        refs.append(
            _write_tsv(
                section_rows, SPL_SECTIONS_COLUMNS, wd.tabular / "dailymed_spl_sections.tsv", store, operation, sections_fp, total_warnings, input_ids
            )
        )

        log.info(
            "extracted dailyMed SPL",
            documents=len(doc_rows),
            sets=len(set_rows),
            approvals=len(approval_rows),
            ingredients=len(ingredient_rows),
            sections=len(section_rows),
            warnings=total_warnings,
        )
        return refs

    def _extract_via_go(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        """Delegate parsing to the Go ``dailymed`` worker, repackaging its TSVs identically.

        Go does the streaming XML parse and emits the five normalized tables as uncompressed
        TSV (byte-for-byte parity with the Python path; see ``go/internal/dailymed``). We read
        those TSVs back and run them through the SAME parquet/TSV writers + manifest registration
        as the Python path, so the returned refs (names, order, schemas) are unchanged.
        """
        spl_inputs = [ref for ref in inputs if _looks_like_spl(ref.uri)]
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        interim_dir = wd.interim / "dailymed"
        interim_dir.mkdir(parents=True, exist_ok=True)
        log = bind(task_id="extract_dailymed_spl")
        operation = OperationBlock(name="extract_dailymed_spl")
        input_ids = [ref.blake3 for ref in spl_inputs]

        with tempfile.TemporaryDirectory() as scratch:
            stage = Path(scratch)
            in_dir = go_runner.stage_inputs(spl_inputs, stage / "in")
            out_dir = stage / "out"
            out_dir.mkdir()
            result = go_runner.get_runner().run_table("dailymed", in_dir, out_dir)
            frames = {name: go_runner.read_go_tsv(out_dir / f"{name}.tsv") for name in _GO_TABLES}
        warnings = go_runner.go_warnings(result)

        specs = [
            ("spl_documents", schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS, interim_dir / "spl_documents.parquet"),
            ("spl_sets", SPL_SETS_COLUMNS, interim_dir / "spl_sets.parquet"),
            ("spl_approvals", SPL_APPROVALS_COLUMNS, interim_dir / "spl_approvals.parquet"),
            ("spl_ingredients", SPL_INGREDIENTS_COLUMNS, interim_dir / "spl_ingredients.parquet"),
            ("spl_sections", SPL_SECTIONS_COLUMNS, interim_dir / "spl_sections.parquet"),
        ]
        # spl_documents is registered FIRST (the locked public contract), exactly as the Python path.
        refs: list[ArtifactRef] = [
            _write_parquet(go_runner.go_rows(frames[name]), columns, out, store, operation, schemas.schema_fingerprint(columns), warnings, input_ids)
            for name, columns, out in specs
        ]
        refs.append(
            _write_tsv(
                go_runner.go_rows(frames["spl_sections"]),
                SPL_SECTIONS_COLUMNS,
                wd.tabular / "dailymed_spl_sections.tsv",
                store,
                operation,
                schemas.schema_fingerprint(SPL_SECTIONS_COLUMNS),
                warnings,
                input_ids,
            )
        )
        log.info("extracted dailyMed SPL via Go worker", artifact_id=result.artifact_id, warnings=warnings)
        return refs


extract = SPLXMLExtractor().extract


# --- streaming parse -----------------------------------------------------------


def _iter_document_records(path: Path, *, source_artifact_id: str, release_file: str) -> Iterator[tuple[DocumentRecord, int]]:
    """Stream SPL documents from ``path``.

    Yields ``(record, warning_count)`` per document. Uses ``iterparse`` so each document
    element is freed after it is parsed (constant memory per document, not per file).
    """
    is_hl7v3 = _looks_hl7v3(path)
    with _open_spl(path) as stream:
        for _event, elem in ET.iterparse(stream, events=("end",)):
            if _local(elem.tag) != "document":
                continue
            record = _parse_hl7v3_document(elem) if is_hl7v3 else _parse_mock_document(elem)
            yield record, len(record.warnings)
            elem.clear()


def _open_spl(path: Path) -> Any:
    """Open an SPL file as a binary stream, gzip-aware."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _looks_hl7v3(path: Path) -> bool:
    """Peek the file to detect the HL7 v3 namespace (real DailyMed SPL)."""
    with _open_spl(path) as stream:
        # Read enough of the head to find a namespace declaration / namespaced tag.
        head = stream.read(4096)
    return _HL7V3_NS.encode("utf-8") in head or b"urn:hl7-org:v3" in head


# --- mock (namespace-free) document parse --------------------------------------


def _parse_mock_document(elem: ET.Element) -> DocumentRecord:
    """Parse the simplified namespace-free SPL fixture."""
    warnings: list[str] = []
    set_id = (elem.findtext("setId") or "").strip()
    if not set_id:
        warnings.append("document missing setId")

    approvals: list[ApprovalRecord] = []
    for ap in _direct_children(elem, "approval"):
        code = _attr(ap, "code")
        approvals.append(ApprovalRecord(approval_id=code, code=code, type=_attr(ap, "type")))

    ingredients: list[IngredientRecord] = []
    for ing in _direct_children(elem, "activeIngredient"):
        ingredients.append(IngredientRecord(name=_attr(ing, "name"), unii=_unii(_attr(ing, "unii")), role="active"))
    for ing in _direct_children(elem, "inactiveIngredient"):
        ingredients.append(IngredientRecord(name=_attr(ing, "name"), unii=_unii(_attr(ing, "unii")), role="inactive"))

    sections = _collect_sections([s for s in elem.iter() if _local(s.tag) == "section"], warnings)
    return DocumentRecord(set_id=set_id, approvals=approvals, ingredients=ingredients, sections=sections, warnings=warnings)


# --- HL7 v3 (real DailyMed) document parse -------------------------------------
# Ports the xpath semantics of ref/legacy/DailyMed/bin/parseXML-xtree.py onto ElementTree using
# local-name traversal. This path is correct-by-construction but is not exercised by the
# test suite (no real DailyMed XML is shipped with the repo).


def _parse_hl7v3_document(elem: ET.Element) -> DocumentRecord:
    warnings: list[str] = []

    set_id = ""
    for node in _descendants(elem, "setId"):
        root = _attr(node, "root")
        if root:
            set_id = root.lower()
            break
    if not set_id:
        warnings.append("document missing setId root")

    approvals = _hl7v3_approvals(elem)
    ingredients = _hl7v3_ingredients(elem)
    sections = _collect_sections(list(_descendants(elem, "section")), warnings)
    return DocumentRecord(set_id=set_id, approvals=approvals, ingredients=ingredients, sections=sections, warnings=warnings)


def _hl7v3_approvals(elem: ET.Element) -> list[ApprovalRecord]:
    by_id: dict[str, ApprovalRecord] = {}
    type_by_id: dict[str, str] = {}
    for approval in _descendants(elem, "approval"):
        nda_id = ""
        for ident in _descendants(approval, "id"):
            if _attr(ident, "root") == _NDA_OID:
                nda_id = _attr(ident, "extension")
                break
        appl_type = ""
        for code in _descendants(approval, "code"):
            if _attr(code, "codeSystem") == _APPL_CODE_SYSTEM:
                appl_type = _attr(code, "code")
                break
        if nda_id:
            rec = by_id.setdefault(nda_id, ApprovalRecord(approval_id=nda_id, code=nda_id, type=""))
            rec.type = appl_type or rec.type
            type_by_id[nda_id] = appl_type
    # Preserve legacy behavior: emit one row per NDA id; carry the application-type code.
    return [ApprovalRecord(approval_id=nda_id, code=nda_id, type=type_by_id.get(nda_id, "")) for nda_id in by_id]


def _hl7v3_ingredients(elem: ET.Element) -> list[IngredientRecord]:
    out: list[IngredientRecord] = []
    seen: set[tuple[str, str, str]] = set()

    def add(substance: ET.Element, role: str) -> None:
        name = ""
        unii = ""
        for child in substance:
            if _local(child.tag) == "name":
                name = _text(child)
            elif _local(child.tag) == "code":
                unii = _unii(_attr(child, "code"))
        key = (role, unii, name.lower())
        if key in seen or not name:
            return
        seen.add(key)
        out.append(IngredientRecord(name=name, unii=unii, role=role))

    for substance in _descendants(elem, "activeMoiety"):
        # activeMoiety wraps an inner activeMoiety/activeIngredientSubstance element.
        for inner in _descendants(substance, "activeMoiety"):
            add(inner, "active")
        for inner in _descendants(substance, "activeIngredientSubstance"):
            add(inner, "active")
    for substance in _descendants(elem, "inactiveIngredientSubstance"):
        add(substance, "inactive")
    for ing in _descendants(elem, "ingredient"):
        if _attr(ing, "classCode") == "IACT":
            for inner in _descendants(ing, "ingredientSubstance"):
                add(inner, "inactive")
    return out


def _collect_sections(section_elems: list[ET.Element], warnings: list[str]) -> list[SectionRecord]:
    """Build :class:`SectionRecord` list from ``<section>`` elements (shared by both shapes)."""
    sections: list[SectionRecord] = []
    for sec in section_elems:
        loinc = ""
        title = ""
        for code in _descendants(sec, "code"):
            candidate = _attr(code, "code")
            if not candidate:
                continue
            # Fall back to the first code present, but prefer a LOINC-shaped code
            # (digit-dash-digit) if one appears (the mock fixture carries LOINC on the section).
            if not loinc:
                loinc = candidate
            if _looks_loinc(candidate):
                loinc = candidate
                break
        loinc = loinc or _attr(sec, "loinc")  # mock stores LOINC directly on <section>
        name = _attr(sec, "name") or SECTION_CODE_NAMES.get(loinc, loinc)
        title_elem = next(_descendants(sec, "title"), None)
        if title_elem is not None:
            title = _text(title_elem)
        if not title:
            title = name
        raw_text = _all_text(sec)
        clean_text = _collapse_ws(raw_text)
        if not loinc:
            warnings.append("section missing LOINC code")
        sections.append(SectionRecord(loinc=loinc, name=name, title=title, raw_text=raw_text, clean_text=clean_text))
    return sections


# --- table builders ------------------------------------------------------------


def _document_rows(record: DocumentRecord, release_file: str) -> list[dict[str, str]]:
    """Wide 11-column contract rows: one per section, with first active ingredient +
    first approval denormalized (the locked public contract consumed by assertion shapers)."""
    active = next((i for i in record.ingredients if i.role == "active"), None)
    approval = record.approvals[0] if record.approvals else None
    ing_name = active.name if active else ""
    ing_unii = active.unii if active else ""
    ap_code = approval.code if approval else ""
    ap_type = approval.type if approval else ""
    rows: list[dict[str, str]] = []
    for sec in record.sections:
        document_id = f"{record.set_id}#{sec.loinc}" if sec.loinc else record.set_id
        rows.append(
            {
                "spl_document_id": document_id,
                "spl_set_id": record.set_id,
                "xml_path": release_file,
                "release_file": release_file,
                "approval_code": ap_code,
                "approval_type": ap_type,
                "loinc_code": sec.loinc,
                "section_name": sec.name,
                "section_text": sec.clean_text,
                "active_ingredient_name": ing_name,
                "active_ingredient_unii": ing_unii,
            }
        )
    return rows


def _set_rows(record: DocumentRecord, source_artifact_id: str, release_file: str) -> list[dict[str, str]]:
    if not record.set_id:
        return []
    return [
        {
            "source_record_id": _source_record_id(source_artifact_id, "set", record.set_id),
            "spl_set_id": record.set_id,
            "release_file": release_file,
            "xml_path": release_file,
        }
    ]


def _approval_rows(record: DocumentRecord, source_artifact_id: str, release_file: str) -> list[dict[str, str]]:
    return [
        {
            "source_record_id": _source_record_id(source_artifact_id, "approval", record.set_id, ap.approval_id),
            "spl_set_id": record.set_id,
            "approval_id": ap.approval_id,
            "approval_code": ap.code,
            "approval_type": ap.type,
            "release_file": release_file,
            "xml_path": release_file,
        }
        for ap in record.approvals
    ]


def _ingredient_rows(record: DocumentRecord, source_artifact_id: str, release_file: str) -> list[dict[str, str]]:
    return [
        {
            "source_record_id": _source_record_id(source_artifact_id, "ingredient", record.set_id, ing.unii, ing.name),
            "spl_set_id": record.set_id,
            "ingredient_name": ing.name,
            "ingredient_unii": ing.unii,
            "role": ing.role,
            "release_file": release_file,
            "xml_path": release_file,
        }
        for ing in record.ingredients
    ]


def _section_rows(record: DocumentRecord, source_artifact_id: str, release_file: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sec in record.sections:
        document_id = f"{record.set_id}#{sec.loinc}" if sec.loinc else record.set_id
        rows.append(
            {
                "source_record_id": _source_record_id(source_artifact_id, "section", record.set_id, sec.loinc),
                "spl_document_id": document_id,
                "spl_set_id": record.set_id,
                "loinc_code": sec.loinc,
                "section_name": sec.name,
                "section_title": sec.title,
                "raw_text": sec.raw_text,
                "clean_text": sec.clean_text,
                "release_file": release_file,
                "xml_path": release_file,
            }
        )
    return rows


def _source_record_id(source_artifact_id: str, kind: str, *local_keys: str) -> str:
    """Stable ``b3:`` id = BLAKE3 over (source hash, kind, source-local keys).

    Deterministic across runs for the same source artifact and document-local key, so
    joins across normalized tables are stable and re-extraction is idempotent.
    """
    parts = [source_artifact_id, kind, *local_keys]
    return hash_bytes("\x1f".join(parts).encode("utf-8"))


# --- low-level XML helpers -----------------------------------------------------


def _local(tag: object) -> str:
    """Return the local name of a (possibly namespaced) element tag."""
    text = str(tag)
    brace = text.rfind("}")
    return text[brace + 1 :] if brace != -1 else text


def _direct_children(elem: ET.Element, name: str) -> Iterator[ET.Element]:
    for child in elem:
        if _local(child.tag) == name:
            yield child


def _descendants(elem: ET.Element, name: str) -> Iterator[ET.Element]:
    for node in elem.iter():
        if _local(node.tag) == name:
            yield node


def _attr(elem: ET.Element, key: str) -> str:
    value = elem.get(key)
    return value.strip() if value else ""


def _text(elem: ET.Element) -> str:
    return _collapse_ws("".join(elem.itertext()))


def _all_text(elem: ET.Element) -> str:
    """Concatenate all descendant text of ``elem`` (preserving original whitespace)."""
    return "".join(elem.itertext()).strip()


def _collapse_ws(text: str) -> str:
    """Collapse runs of whitespace to single spaces (mirrors the legacy ``' '.join(split())``)."""
    return " ".join(text.split())


def _unii(code: str) -> str:
    """Normalize a UNII code to the ``UNII:<code>`` form (empty string if absent)."""
    return f"UNII:{code}" if code else ""


def _looks_loinc(code: str) -> bool:
    """Loose LOINC shape check: ``<digits>-<alphanum>``."""
    dash = code.find("-")
    if dash <= 0:
        return False
    return code[:dash].isdigit() and bool(code[dash + 1 :])


def _looks_like_spl(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".xml.gz", ".xml"))


# --- output --------------------------------------------------------------------


def _to_frame(rows: list[dict[str, str]], columns: list[str]) -> pl.DataFrame:
    schema = dict.fromkeys(columns, pl.Utf8)
    if not rows:
        return pl.DataFrame(schema=schema)
    # Re-key each row to exactly `columns` (extra keys dropped, missing -> "") and coerce to str.
    coerced = [{col: str(row.get(col, "")) for col in columns} for row in rows]
    return pl.DataFrame(coerced, schema=schema)


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
    frame = _to_frame(rows, columns)
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


def _write_tsv(
    rows: list[dict[str, str]],
    columns: list[str],
    out: Path,
    store: ArtifactStore,
    operation: OperationBlock,
    fingerprint: str,
    warnings: int,
    inputs: list[str],
) -> ArtifactRef:
    frame = _to_frame(rows, columns)
    rows_written = schemas.write_tsv(frame, out)
    return store.register(
        out,
        media_type=schemas.TSV_MEDIA_TYPE,
        rows=rows_written,
        schema_fingerprint=fingerprint,
        inputs=inputs,
        operation=operation,
        table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint, warnings=warnings),
    )


__all__ = [
    "SECTION_CODE_NAMES",
    "SPL_APPROVALS_COLUMNS",
    "SPL_INGREDIENTS_COLUMNS",
    "SPL_SECTIONS_COLUMNS",
    "SPL_SETS_COLUMNS",
    "ApprovalRecord",
    "DocumentRecord",
    "SPLXMLExtractor",
    "extract",
]
