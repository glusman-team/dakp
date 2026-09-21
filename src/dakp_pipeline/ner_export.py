"""NER training-data export: the ``dakp.ner.export.v1`` bundle.

Hands DAKP's DailyMed / FAERS / EMA corpora to the GLiNER2 training stack (RelMedNER,
github.com/SkyeAv/RelMedNER) as a self-describing, versioned, deterministic bundle — a
single directory with exactly four files:

* ``manifest.json`` — schema ``dakp.ner.export.v1``: schema strings, ``generated_at``,
  per-file blake3 hashes + counts, per-task and per-source-family counts, and the blake3
  ids of the consumed interim tables.
* ``examples.avro`` — the primary artifact: one Avro record per ``TrainingExample``
  (RelMedNER's exact record schema, mirrored field-for-field under the same
  ``relmedner.ingests`` namespace and pinned by ``tests/eval/ner_training_schema.json``).
* ``examples.ndjson`` — the gliner2 JSONL projection of the same rows
  (``TrainingExample.to_output()``): newline-delimited JSON objects, one per line, in the
  same deterministic order as the Avro records.
* ``ner_gold.json`` — byte-identical copy of the committed NER gold benchmark
  (``tests/eval/ner_gold.json``), the eval artifact for the trained model.

It runs the production composite NER backend
(:class:`~dakp_pipeline.ner.ner.DiseaseNER`, GLiNER2 inference, ``offline=False``) over
every candidate row on the GPU pool. In the DAG it is scheduled AFTER the shape stage so
the ``dakp-nercache`` store warmed by assertion mining turns most extraction calls into
cache hits (the backend + config fingerprint key the cache, so the export's re-extraction
of the same section texts is free). Tests inject a deterministic backend or a fake via
``ctx.params["ner"]`` — the established shaper seam.

Each row becomes one ``TrainingExample`` (see :func:`build_example`):

* ``entities`` — every mined channel, grouped by canonical ``Mention.type``: the object
  channel (``Disease`` / ``PhenotypicFeature``) plus the full qualifier channel
  (``QUALIFIER_TYPES``).
* ``relations`` — two kinds on one list. Normal assertion relations named by
  ``contexts.context_predicate`` (``contraindicated_in`` / ``treats`` /
  ``applied_to_treat``; head = the row's subject drug surface, tail = each object mention;
  ``evidence="asserted"``); qualifier relations built with
  ``contexts.attach_qualifiers_with_scores`` (head = the qualified object mention, tail =
  the qualifier mention; ``evidence="mined"``). A relation ships only when both of its
  surfaces occur in the row text — gliner2's in-text invariant.
* ``classifications`` — one "indication context classification" example per row derived
  from ``contexts.assertion_context`` + ``PREVENTION_CUE``.

Candidate derivation (the corpus families): DailyMed contraindication sections (LOINC
``34070-3``), DailyMed boxed-warning / warnings-and-precautions sections (LOINCs
``34066-1``, ``43685-7``, ``34071-1``, ``42232-9``), DailyMed indications-and-usage
sections (LOINC ``34067-9``), EMA centrally-authorised ``therapeutic_indication`` free
text (``ema_registry.parquet`` — the same NER-mined rows the shaper publishes as
``infores:epar``), and FAERS observed-use ``indication`` strings from the global
``cases.parquet``. Zero-span rows are kept on purpose: deliberate no-entity examples
train abstention.

Subjects: FAERS rows carry their ``drugname``; DailyMed rows carry the document's active
ingredient ONLY when the document is a singleton-ingredient set (the legacy
``selectActiveIngredientSingletons.pl`` discipline — a combination product's label applies
to the mixture); EMA rows carry the ``active_substance`` when it splits to exactly one
value (semicolon fan-out with the ``inn`` fallback).

Determinism: rows dedupe on ``(task, normalized text)`` (contract R5) and sort
deterministically, so identical inputs produce byte-identical bundle files regardless of
input row order. A missing interim table is a loud error naming the file; a
missing/corrupt gold benchmark fails before ANY bundle file is written; a table that
yields zero exportable rows is legal and produces a valid bundle with 0 counts;
re-running over unchanged inputs returns the previously registered refs without touching
the bundle, unless ``ctx.params["force"]``.

See ``plans/ner-export-gliner2.md`` for the design record.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from dataclasses_avroschema.pydantic import AvroBaseModel
from fastavro import writer as _avro_writer
from pydantic import ConfigDict, Field

from dakp_pipeline.assertions import object_mentions
from dakp_pipeline.assertions.contexts import (
    ASSERTION_CONTEXTS,
    assertion_context,
    attach_qualifiers_with_scores,
    context_predicate,
    patient_clause_contexts,
)
from dakp_pipeline.assertions.contraindications import _sentence_spans
from dakp_pipeline.assertions.evidence import CONTRAINDICATION_LOINC, INDICATION_LOINC, dailymed_document_url, faers_record_url, find_faers_cases
from dakp_pipeline.assertions.ner_dispatch import default_ner, mine_by_position
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.ner.dictionary import QUALIFIER_TYPES, canonical_type
from dakp_pipeline.ner.lexical import Mention
from dakp_pipeline.ner.mention_cache import MentionCache
from dakp_pipeline.ner.ner import DiseaseNER
from dakp_pipeline.paths import Workdir

# ======================================================================================
# Avro model mirror of RelMedNER's relmedner.models TrainingExample tree.
#
# Field names, order, types, and defaults are copied VERBATIM from
# RelMedNER origin/main `src/relmedner/models.py` (branch gliner-biomed-post-training,
# merged 2026-09) under the same Meta namespace, so `avro_schema_to_python()` here is
# name-and-shape compatible with RelMedNER's own records — the exported `examples.avro`
# reads back as `TrainingExample` there. Compatibility is pinned by the committed schema
# snapshot `tests/eval/ner_training_schema.json`; regenerate that snapshot on any
# RelMedNER model churn, never hand-edit it.
#
# Deliberately NOT mirrored: RunConfig / Task / Dataset / YamlIngests (RelMedNER pipeline
# plumbing), WorkerNode / Cluster / FlinkJob (cluster deployment) — DAKP exports records,
# not a pipeline.
# ======================================================================================

class StrictBase(AvroBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    class Meta:
        namespace: str = "relmedner.ingests"

class Entity(StrictBase):
    label: str = Field(...)
    mentions: list[str] = Field(...)
    description: str | None = Field(None)

class Description(StrictBase):
    key: str = Field(...)
    description: str = Field(...)

class ChoiceField(StrictBase):
    value: str = Field(...)
    choices: list[str] = Field(...)

    def to_output(self) -> dict[str, Any]:
        return {"value": self.value, "choices": self.choices}

class StructureField(StrictBase):
    name: str = Field(...)
    value: str | list[str] | ChoiceField = Field(...)
    description: str | None = Field(None)

    def to_value(self) -> Any:
        return self.value.to_output() if isinstance(self.value, ChoiceField) else self.value

class Structure(StrictBase):
    name: str = Field(...)
    fields: list[StructureField] = Field(...)

class Classification(StrictBase):
    task: str = Field(...)
    labels: list[str] = Field(...)
    true_label: list[str] = Field(..., min_length=1)
    multi_label: bool = Field(False)
    prompt: str | None = Field(None)
    label_descriptions: list[Description] | None = Field(None)

class RelationField(StrictBase):
    name: str = Field(...)
    value: str = Field(...)

class Relation(StrictBase):
    name: str = Field(...)
    fields: list[RelationField] = Field(...)
    negated: bool = Field(False)
    """biolink Association.negated: True asserts the relation is false; the export never
    asserts negations, so every emitted relation carries False"""
    evidence: str = Field("asserted")
    """how the triple was observed: asserted (the row's drug->object assertion relation
    named by context_predicate) or mined (qualifier attachment)"""

def describe(descriptions: list[Description] | None) -> dict[str, str]:
    return {entry.key: entry.description for entry in descriptions or []}

class TrainingExample(StrictBase):
    text: str = Field(...)
    entities: list[Entity] = Field(default_factory=list)
    classifications: list[Classification] = Field(default_factory=list)
    structures: list[Structure] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    def populated(self) -> frozenset[str]:
        return frozenset(shape for shape in ("entities", "classifications", "structures", "relations") if getattr(self, shape))

    def entities_out(self) -> dict[str, Any]:
        described: list[Description] = [
            Description(key=entity.label, description=entity.description) for entity in self.entities if entity.description
        ]
        output: dict[str, Any] = {"entities": {entity.label: entity.mentions for entity in self.entities}}
        return output | ({"entity_descriptions": describe(described)} if described else {})

    def classifications_out(self) -> dict[str, Any]:
        return {
            "classifications": [
                {"task": task.task, "labels": task.labels, "true_label": task.true_label}
                | ({"multi_label": True} if task.multi_label else {})
                | ({"prompt": task.prompt} if task.prompt else {})
                | ({"label_descriptions": describe(task.label_descriptions)} if task.label_descriptions else {})
                for task in self.classifications
            ]
        }

    def structures_out(self) -> dict[str, Any]:
        described: dict[str, dict[str, str]] = {
            structure.name: describe([Description(key=field.name, description=field.description) for field in structure.fields if field.description])
            for structure in self.structures
            if any(field.description for field in structure.fields)
        }
        output: dict[str, Any] = {
            "json_structures": [{structure.name: {field.name: field.to_value() for field in structure.fields}} for structure in self.structures]
        }
        return output | ({"json_descriptions": described} if described else {})

    def relations_out(self) -> dict[str, Any]:
        # negated/evidence deliberately stay OUT of the gliner2 projection: gliner2's
        # Relation(name, **fields) swallows extra keys into _fields (dropping head/tail on
        # round-trip) and InputExample.validate() requires every relation value to occur in
        # the text. Provenance rides in the avro records instead.
        return {"relations": [{relation.name: {field.name: field.value for field in relation.fields}} for relation in self.relations]}

    def to_output(self) -> dict[str, Any]:
        populated: frozenset[str] = self.populated()
        output: dict[str, Any] = {}
        for shape in ("entities", "classifications", "structures", "relations"):
            if shape in populated:
                output |= getattr(self, f"{shape}_out")()
        return {"input": self.text, "output": output}

# ======================================================================================
# Bundle constants
# ======================================================================================

#: Bundle schema version; bump on any layout/field change.
SCHEMA_VERSION = "dakp.ner.export.v1"
#: The gliner2 row contract the examples honor (RelMedNER's ``TrainingExample`` tree).
EXAMPLE_SCHEMA = "relmedner.training_example.v1"
#: The ``schema_version`` the NER gold benchmark must carry to be exported.
GOLD_SCHEMA_VERSION = "dakp.ner.gold.v1"
#: Bundle directory name under ``Workdir.store``.
OUT_DIRNAME = "ner-export"

MANIFEST_FILENAME = "manifest.json"
EXAMPLES_AVRO_FILENAME = "examples.avro"
EXAMPLES_NDJSON_FILENAME = "examples.ndjson"
GOLD_FILENAME = "ner_gold.json"

_JSON_MEDIA_TYPE = "application/json"
_AVRO_MEDIA_TYPE = "application/avro"
_NDJSON_MEDIA_TYPE = "application/x-ndjson"

_OPERATION = "export_ner"
_DAILYMED_TABLE = "spl_documents.parquet"
_FAERS_CASES_TABLE = "cases.parquet"
_EMA_TABLE = "ema_registry.parquet"
#: The only ``spl_documents.parquet`` columns the export reads. The table carries one row per
#: SPL *section* of every type, so the unprojected read materializes ~1.1 GB of ``section_text``
#: for sections this export discards. ``active_ingredient_name`` feeds the singleton-subject map.
_DAILYMED_COLUMNS: tuple[str, ...] = ("spl_document_id", "loinc_code", "section_text", "active_ingredient_name")
#: Column projection for the case-table read: the indication string, its subject drug, the
#: primaryid provenance, and the quarter for the FDA source URL.
_FAERS_CASES_COLUMNS: tuple[str, ...] = ("quarter", "primaryid", "drugname", "indication")
#: Column projection for the EMA registry read.
_EMA_COLUMNS: tuple[str, ...] = ("therapeutic_indication", "active_substance", "inn", "ema_product_number", "medicine_url")
_FAERS_QUARTER_PARTITION = re.compile(r"^quarter=(?:\d{2}|\d{4})Q[1-4]$", re.IGNORECASE)

#: DailyMed boxed-warning / warnings-and-precautions sections (the shaper's Pass 3): mined as
#: contraindication-family rows, since their prohibitions are contraindication statements.
_WARNING_LOINCS: tuple[str, ...] = ("34066-1", "43685-7", "34071-1", "42232-9")

_TASKS: tuple[str, ...] = ("contraindication", "indication")
_FAMILIES: tuple[str, ...] = ("dailymed", "ema", "faers")
_CLASSIFICATION_TASK = "indication context classification"
_CONTEXT_LABELS: tuple[str, ...] = tuple(ASSERTION_CONTEXTS)
_BIOLINK_PREFIX = "biolink:"

def _is_faers_partition_ref(ref: ArtifactRef) -> bool:
    """Return whether a cases table is under an extractor quarter partition."""
    return bool(_FAERS_QUARTER_PARTITION.fullmatch(ref.uri.parent.name))

#: The training task for a DailyMed section LOINC: contraindications (``34070-3``) and the
#: boxed-warning/warnings sections (Pass 3) both carry contraindication-family supervision;
#: indications (``34067-9``) carry indication supervision. Other LOINCs are not exported.
_TASK_BY_LOINC: dict[str, str] = {
    CONTRAINDICATION_LOINC: "contraindication",
    INDICATION_LOINC: "indication",
    **dict.fromkeys(_WARNING_LOINCS, "contraindication"),
}

#: Qualifier attachment field (``contexts.attach_qualifiers_with_scores`` output) ->
#: gliner2 relation name. The temporal interval collapses into ``temporal_context_text``
#: exactly as it does in the assertion tables.
_RELATION_BY_FIELD: dict[str, str] = {
    "anatomical_context_text": "anatomical_context_qualifier",
    "sex_text": "sex_qualifier",
    "population_context_text": "population_context_qualifier",
    "species_context_text": "species_context_qualifier",
    "frequency_text": "frequency_qualifier",
    "temporal_context_text": "temporal_context_qualifier",
}

# --- inputs -------------------------------------------------------------------------

def gold_path() -> Path:
    """The committed NER gold benchmark, resolved from the repo root.

    Same repo-root convention as ``cli._REPO_ROOT`` (``Path(__file__).resolve().parents[2]``).
    Raises ``FileNotFoundError`` when the file is absent — the exporter never runs without
    its eval gold (no silent fallback).
    """
    path = Path(__file__).resolve().parents[2] / "tests" / "eval" / GOLD_FILENAME
    if not path.exists():
        msg = f"ner_export: NER gold benchmark is missing: {path}"
        raise FileNotFoundError(msg)
    return path

def read_dailymed_sections(path: Path) -> pl.DataFrame | None:
    """Read only the export-LOINC sections of ``spl_documents.parquet``, projected to 4 columns.

    Equivalent to reading the whole table and letting :func:`select_dailymed_rows` skip the
    other LOINCs, but the skipping happens in the parquet reader instead of in Python, so the
    ~95% of section text that is about to be discarded is never decoded. Falls back to a full
    read when the table lacks the expected columns, so an older schema still exports.
    """
    try:
        scan = pl.scan_parquet(path)
        available = set(scan.collect_schema().names())
        if not set(_DAILYMED_COLUMNS) <= available:
            return scan.collect(engine="streaming")
        return (
            scan.select(_DAILYMED_COLUMNS)
            .filter(pl.col("loinc_code").cast(pl.Utf8).fill_null("").str.strip_chars().is_in(list(_TASK_BY_LOINC)))
            .collect(engine="streaming")
        )
    except Exception as exc:
        logger.warning("export_ner: skipping unreadable input {} ({})", path, exc)
        return None

def read_ema_registry(path: Path) -> pl.DataFrame | None:
    """Read the EMA registry's indication columns, projected to 5 columns.

    The interim table is small (one row per centrally-authorised human medicine), so the
    projection is about column hygiene rather than bytes. Falls back to a full read on an
    older schema; returns None when the file is unreadable (the caller raises a loud error).
    """
    try:
        scan = pl.scan_parquet(path)
        available = set(scan.collect_schema().names())
        if not set(_EMA_COLUMNS) <= available:
            return scan.collect(engine="streaming")
        return scan.select(_EMA_COLUMNS).collect(engine="streaming")
    except Exception as exc:
        logger.warning("export_ner: skipping unreadable input {} ({})", path, exc)
        return None

def _normalized(text: str) -> str:
    """Dedupe normalization: lowercase + whitespace collapse."""
    return " ".join(text.split()).lower()

def _singleton_subjects(table: pl.DataFrame) -> dict[str, str]:
    """One subject ingredient per SPL document that carries EXACTLY one distinct non-blank one.

    The legacy ``selectActiveIngredientSingletons.pl`` discipline: a combination product's
    label applies to the mixture, so its sections get no subject (no drug->disease training
    relation) rather than over-attributing the statement to one component.
    """
    if "active_ingredient_name" not in table.columns or "spl_document_id" not in table.columns:
        return {}
    subjects: dict[str, str] = {}
    for document_id, ingredient in (
        table.select("spl_document_id", "active_ingredient_name")
        .with_columns(
            pl.col("spl_document_id").cast(pl.Utf8).fill_null("").str.strip_chars(),
            pl.col("active_ingredient_name").cast(pl.Utf8).fill_null("").str.strip_chars(),
        )
        .filter(pl.col("active_ingredient_name") != "")
        .unique()
        .sort(["spl_document_id", "active_ingredient_name"])
        .iter_rows()
    ):
        if not document_id:
            continue
        known = subjects.setdefault(document_id, ingredient)
        if known != ingredient:
            subjects[document_id] = ""
    return {document_id: ingredient for document_id, ingredient in subjects.items() if ingredient}

def select_dailymed_rows(table: pl.DataFrame) -> list[dict[str, str]]:
    """DailyMed candidate rows: contraindication (34070-3), boxed warnings (34066-1, 43685-7,
    34071-1, 42232-9), and indications-and-usage (34067-9) sections.

    Every ``spl_documents.parquet`` row with an export LOINC and non-blank ``section_text``
    becomes one row; other LOINC sections are skipped. Text is exported verbatim except for
    leading/trailing strip; ``source_uri`` is the DailyMed label URL carrying the LOINC
    fragment (``dailymed_document_url`` form); ``subject`` is the singleton active ingredient.
    """
    subjects = _singleton_subjects(table)
    rows: list[dict[str, str]] = []
    for rec in table.iter_rows(named=True):
        loinc = str(rec.get("loinc_code") or "").strip()
        task = _TASK_BY_LOINC.get(loinc)
        if task is None:
            continue
        text = str(rec.get("section_text") or "").strip()
        if not text:
            continue
        document_id = str(rec.get("spl_document_id") or "").strip()
        rows.append(
            {
                "text": text,
                "task": task,
                "source_family": "dailymed",
                "source_document_id": document_id,
                "section": loinc,
                "source_uri": dailymed_document_url(document_id),
                "subject": subjects.get(document_id, ""),
            }
        )
    return rows

def _split_semicolons(cell: str) -> list[str]:
    """Split a semicolon-joined EMA cell into its stripped, non-empty values."""
    return [part.strip() for part in cell.split(";") if part.strip()]

def select_ema_rows(table: pl.DataFrame) -> list[dict[str, str]]:
    """EMA candidate rows: one ``indication`` row per registry row with indication text.

    Mirrors the shaper's EPAR fan-in (``approved_treats._ema_indication_items``): the free-text
    ``therapeutic_indication`` is the row text; the subject is the ``active_substance`` cell
    (``inn`` fallback) ONLY when it splits to exactly one value. ``source_uri`` is the EPAR
    medicine URL when the registry carries one.
    """
    rows: list[dict[str, str]] = []
    for rec in table.iter_rows(named=True):
        text = str(rec.get("therapeutic_indication") or "").strip()
        if not text:
            continue
        substances = _split_semicolons(str(rec.get("active_substance") or "").strip() or str(rec.get("inn") or "").strip())
        product_number = str(rec.get("ema_product_number") or "").strip()
        medicine_url = str(rec.get("medicine_url") or "").strip()
        rows.append(
            {
                "text": text,
                "task": "indication",
                "source_family": "ema",
                "source_document_id": product_number or medicine_url,
                "section": "therapeutic_indication",
                "source_uri": medicine_url,
                "subject": substances[0] if len(substances) == 1 else "",
            }
        )
    return rows

def reduce_faers_frame(table: pl.DataFrame) -> pl.DataFrame:
    """Collapse duplicate FAERS indication strings before any of them becomes a Python dict.

    The production case table is 15-25 M rows and its ``indication`` column has only a few
    thousand distinct values, so building one dict per row and letting :func:`dedupe_sort`
    discard 98% of them costs seconds and gigabytes for nothing. This does the collapse in
    polars and hands :func:`select_faers_rows` only the rows that can still win.

    Winner selection matches :func:`_sort_key` exactly. Rows are grouped on the *verbatim*
    stripped text -- a strictly finer key than ``_sort_key``'s normalized text, so this can
    only under-collapse, and the global :func:`dedupe_sort` still runs afterwards. Within a
    group the kept row minimizes ``(source_record_id, quarter)``; ``faers_record_url`` embeds
    a zero-padded year and quarter digit, so minimal quarter label is minimal ``source_uri``.
    """
    if "indication" not in table.columns:
        return table
    has_record = "primaryid" in table.columns
    ordering = ["_text", *(["_record"] if has_record else []), *(["quarter"] if "quarter" in table.columns else [])]
    return (
        table.lazy()
        .with_columns(
            _text=pl.col("indication").cast(pl.Utf8).fill_null("").str.strip_chars(),
            **({"_record": pl.col("primaryid").cast(pl.Utf8).fill_null("").str.strip_chars()} if has_record else {}),
        )
        .filter(pl.col("_text") != "")
        .sort(ordering, nulls_last=True)
        .unique(subset=["_text"], keep="first", maintain_order=True)
        .drop("_text", "_record", strict=False)
        .collect(engine="streaming")
    )

def _quarter_urls(table: pl.DataFrame) -> dict[str, str]:
    """One validated FDA quarter URL per distinct quarter label in ``table``.

    ``faers_record_url`` runs a regex, an ``int`` parse and an f-string per call; over 20 M
    rows that is ~20 M recomputations of one of ~80 strings. Building the map from the distinct
    labels keeps the loud ``ValueError`` on a malformed quarter -- it just raises once instead
    of on the first offending row.
    """
    if "quarter" not in table.columns:
        return {}
    labels = table.get_column("quarter").unique().to_list()
    return {("" if label is None else str(label).strip().upper()): faers_record_url(label) for label in labels}

def _nonblank_indications(table: pl.DataFrame) -> pl.DataFrame:
    """The rows :func:`select_faers_rows` will actually emit, so quarter validation matches it.

    A malformed quarter on a row whose indication is blank is skipped today and must stay
    skipped: validating every distinct quarter in the raw table would turn it into an error.
    """
    if "indication" not in table.columns:
        return table
    return table.filter(pl.col("indication").cast(pl.Utf8).fill_null("").str.strip_chars() != "")

def select_faers_rows(table: pl.DataFrame) -> list[dict[str, str]]:
    """FAERS candidate rows: one indication row per case row with non-blank ``indication``.

    Duplicate indication strings collapse in :func:`reduce_faers_frame` (when the caller
    pre-reduces) and again in :func:`dedupe_sort`; ``source_uri`` is the FDA FAERS quarter URL
    (``faers_record_url`` form), resolved through a per-quarter map rather than per row. The
    subject is the case's ``drugname`` verbatim.
    """
    quarter_urls = _quarter_urls(_nonblank_indications(table))
    rows: list[dict[str, str]] = []
    for rec in table.iter_rows(named=True):
        text = str(rec.get("indication") or "").strip()
        if not text:
            continue
        record_id = str(rec.get("primaryid") or "").strip()
        rows.append(
            {
                "text": text,
                "task": "indication",
                "source_family": "faers",
                "source_document_id": "",
                "section": "indication",
                "source_record_id": record_id,
                "source_uri": faers_record_url(rec.get("quarter"), quarter_urls),
                "subject": str(rec.get("drugname") or "").strip(),
            }
        )
    return rows

# --- dedupe + determinism -------------------------------------------------------------

def _sort_key(row: Mapping[str, str]) -> tuple[str, ...]:
    """Deterministic row order (export contract R5).

    The first four fields ARE the R5 order ``(task, normalized text, source_document_id or
    "", source_record_id or "")``; verbatim text, URI, and subject are tie-breakers so even
    degenerate duplicate sets order identically for any input row order (they cannot reorder
    surviving rows, which all differ on ``(task, normalized text)`` after dedupe).
    """
    text = row.get("text", "")
    return (
        row.get("task", ""),
        _normalized(text),
        row.get("source_document_id") or "",
        row.get("source_record_id") or "",
        text,
        row.get("source_uri") or "",
        row.get("subject") or "",
    )

def dedupe_sort(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Dedupe on ``(task, normalized text)`` and sort deterministically (export contract R5).

    The retained row is the first in :func:`_sort_key` order (its verbatim ``text``,
    provenance, and subject survive); output lines follow the same sorted order, so identical
    input sets yield byte-identical ``examples.ndjson`` / ``examples.avro`` regardless of
    input row order.
    """
    winners: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=_sort_key):
        key = (row.get("task", ""), _normalized(row.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        winners.append(dict(row))
    return winners

# --- gold validation -----------------------------------------------------------------

def _load_gold(path: Path) -> dict[str, Any]:
    """Parse and validate a gold benchmark; loud errors, never a silent fallback.

    ``FileNotFoundError`` when absent; ``ValueError`` when the content is not readable JSON
    or its ``schema_version`` is not ``dakp.ner.gold.v1``.
    """
    if not path.exists():
        msg = f"ner_export: NER gold benchmark is missing: {path}"
        raise FileNotFoundError(msg)
    try:
        gold = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"ner_export: NER gold benchmark is not readable JSON ({path}): {exc}"
        raise ValueError(msg) from exc
    if not isinstance(gold, dict):
        msg = f"ner_export: NER gold benchmark {path} must be a JSON object"
        raise ValueError(msg)
    schema_version = gold.get("schema_version")
    if schema_version != GOLD_SCHEMA_VERSION:
        msg = f"ner_export: NER gold benchmark {path} has schema_version {schema_version!r}, expected {GOLD_SCHEMA_VERSION!r}"
        raise ValueError(msg)
    annotation_policy = gold.get("annotation_policy")
    if not isinstance(annotation_policy, str) or not annotation_policy.strip():
        msg = f"ner_export: NER gold benchmark {path} is missing required annotation_policy"
        raise ValueError(msg)
    cases = gold.get("cases")
    if not isinstance(cases, list) or not cases:
        msg = f"ner_export: NER gold benchmark {path} must contain a non-empty cases list"
        raise ValueError(msg)
    return gold

# --- mining + row build ----------------------------------------------------------------

def mine_rows(rows: Sequence[Mapping[str, str]], ner: DiseaseNER, cache: MentionCache | None) -> list[list[Mention]]:
    """Extract mixed-channel mentions per row, cache-backed (one mention list per row, in order).

    Work items are positional tuples so identical texts across families still dedupe to one
    cache entry / one mining call, while every row receives its own mention list (the
    per-section keying discipline that keeps one document's sections from swapping mentions).
    """
    items = [(f"row-{index:08d}", "text", str(row.get("text") or "")) for index, row in enumerate(rows)]

    def _mine(work_items: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        return {(item[0], item[1]): ner.extract(item[2]) for item in work_items}

    return mine_by_position(items, ner, _mine, cache)

def _localize(text: str, mentions: Sequence[Mention]) -> tuple[Callable[[Mention], str | None], list[Mention]]:
    """Rewrite ``mentions`` to sentence-relative offsets and map each to its sentence text.

    Returns the ``sentence_of`` callable that :func:`attach_qualifiers_with_scores` and
    :func:`patient_clause_contexts` consume, plus the rewritten mentions (the export does
    not mutate its inputs). A mention covered by no sentence span (tokenization gaps) is
    excluded from the relation build but keeps its entity slot.
    """
    spans = _sentence_spans(text)
    localized: list[Mention] = []
    sentence_by_mention: dict[int, str] = {}
    for mention in mentions:
        for span in spans:
            if mention.start < span.end and span.start < mention.end:
                local = Mention(
                    text=span.text[mention.start - span.start : mention.end - span.start],
                    start=mention.start - span.start,
                    end=mention.end - span.start,
                    type=mention.type,
                    score=mention.score,
                )
                localized.append(local)
                sentence_by_mention[id(local)] = span.text
                break
    return (lambda mention: sentence_by_mention.get(id(mention))), localized

def _entity_groups(mentions: Sequence[Mention]) -> list[Entity]:
    """Group mined mentions (both channels) by canonical type, first-occurrence ordered."""
    mentions_by_label: dict[str, list[str]] = {}
    for mention in mentions:
        label = canonical_type(mention.type)
        bucket = mentions_by_label.setdefault(label, [])
        if mention.text not in bucket:
            bucket.append(mention.text)
    return [Entity(label=label, mentions=values, description=None) for label, values in mentions_by_label.items() if values]

def _qualifier_relations(objects: Sequence[Mention], qualifiers: Sequence[Mention], sentence_of: Callable[[Mention], str | None]) -> list[Relation]:
    """Qualifier mentions -> relations (head = qualified object, tail = qualifier mention).

    Attachment reuses ``contexts.attach_qualifiers_with_scores`` verbatim — the score floor,
    one-host-per-sentence rule, patient-clause template host, and the
    strictly-inside-or-disjoint overlap rule — so training rows and assertion tables attach
    qualifiers identically.
    """
    attached, _scores = attach_qualifiers_with_scores(objects, qualifiers, sentence_of)
    relations: list[Relation] = []
    for host_index in sorted(attached):
        host = objects[host_index]
        for field in sorted(attached[host_index]):
            value = attached[host_index][field]
            relations.append(
                Relation(
                    name=_RELATION_BY_FIELD.get(field, field),
                    fields=[RelationField(name="head", value=host.text), RelationField(name="tail", value=value)],
                    negated=False,
                    evidence="mined",
                )
            )
    return relations

def _disease_context_relations(objects: Sequence[Mention], sentence_of: Callable[[Mention], str | None]) -> tuple[list[Relation], set[int]]:
    """Build patient-clause disease-context relations and return context-only object indexes.

    Shared ``patient_clause_contexts`` owns explicit ``for treatment of A in patients with B``
    detection. The emitted relation is ``B --disease_context_qualifier--> A``: B is the
    patient disease qualified by context A. Context-only indexes leave ordinary drug->object
    relations, matching the assertion shaper; both spans remain in ``entities``.
    """
    clause = patient_clause_contexts(objects, sentence_of)
    relations = [
        Relation(
            name="disease_context_qualifier",
            fields=[RelationField(name="head", value=objects[index].text), RelationField(name="tail", value=context)],
            negated=False,
            evidence="mined",
        )
        for index, context in sorted(clause.contexts.items())
    ]
    return relations, {*clause.context_only, *clause.ambiguous}

def _normal_relations(row: Mapping[str, str], objects: Sequence[Mention]) -> list[Relation]:
    """The row's ordinary assertion relation(s), subject drug -> each object mention.

    Named by ``contexts.context_predicate`` over ``contexts.assertion_context`` (the exact
    predicate the assertion tables emit for the same context). Ships only when the row HAS a
    subject AND the subject surface occurs in the text — gliner2 validates every relation
    value as an in-text mention, so a FAERS indication string that never names its drug
    simply omits the normal relation and keeps its entities + qualifiers.
    """
    subject = str(row.get("subject") or "").strip()
    if not subject or subject.lower() not in str(row.get("text") or "").lower():
        return []
    context = assertion_context(str(row.get("source_family") or ""), str(row.get("section") or ""), str(row.get("text") or ""))
    name = context_predicate(context, str(row.get("source_family") or "")).removeprefix(_BIOLINK_PREFIX)
    relations: list[Relation] = []
    seen: set[str] = set()
    for mention in objects:
        if mention.text in seen:
            continue
        seen.add(mention.text)
        relations.append(
            Relation(
                name=name,
                fields=[RelationField(name="head", value=subject), RelationField(name="tail", value=mention.text)],
                negated=False,
                evidence="asserted",
            )
        )
    return relations

def build_example(row: Mapping[str, str], mentions: Sequence[Mention]) -> TrainingExample:
    """Turn one deduped candidate row + its mined mentions into one gliner2 ``TrainingExample``.

    Entities carry BOTH channels (object + the full ``QUALIFIER_TYPES`` qualifier channel);
    relations carry the normal assertion relation(s) and the qualifier relations; one
    context classification rides along. Zero-span rows are legal and kept (deliberate
    no-entity examples train abstention).
    """
    objects = object_mentions(list(mentions))
    qualifiers = [mention for mention in mentions if canonical_type(mention.type) in QUALIFIER_TYPES]
    # ONE localization pass over objects + qualifiers so a single ``sentence_of`` map serves
    # the attachment call (two maps keyed by object identity would leave the qualifiers
    # unmapped and every attachment would be withheld as outside-sentence).
    sentence_of, localized = _localize(str(row.get("text") or ""), [*objects, *qualifiers])
    localized_objects = localized[: len(objects)]
    localized_qualifiers = localized[len(objects) :]
    disease_relations, _context_only = _disease_context_relations(localized_objects, sentence_of)
    context_surfaces = {relation.fields[0].value for relation in disease_relations}
    normal_objects = [mention for mention in objects if mention.text not in context_surfaces]
    relations = [
        *_normal_relations(row, normal_objects),
        *disease_relations,
        *_qualifier_relations(localized_objects, localized_qualifiers, sentence_of),
    ]
    context = assertion_context(str(row.get("source_family") or ""), str(row.get("section") or ""), str(row.get("text") or ""))
    return TrainingExample(
        text=str(row.get("text") or ""),
        entities=_entity_groups(mentions),
        classifications=[
            Classification(
                task=_CLASSIFICATION_TASK,
                labels=list(_CONTEXT_LABELS),
                true_label=[context],
                multi_label=False,
                prompt=None,
                label_descriptions=None,
            )
        ],
        relations=relations,
    )

def build_examples(rows: Sequence[Mapping[str, str]], ner: DiseaseNER, cache: MentionCache | None) -> list[TrainingExample]:
    """Mine every deduped row and build its ``TrainingExample`` (one call per row's text)."""
    return [build_example(row, mentions) for row, mentions in zip(rows, mine_rows(rows, ner, cache), strict=True)]

# --- bundle assembly -------------------------------------------------------------------

def build_manifest(
    avro_path: Path,
    ndjson_path: Path,
    gold_path: Path,
    input_refs: Iterable[ArtifactRef],
    *,
    rows: Sequence[Mapping[str, str]] | None = None,
    gold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble ``manifest.json`` from the payload files as written.

    Blake3 hashes are computed over the bundle's own files, so self-consistency holds by
    construction; ``generated_at`` is the only non-deterministic field. Every
    ``task_counts`` and ``family_counts`` key is always present. ``rows`` and ``gold`` let
    the caller pass the payloads it just wrote (the files are serialized straight from
    them); omit them and the NDJSON is read back and re-counted, which is what an external
    caller wants.
    """
    rows = list(rows) if rows is not None else [json.loads(line) for line in ndjson_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = gold if gold is not None else _load_gold(gold_path)
    gold_cases = gold.get("cases")
    task_tally = Counter(str(row.get("task")) for row in rows)
    family_tally = Counter(str(row.get("source_family")) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "example_schema": EXAMPLE_SCHEMA,
        "benchmark_schema": GOLD_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "files": {
            EXAMPLES_AVRO_FILENAME: {"blake3": hash_file(avro_path), "examples": len(rows)},
            EXAMPLES_NDJSON_FILENAME: {"blake3": hash_file(ndjson_path), "rows": len(rows)},
            GOLD_FILENAME: {"blake3": hash_file(gold_path), "cases": len(gold_cases) if isinstance(gold_cases, list) else 0},
        },
        "task_counts": {task: task_tally.get(task, 0) for task in _TASKS},
        "family_counts": {family: family_tally.get(family, 0) for family in _FAMILIES},
        "inputs": sorted({ref.blake3 for ref in input_refs}),
    }

def write_bundle(
    out_dir: Path,
    candidate_rows: Iterable[Mapping[str, str]],
    gold_src: Path,
    input_refs: Iterable[ArtifactRef],
    *,
    ner: DiseaseNER,
    cache: MentionCache | None = None,
) -> dict[str, Path]:
    """Write the four-file bundle into ``out_dir``; returns the paths keyed by filename.

    Validates the gold benchmark BEFORE any file is written, then dedupes + sorts the rows,
    mines them (GLiNER inference via ``ner``, cache-backed), and writes the Avro records,
    their gliner2 NDJSON projection (one JSON object per line), a byte-identical copy of the
    gold, and the manifest. Existing files are overwritten cleanly (idempotent re-run). Zero
    candidate rows is legal and yields an empty ``examples.avro`` / ``examples.ndjson``.
    """
    gold = _load_gold(gold_src)
    rows = dedupe_sort(candidate_rows)
    examples = build_examples(rows, ner, cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        EXAMPLES_AVRO_FILENAME: out_dir / EXAMPLES_AVRO_FILENAME,
        EXAMPLES_NDJSON_FILENAME: out_dir / EXAMPLES_NDJSON_FILENAME,
        GOLD_FILENAME: out_dir / GOLD_FILENAME,
        MANIFEST_FILENAME: out_dir / MANIFEST_FILENAME,
    }
    with paths[EXAMPLES_AVRO_FILENAME].open("wb") as handle:
        _avro_writer(handle, TrainingExample.avro_schema_to_python(), [example.model_dump() for example in examples])
    # Streamed rather than joined: the joined form holds the whole payload in memory twice.
    with paths[EXAMPLES_NDJSON_FILENAME].open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(example.to_output(), ensure_ascii=False) + "\n" for example in examples)
    shutil.copyfile(gold_src, paths[GOLD_FILENAME])
    manifest = build_manifest(paths[EXAMPLES_AVRO_FILENAME], paths[EXAMPLES_NDJSON_FILENAME], paths[GOLD_FILENAME], list(input_refs), rows=rows, gold=gold)
    paths[MANIFEST_FILENAME].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths

# --- stage entry point ------------------------------------------------------------------

def export(inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Export the GLiNER2 training-data bundle (Transformer-shaped entry point).

    Locates ``spl_documents.parquet``, the FAERS global ``cases.parquet``, and
    ``ema_registry.parquet`` among ``inputs`` — a missing or unusable table is a loud
    ``RuntimeError`` naming the file. Derives the candidate rows, mines them with the
    production composite NER backend (``ctx.params["ner"]``, else the deterministic
    offline gazetteer fallback the tests use), writes the bundle under
    ``Workdir(ctx.workdir).store / "ner-export"``, registers the four files with the
    artifact store (provenance inputs = the consumed tables' blake3 ids), and returns
    ``[manifest, examples.avro, examples.ndjson, gold]`` refs.
    """
    refs = list(inputs)
    with step(logger, _OPERATION):
        dailymed_ref = next((ref for ref in refs if ref.uri.name == _DAILYMED_TABLE), None)
        if dailymed_ref is None:
            msg = f"ner_export: missing DailyMed interim table {_DAILYMED_TABLE} among the input refs"
            raise RuntimeError(msg)
        faers_ref = next((ref for ref in refs if ref.uri.name == _FAERS_CASES_TABLE and not _is_faers_partition_ref(ref)), None)
        if faers_ref is None:
            msg = f"ner_export: missing FAERS case table {_FAERS_CASES_TABLE} among the input refs"
            raise RuntimeError(msg)
        ema_ref = next((ref for ref in refs if ref.uri.name == _EMA_TABLE), None)
        if ema_ref is None:
            msg = f"ner_export: missing EMA registry table {_EMA_TABLE} among the input refs"
            raise RuntimeError(msg)
        store = ArtifactStore(Workdir(ctx.workdir))
        input_ids = sorted({dailymed_ref.blake3, faers_ref.blake3, ema_ref.blake3})
        # Every other expensive stage in this pipeline skips on unchanged inputs; the export
        # already registers usable index entries, it just never consulted them.
        cached = None if ctx.params.get("force") else store.find_by_operation(_OPERATION, input_ids)
        if cached is not None:
            stats(logger, _OPERATION, skipped=True, reason="inputs unchanged", outputs=len(cached))
            return cached
        dailymed_table = read_dailymed_sections(dailymed_ref.uri)
        if dailymed_table is None:
            msg = f"ner_export: unreadable DailyMed interim table {dailymed_ref.uri}"
            raise RuntimeError(msg)
        faers_table = find_faers_cases([faers_ref], columns=_FAERS_CASES_COLUMNS)
        if faers_table is None:
            msg = f"ner_export: unusable FAERS case table {faers_ref.uri} (needs drugname/indication columns)"
            raise RuntimeError(msg)
        ema_table = read_ema_registry(ema_ref.uri)
        if ema_table is None:
            msg = f"ner_export: unreadable EMA registry table {ema_ref.uri}"
            raise RuntimeError(msg)
        faers_table = reduce_faers_frame(faers_table)
        stats(logger, _OPERATION, dailymed_sections=dailymed_table.height, faers_distinct_indications=faers_table.height, ema_rows=ema_table.height)
        candidate_rows = [*select_dailymed_rows(dailymed_table), *select_ema_rows(ema_table), *select_faers_rows(faers_table)]
        out_dir = Workdir(ctx.workdir).store / OUT_DIRNAME
        ner_param = ctx.params.get("ner")
        ner = ner_param if isinstance(ner_param, DiseaseNER) else default_ner(ctx.fixture_root)
        with MentionCache(ctx.workdir) as cache:
            paths = write_bundle(out_dir, candidate_rows, gold_path(), [dailymed_ref, faers_ref, ema_ref], ner=ner, cache=cache)
        rows_written = int(json.loads(paths[MANIFEST_FILENAME].read_text(encoding="utf-8"))["files"][EXAMPLES_NDJSON_FILENAME]["rows"])
        operation = OperationBlock(name=_OPERATION)
        manifest_ref = store.register(paths[MANIFEST_FILENAME], media_type=_JSON_MEDIA_TYPE, inputs=input_ids, operation=operation)
        avro_ref = store.register(paths[EXAMPLES_AVRO_FILENAME], media_type=_AVRO_MEDIA_TYPE, rows=rows_written, inputs=input_ids, operation=operation)
        ndjson_ref = store.register(paths[EXAMPLES_NDJSON_FILENAME], media_type=_NDJSON_MEDIA_TYPE, rows=rows_written, inputs=input_ids, operation=operation)
        gold_ref = store.register(paths[GOLD_FILENAME], media_type=_JSON_MEDIA_TYPE, inputs=input_ids, operation=operation)
        stats(logger, _OPERATION, out_dir=str(out_dir), examples=rows_written, manifest_blake3=manifest_ref.blake3)
    return [manifest_ref, avro_ref, ndjson_ref, gold_ref]

__all__ = [
    "EXAMPLES_AVRO_FILENAME",
    "EXAMPLES_NDJSON_FILENAME",
    "EXAMPLE_SCHEMA",
    "GOLD_FILENAME",
    "GOLD_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "OUT_DIRNAME",
    "SCHEMA_VERSION",
    "ChoiceField",
    "Classification",
    "Description",
    "Entity",
    "Relation",
    "RelationField",
    "StrictBase",
    "Structure",
    "StructureField",
    "TrainingExample",
    "build_example",
    "build_examples",
    "build_manifest",
    "dedupe_sort",
    "describe",
    "export",
    "gold_path",
    "mine_rows",
    "read_dailymed_sections",
    "read_ema_registry",
    "reduce_faers_frame",
    "select_dailymed_rows",
    "select_ema_rows",
    "select_faers_rows",
    "write_bundle",
]
