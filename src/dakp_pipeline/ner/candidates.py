"""Mention-inventory emission (PLAN.md ``data/tabular/mention_candidates.tsv``).

Mines source text records (DailyMed SPL section text + FAERS indication strings) with the
deterministic :class:`~dakp_pipeline.ner.lexical.LexicalMatcher` and emits an inventory of the
**unique mention strings** found. A mention is a text span + entity type ONLY — this module
does **NOT** resolve mentions to ontology CURIEs/names/categories (that is Tablassert's job at
``tablassert build-kg``). The candidate set is computed once per distinct normalized mention
string; occurrences only contribute a deterministic representative + a count.

The pure :func:`resolve_mention_candidates` is framework-free; :class:`MentionCandidateTransformer`
is the pipeline-ready wrapper (Transformer protocol) that reads interim tables and registers the
output artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.ner.lexical import LexicalMatcher, Mention
from dakp_pipeline.paths import Workdir

# PLAN.md "data/tabular/mention_candidates.tsv" column contract (ordered). Mentions carry text
# span + type only; no CURIE/name/category columns (Tablassert resolves ontology concepts).
MENTION_CANDIDATES_COLUMNS = [
    "source_table",
    "source_record_id",
    "text_field",
    "mention_text",
    "mention_start",
    "mention_end",
    "type",
    "score",
    "occurrences",
    "section",
]


@dataclass(frozen=True)
class TextRecord:
    """One mineable text field from a source table.

    ``section`` carries DailyMed section context (e.g. ``indications_and_usage``); it is empty
    for source fields without section structure (FAERS indication strings).
    """

    source_table: str
    source_record_id: str
    text_field: str
    text: str
    section: str = ""


@dataclass(frozen=True)
class _Occurrence:
    record: TextRecord
    mention: Mention


# --- text-record builders (consume Milestone-3 interim contracts) --------------


def text_records_from_dailymed_sections(frame: pl.DataFrame) -> list[TextRecord]:
    """Build text records from the DailyMed ``spl_sections`` interim table.

    Uses the cleaned section text (``clean_text``) so NER changes never require reparsing XML;
    ``section_name`` is preserved as section context.
    """
    records: list[TextRecord] = []
    for row in frame.iter_rows(named=True):
        text = str(row.get("clean_text") or "")
        if not text.strip():
            continue
        records.append(
            TextRecord(
                source_table="dailymed_spl_sections",
                source_record_id=str(row.get("source_record_id") or ""),
                text_field="section_text",
                text=text,
                section=str(row.get("section_name") or ""),
            )
        )
    return records


def text_records_from_faers_cases(frame: pl.DataFrame) -> list[TextRecord]:
    """Build text records from FAERS case rows (the ``indication`` field).

    Prefers the stable ``source_record_id`` (b3) when present (cases parquet); falls back to
    ``primaryid`` for the public ``faers_cases.tsv`` projection which lacks it.
    """
    records: list[TextRecord] = []
    for row in frame.iter_rows(named=True):
        text = str(row.get("indication") or "")
        if not text.strip():
            continue
        record_id = str(row.get("source_record_id") or "") or str(row.get("primaryid") or "")
        records.append(TextRecord(source_table="faers_cases", source_record_id=record_id, text_field="indication", text=text))
    return records


# --- pure resolution -----------------------------------------------------------


def resolve_mention_candidates(records: Sequence[TextRecord], matcher: LexicalMatcher) -> pl.DataFrame:
    """Resolve unique mention strings across ``records`` to one inventory row each.

    Returns a frame over :data:`MENTION_CANDIDATES_COLUMNS` (all-Utf8). One row per unique
    normalized mention string; the representative occurrence is the first by deterministic
    order and ``occurrences`` records how many times the string was seen.
    """
    occurrences: dict[str, list[_Occurrence]] = {}
    for record in records:
        for mention in matcher.match(record.text, section=record.section):
            occurrences.setdefault(mention.normalized, []).append(_Occurrence(record, mention))

    rows: list[dict[str, str]] = []
    for normalized in sorted(occurrences):
        occs = sorted(occurrences[normalized], key=_occurrence_sort_key)
        rows.append(_row_for(occs[0], len(occs)))
    return _to_frame(rows)


def write_mention_candidates(frame: pl.DataFrame, path: Path) -> int:
    """Write the uncompressed mention-inventory TSV (Tablassert-readable). Returns rows."""
    return schemas.write_tsv(frame, path)


# --- row building --------------------------------------------------------------


def _row_for(representative: _Occurrence, occurrence_count: int) -> dict[str, str]:
    mention = representative.mention
    record = representative.record
    return {
        "source_table": record.source_table,
        "source_record_id": record.source_record_id,
        "text_field": record.text_field,
        "mention_text": mention.text,
        "mention_start": str(mention.start),
        "mention_end": str(mention.end),
        "type": mention.type,
        "score": f"{mention.score:.6f}",
        "occurrences": str(occurrence_count),
        "section": record.section,
    }


def _occurrence_sort_key(occ: _Occurrence) -> tuple[str, str, int, int, str]:
    return (occ.record.source_table, occ.record.source_record_id, occ.mention.start, occ.mention.end, occ.mention.type)


def _to_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
    schema = dict.fromkeys(MENTION_CANDIDATES_COLUMNS, pl.Utf8)
    if not rows:
        return pl.DataFrame(schema=schema)
    coerced = [{col: str(row.get(col, "")) for col in MENTION_CANDIDATES_COLUMNS} for row in rows]
    return pl.DataFrame(coerced, schema=schema)


# --- pipeline-ready transformer ------------------------------------------------


class MentionCandidateTransformer:
    """Transformer that reads interim text tables and emits ``mention_candidates.tsv``.

    The matcher is dependency-injected so mention generation stays decoupled from any ontology
    mapping and fully monkeypatchable. Inputs are dispatched by filename: DailyMed
    ``spl_sections`` and FAERS ``cases``/``faers_cases``.
    """

    def __init__(self, matcher: LexicalMatcher) -> None:
        self._matcher = matcher

    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        workdir = Workdir(ctx.workdir)
        store = ArtifactStore(workdir)
        log = bind(task_id="ner_mention_candidates")

        records: list[TextRecord] = []
        input_ids: list[str] = []
        for ref in inputs:
            frame = schemas.read_table(ref.uri)
            added = self._records_for(ref.uri.name, str(ref.uri), frame)
            if added:
                records.extend(added)
                input_ids.append(ref.blake3)

        frame = resolve_mention_candidates(records, self._matcher)
        out = workdir.tabular / "mention_candidates.tsv"
        rows_written = write_mention_candidates(frame, out)
        fingerprint = schemas.schema_fingerprint(MENTION_CANDIDATES_COLUMNS)
        ref = store.register(
            out,
            media_type=schemas.TSV_MEDIA_TYPE,
            rows=rows_written,
            schema_fingerprint=fingerprint,
            inputs=input_ids,
            operation=OperationBlock(name="ner_mention_candidates"),
            table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint),
        )
        log.info("emitted mention candidates", rows=rows_written, unique_strings=frame.height, inputs=len(input_ids))
        return [ref]

    @staticmethod
    def _records_for(name: str, uri: str, frame: pl.DataFrame) -> list[TextRecord]:
        if "spl_sections" in name:
            return text_records_from_dailymed_sections(frame)
        # FAERS case rows: the public TSV, or the global/per-quarter cases parquet (which lives
        # under a faers/ interim dir). Requiring the faers path avoids mis-dispatching an
        # unrelated file that happens to be named cases.parquet.
        if name == "faers_cases.tsv" or (name == "cases.parquet" and "faers" in uri):
            return text_records_from_faers_cases(frame)
        return []


__all__ = [
    "MENTION_CANDIDATES_COLUMNS",
    "MentionCandidateTransformer",
    "TextRecord",
    "resolve_mention_candidates",
    "text_records_from_dailymed_sections",
    "text_records_from_faers_cases",
    "write_mention_candidates",
]
