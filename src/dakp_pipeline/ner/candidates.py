"""Mention-candidate emission (PLAN.md ``data/tabular/mention_candidates.tsv``).

Ties mention generation together without coupling it to final canonical mapping:

1. Mine source text records (DailyMed SPL section text + FAERS indication strings) with
   the deterministic :class:`~dakp_pipeline.ner.lexical.LexicalMatcher`.
2. **Resolve unique mention strings, not every occurrence** (PLAN.md "DailyMed/FAERS NER"):
   the candidate set is computed once per distinct normalized mention string; occurrences
   only contribute a deterministic representative + a count.
3. Optionally enrich each unique string via an injected
   :class:`~dakp_pipeline.ner.mapping.MappingBackend` (the mocked fullmap seam) — final
   mapping decisions remain a separate, later concern.
4. Emit the uncompressed, Tablassert-readable ``mention_candidates.tsv`` with all
   candidate evidence ranked deterministically.

The pure :func:`resolve_mention_candidates` is framework-free; :class:`MentionCandidateTransformer`
is the pipeline-ready wrapper (Transformer protocol) that reads interim tables and registers
the output artifact.
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
from dakp_pipeline.ner.dictionary import semantic_group_for
from dakp_pipeline.ner.lexical import LexicalMatcher, Mention
from dakp_pipeline.ner.mapping import MappedTerm, MappingBackend
from dakp_pipeline.paths import Workdir

# PLAN.md "data/tabular/mention_candidates.tsv" column contract (ordered).
MENTION_CANDIDATES_COLUMNS = [
    "source_table",
    "source_record_id",
    "text_field",
    "mention_text",
    "mention_start",
    "mention_end",
    "semantic_group",
    "candidate_curie",
    "candidate_name",
    "candidate_category",
    "candidate_source",
    "score",
    "rank",
    "normalization_notes",
]

# Confidence for a canonical backend (fullmap) resolution candidate.
BACKEND_SCORE = 1.0
_BACKEND_NOTES = "fullmap-resolve"

# Deterministic candidate tie-breaking: canonical fullmap first, then curated ontologies.
_SOURCE_RANK = {"fullmap": 0, "MONDO": 1, "HPO": 2, "BABEL": 3, "NCIT": 4, "DRUGBANK": 5}


@dataclass(frozen=True)
class TextRecord:
    """One mineable text field from a source table.

    ``section`` carries DailyMed section context (e.g. ``indications_and_usage``); it is
    empty for source fields without section structure (FAERS indication strings).
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


@dataclass(frozen=True)
class _Candidate:
    curie: str
    name: str
    category: str
    source: str
    semantic_group: str
    score: float
    notes: str


# --- text-record builders (consume Milestone-3 interim contracts) --------------


def text_records_from_dailymed_sections(frame: pl.DataFrame) -> list[TextRecord]:
    """Build text records from the DailyMed ``spl_sections`` interim table.

    Uses the cleaned section text (``clean_text``) so NER changes never require reparsing
    XML; ``section_name`` is preserved as section context.
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

    Prefers the stable ``source_record_id`` (b3) when present (cases parquet); falls back
    to ``primaryid`` for the public ``faers_cases.tsv`` projection which lacks it.
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


def resolve_mention_candidates(records: Sequence[TextRecord], matcher: LexicalMatcher, *, backend: MappingBackend | None = None) -> pl.DataFrame:
    """Resolve unique mention strings across ``records`` to ranked candidate rows.

    Returns a frame over :data:`MENTION_CANDIDATES_COLUMNS` (all-Utf8). One row per
    (unique normalized mention string, candidate); the representative occurrence is the
    first by deterministic order and ``normalization_notes`` records the occurrence count.
    """
    occurrences: dict[str, list[_Occurrence]] = {}
    for record in records:
        for mention in matcher.match(record.text, section=record.section):
            occurrences.setdefault(mention.normalized, []).append(_Occurrence(record, mention))

    unique_texts = sorted(occurrences)
    backend_map = backend.resolve_many(unique_texts) if backend is not None else {}

    rows: list[dict[str, str]] = []
    for normalized in unique_texts:
        occs = sorted(occurrences[normalized], key=_occurrence_sort_key)
        representative = occs[0]
        candidates = _collect_candidates(occs, backend_map.get(normalized, []))
        for rank, candidate in enumerate(_rank_candidates(candidates), start=1):
            rows.append(_row_for(representative, len(occs), candidate, rank))
    return _to_frame(rows)


def write_mention_candidates(frame: pl.DataFrame, path: Path) -> int:
    """Write the uncompressed mention-candidates TSV (Tablassert-readable). Returns rows."""
    return schemas.write_tsv(frame, path)


# --- candidate collection / ranking -------------------------------------------


def _collect_candidates(occs: list[_Occurrence], mapped: list[MappedTerm]) -> list[_Candidate]:
    """Union dictionary candidates (from every occurrence) with backend resolutions.

    De-duplicated by ``(curie, source)``; the highest-confidence note wins for a shared
    key (a direct ``exact`` match is preferred over a synonym or backend annotation).
    """
    by_key: dict[tuple[str, str], _Candidate] = {}
    for occ in occs:
        entry = occ.mention.entry
        key = (entry.curie, entry.source)
        candidate = _Candidate(entry.curie, entry.name, entry.category, entry.source, entry.semantic_group, occ.mention.score, occ.mention.notes)
        _merge_candidate(by_key, key, candidate)
    for term in mapped:
        key = (term.curie, term.source)
        candidate = _Candidate(term.curie, term.name, term.category, term.source, semantic_group_for(term.category), BACKEND_SCORE, _BACKEND_NOTES)
        _merge_candidate(by_key, key, candidate)
    return list(by_key.values())


def _merge_candidate(by_key: dict[tuple[str, str], _Candidate], key: tuple[str, str], candidate: _Candidate) -> None:
    existing = by_key.get(key)
    if existing is None or candidate.score > existing.score:
        by_key[key] = candidate


def _rank_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    return sorted(candidates, key=lambda c: (-c.score, _SOURCE_RANK.get(c.source, 9), c.curie, c.name))


def _row_for(representative: _Occurrence, occurrence_count: int, candidate: _Candidate, rank: int) -> dict[str, str]:
    mention = representative.mention
    record = representative.record
    notes = [candidate.notes, f"occurrences={occurrence_count}"]
    if record.section:
        notes.append(f"section:{record.section}")
    return {
        "source_table": record.source_table,
        "source_record_id": record.source_record_id,
        "text_field": record.text_field,
        "mention_text": mention.mention_text,
        "mention_start": str(mention.mention_start),
        "mention_end": str(mention.mention_end),
        "semantic_group": candidate.semantic_group,
        "candidate_curie": candidate.curie,
        "candidate_name": candidate.name,
        "candidate_category": candidate.category,
        "candidate_source": candidate.source,
        "score": f"{candidate.score:.6f}",
        "rank": str(rank),
        "normalization_notes": ";".join(notes),
    }


def _occurrence_sort_key(occ: _Occurrence) -> tuple[str, str, int, int, str]:
    return (occ.record.source_table, occ.record.source_record_id, occ.mention.mention_start, occ.mention.mention_end, occ.mention.entry.curie)


def _to_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
    schema = dict.fromkeys(MENTION_CANDIDATES_COLUMNS, pl.Utf8)
    if not rows:
        return pl.DataFrame(schema=schema)
    coerced = [{col: str(row.get(col, "")) for col in MENTION_CANDIDATES_COLUMNS} for row in rows]
    return pl.DataFrame(coerced, schema=schema)


# --- pipeline-ready transformer ------------------------------------------------


class MentionCandidateTransformer:
    """Transformer that reads interim text tables and emits ``mention_candidates.tsv``.

    The dictionary and (optional) mapping backend are dependency-injected so the mention
    generation stays decoupled from canonical mapping and fully monkeypatchable. Inputs are
    dispatched by filename: DailyMed ``spl_sections`` and FAERS ``cases``/``faers_cases``.
    """

    def __init__(self, matcher: LexicalMatcher, *, backend: MappingBackend | None = None) -> None:
        self._matcher = matcher
        self._backend = backend

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

        frame = resolve_mention_candidates(records, self._matcher, backend=self._backend)
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
        if name in {"cases.parquet", "faers_cases.tsv"} or "faers_cases" in name or ("faers" in uri and name == "cases.parquet"):
            return text_records_from_faers_cases(frame)
        return []


__all__ = [
    "BACKEND_SCORE",
    "MENTION_CANDIDATES_COLUMNS",
    "MentionCandidateTransformer",
    "TextRecord",
    "resolve_mention_candidates",
    "text_records_from_dailymed_sections",
    "text_records_from_faers_cases",
    "write_mention_candidates",
]
