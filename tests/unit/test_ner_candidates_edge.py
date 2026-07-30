"""Edge-case tests for ``dakp_pipeline.ner.candidates`` (drive to 100% branch coverage).

Targets the uncovered lines: the empty-indication ``continue`` in the FAERS text-record
builder, and the transformer's "input yielded no records" branch (``_records_for`` fallthrough
returning ``[]`` for an unrecognized filename). Plus adversarial empty / no-match resolution.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.candidates import (
    MENTION_CANDIDATES_COLUMNS,
    MentionCandidateTransformer,
    TextRecord,
    resolve_mention_candidates,
    text_records_from_dailymed_sections,
    text_records_from_faers_cases,
)
from dakp_pipeline.ner.dictionary import DictionaryIndex
from dakp_pipeline.ner.lexical import LexicalMatcher
from dakp_pipeline.ner.mapping import MappedTerm, MockFullmapBackend

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_ONTOLOGY_TSV = _FIXTURE_ROOT / "ontology" / "disease_map.tsv"


def _matcher() -> LexicalMatcher:
    return LexicalMatcher(DictionaryIndex.from_tsv(_ONTOLOGY_TSV))


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


# --- text-record builders: empty / blank fields ---------------------------------


def test_faers_text_records_skip_blank_indication_and_fallback_id() -> None:
    frame = pl.DataFrame(
        {
            "indication": ["hypercholesterolemia", "", "   ", "headache"],
            "source_record_id": ["b3:one", "", "b3:three", ""],
            "primaryid": ["p1", "p2", "p3", "p4"],
        }
    )
    records = text_records_from_faers_cases(frame)
    # Blank/whitespace indications are skipped; record id falls back to primaryid when absent.
    assert [(r.text, r.source_record_id) for r in records] == [("hypercholesterolemia", "b3:one"), ("headache", "p4")]
    assert all(r.source_table == "faers_cases" and r.text_field == "indication" for r in records)


def test_dailymed_text_records_skip_blank_clean_text() -> None:
    frame = pl.DataFrame(
        {
            "clean_text": ["asthma", "", "  ", "pain"],
            "source_record_id": ["r1", "r2", "r3", "r4"],
            "section_name": ["indications_and_usage", "x", "y", "indications_and_usage"],
        }
    )
    records = text_records_from_dailymed_sections(frame)
    assert [(r.text, r.section) for r in records] == [("asthma", "indications_and_usage"), ("pain", "indications_and_usage")]


# --- pure resolution: empty / no-match / backend --------------------------------


def test_resolve_mention_candidates_empty_records_yields_empty_frame() -> None:
    frame = resolve_mention_candidates([], _matcher())
    assert frame.columns == MENTION_CANDIDATES_COLUMNS
    assert frame.height == 0


def test_resolve_mention_candidates_no_matches_yields_empty_frame() -> None:
    records = [TextRecord("t", "r1", "section_text", "nothing matches the dictionary here")]
    frame = resolve_mention_candidates(records, _matcher())
    assert frame.height == 0


def test_resolve_mention_candidates_counts_occurrences_of_unique_string() -> None:
    records = [TextRecord("t", "r1", "section_text", "asthma"), TextRecord("t", "r2", "section_text", "asthma and asthma")]
    frame = resolve_mention_candidates(records, _matcher())
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["mention_text"] == "asthma"
    assert "occurrences=3" in row["normalization_notes"]
    assert row["rank"] == "1"


def test_resolve_mention_candidates_with_backend_adds_fullmap_candidate() -> None:
    backend = MockFullmapBackend([MappedTerm("MONDO:0004979", "asthma", "Disease", "NCBITaxon:9606", "fullmap")])
    records = [TextRecord("t", "r1", "section_text", "asthma", section="indications_and_usage")]
    frame = resolve_mention_candidates(records, _matcher(), backend=backend)
    sources = set(frame["candidate_source"].to_list())
    # Both the dictionary candidate (MONDO) and the backend candidate (fullmap) are present.
    assert {"MONDO", "fullmap"} <= sources
    # fullmap ranks first (rank 1) by the deterministic source ranking.
    fullmap_row = frame.filter(pl.col("candidate_source") == "fullmap").row(0, named=True)
    assert fullmap_row["rank"] == "1"
    assert "section:indications_and_usage" in fullmap_row["normalization_notes"]


# --- transformer: unrecognized input filename yields no records -----------------


def test_transformer_skips_input_that_yields_no_records(ctx: TaskContext, tmp_path: Path) -> None:
    # An unrecognized filename (not spl_sections / faers_cases.tsv / faers cases.parquet)
    # dispatches to the [] fallthrough, so it contributes no records and no input id.
    unrelated = tmp_path / "products.parquet"
    pl.DataFrame({"unrelated": ["x"]}).write_parquet(unrelated)

    transformer = MentionCandidateTransformer(_matcher())
    refs = transformer.transform([_ref(unrelated)], ctx)

    assert len(refs) == 1
    out = refs[0]
    assert out.uri.name == "mention_candidates.tsv"
    # No records -> empty output table, and the unrecognized input was NOT registered as an input.
    assert out.rows == 0
    assert out.manifest is not None


def test_transformer_dispatches_spl_sections_and_faers_cases(ctx: TaskContext, tmp_path: Path) -> None:
    sections = tmp_path / "spl_sections.parquet"
    pl.DataFrame({"clean_text": ["asthma"], "source_record_id": ["r1"], "section_name": ["indications_and_usage"]}).write_parquet(sections)

    faers_tsv = tmp_path / "faers_cases.tsv"
    pl.DataFrame({"indication": ["pain"], "source_record_id": ["b3:x"], "primaryid": ["p1"]}).write_csv(faers_tsv, separator="\t")

    transformer = MentionCandidateTransformer(_matcher())
    refs = transformer.transform([_ref(sections), _ref(faers_tsv)], ctx)
    assert len(refs) == 1
    # Both inputs produced records -> both registered as inputs; two unique mention strings.
    assert refs[0].rows == 2
