"""Edge-case tests for ``dakp_pipeline.ner.candidates`` (drive to 100% branch coverage).

Targets every remaining branch: blank-field skips in both text-record builders, the FAERS
``source_record_id`` -> ``primaryid`` fallback (and both empty -> ``""``), all four
``_records_for`` filename-dispatch branches (plus the ``cases.parquet``-without-faers-uri
fallthrough), zero-record resolution yielding an empty full-schema frame, the transformer's
"input yielded no records" branch (not added to ``input_ids``), and multi-record occurrence
counting.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import read_manifest
from dakp_pipeline.ner.candidates import (
    MENTION_CANDIDATES_COLUMNS,
    MentionCandidateTransformer,
    TextRecord,
    resolve_mention_candidates,
    text_records_from_dailymed_sections,
    text_records_from_faers_cases,
)
from dakp_pipeline.ner.dictionary import Gazetteer
from dakp_pipeline.ner.lexical import LexicalMatcher
from dakp_pipeline.paths import Workdir


def _matcher() -> LexicalMatcher:
    return LexicalMatcher(Gazetteer({"asthma": "disease", "headache": "phenotype"}))


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _ctx(tmp_path: Path) -> TaskContext:
    return TaskContext(profile="mock", workdir=tmp_path / "work", fixture_root=None, threads=1, memory_budget_gb=1, params={})


# --- text-record builders: blank-field skips & id fallback ---------------------


def test_dailymed_text_records_skip_blank_clean_text() -> None:
    frame = pl.DataFrame(
        {"clean_text": ["asthma", "", "   "], "source_record_id": ["r1", "r2", "r3"], "section_name": ["indications_and_usage", "x", "y"]}
    )
    records = text_records_from_dailymed_sections(frame)
    # Blank/whitespace-only clean_text rows are skipped.
    assert [(r.text, r.source_record_id) for r in records] == [("asthma", "r1")]


def test_faers_text_records_skip_blank_indication() -> None:
    frame = pl.DataFrame({"indication": ["asthma", "", "   "], "source_record_id": ["b3:one", "b3:two", "b3:three"]})
    records = text_records_from_faers_cases(frame)
    assert [(r.text, r.source_record_id) for r in records] == [("asthma", "b3:one")]


def test_faers_text_records_fall_back_to_primaryid() -> None:
    # The public faers_cases.tsv projection lacks a stable source_record_id.
    frame = pl.DataFrame({"source_record_id": [""], "primaryid": ["1001"], "indication": ["asthma"]})
    records = text_records_from_faers_cases(frame)
    assert records[0].source_record_id == "1001"


def test_faers_text_records_primaryid_empty_yields_empty_id() -> None:
    # Both source_record_id and primaryid empty -> record id is the empty string.
    frame = pl.DataFrame({"source_record_id": [""], "primaryid": [""], "indication": ["asthma"]})
    records = text_records_from_faers_cases(frame)
    assert records[0].source_record_id == ""


# --- _records_for filename dispatch --------------------------------------------


def test_records_for_dispatches_spl_sections() -> None:
    frame = pl.DataFrame({"clean_text": ["asthma"], "source_record_id": ["r1"], "section_name": ["indications_and_usage"]})
    records = MentionCandidateTransformer._records_for("spl_sections.parquet", "/any/spl_sections.parquet", frame)
    assert [r.source_table for r in records] == ["dailymed_spl_sections"]


def test_records_for_dispatches_faers_cases_tsv() -> None:
    frame = pl.DataFrame({"indication": ["asthma"], "source_record_id": ["b3:x"], "primaryid": ["p1"]})
    records = MentionCandidateTransformer._records_for("faers_cases.tsv", "/any/faers_cases.tsv", frame)
    assert [r.source_table for r in records] == ["faers_cases"]


def test_records_for_dispatches_cases_parquet_under_faers_uri() -> None:
    frame = pl.DataFrame({"indication": ["asthma"], "source_record_id": ["b3:x"], "primaryid": ["p1"]})
    records = MentionCandidateTransformer._records_for("cases.parquet", "/work/data/interim/faers/cases.parquet", frame)
    assert [r.source_table for r in records] == ["faers_cases"]


def test_records_for_cases_parquet_without_faers_uri_returns_empty() -> None:
    # A cases.parquet NOT under a faers path is not mis-dispatched to the FAERS builder.
    frame = pl.DataFrame({"indication": ["asthma"], "source_record_id": ["b3:x"], "primaryid": ["p1"]})
    assert MentionCandidateTransformer._records_for("cases.parquet", "/work/data/interim/other/cases.parquet", frame) == []


def test_records_for_unrelated_file_returns_empty() -> None:
    frame = pl.DataFrame({"unrelated": ["x"]})
    assert MentionCandidateTransformer._records_for("other.parquet", "/any/other.parquet", frame) == []


# --- pure resolution: zero records & multi-record counting ---------------------


def test_resolve_zero_records_yields_empty_frame_with_full_schema() -> None:
    frame = resolve_mention_candidates([], _matcher())
    assert frame.columns == MENTION_CANDIDATES_COLUMNS
    assert frame.height == 0
    assert set(frame.schema.values()) == {pl.Utf8}


def test_resolve_counts_occurrences_across_multiple_records() -> None:
    # Same normalized mention across two records, one of which contains it twice -> count 3.
    records = [
        TextRecord("dailymed_spl_sections", "rec1", "section_text", "asthma and asthma", "indications_and_usage"),
        TextRecord("faers_cases", "rec2", "indication", "asthma"),
    ]
    frame = resolve_mention_candidates(records, _matcher())
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["mention_text"] == "asthma"
    assert row["occurrences"] == "3"
    # Representative is the earliest occurrence in the deterministic-first record.
    assert row["source_record_id"] == "rec1"
    assert row["mention_start"] == "0"


# --- transformer: input that yields no records is not registered ---------------


def test_transformer_skips_input_that_yields_no_records(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    unrelated = tmp_path / "other.parquet"
    pl.DataFrame({"unrelated": ["x"]}).write_parquet(unrelated)
    unrelated_ref = _ref(unrelated)

    transformer = MentionCandidateTransformer(_matcher())
    (out_ref,) = transformer.transform([unrelated_ref], ctx)

    # No records -> empty output table, and the unrecognized input was NOT added to input_ids.
    assert out_ref.rows == 0
    assert out_ref.manifest is not None
    manifest = read_manifest(out_ref.manifest)
    assert manifest.inputs == []
