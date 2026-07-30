"""Unit tests for mention-inventory emission (``mention_candidates.tsv``).

Covers the refactored mention-inventory contract: the PLAN.md column contract (text span +
type only, no ontology CURIE/name/category), unique-normalized-string resolution (occurrences
deduped to one deterministic representative + a count), the DailyMed/FAERS text-record
builders, uncompressed TSV output, and an end-to-end transformer run registering one artifact.
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
    write_mention_candidates,
)
from dakp_pipeline.ner.dictionary import Gazetteer
from dakp_pipeline.ner.lexical import LexicalMatcher
from dakp_pipeline.paths import Workdir


def _matcher() -> LexicalMatcher:
    return LexicalMatcher(Gazetteer({"asthma": "disease", "headache": "phenotype", "hypercholesterolemia": "disease"}))


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _ctx(tmp_path: Path) -> TaskContext:
    return TaskContext(profile="mock", workdir=tmp_path / "work", fixture_root=None, threads=1, memory_budget_gb=1, params={})


# --- column contract & unique-string resolution -------------------------------


def test_resolve_emits_contract_columns_all_utf8() -> None:
    records = [TextRecord("faers_cases", "rec1", "indication", "asthma")]
    frame = resolve_mention_candidates(records, _matcher())
    assert frame.columns == MENTION_CANDIDATES_COLUMNS
    assert set(frame.schema.values()) == {pl.Utf8}


def test_unique_string_resolution_dedupes_occurrences_to_one_row() -> None:
    records = [
        TextRecord("dailymed_spl_sections", "rec1", "section_text", "treatment of asthma in adults", "indications_and_usage"),
        TextRecord("faers_cases", "rec2", "indication", "asthma"),
        TextRecord("faers_cases", "rec3", "indication", "Asthma"),  # same normalized string
    ]
    frame = resolve_mention_candidates(records, _matcher())
    # Three occurrences of one unique normalized mention string -> one inventory row.
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["mention_text"] == "asthma"
    assert row["type"] == "disease"
    assert row["occurrences"] == "3"


def test_representative_occurrence_is_deterministic_first() -> None:
    records = [
        TextRecord("faers_cases", "rec2", "indication", "asthma"),
        TextRecord("dailymed_spl_sections", "rec1", "section_text", "asthma in adults", "indications_and_usage"),
    ]
    frame = resolve_mention_candidates(records, _matcher())
    row = frame.row(0, named=True)
    # "dailymed_spl_sections" sorts before "faers_cases" -> dailymed is the representative,
    # regardless of input order.
    assert row["source_table"] == "dailymed_spl_sections"
    assert row["source_record_id"] == "rec1"
    assert row["text_field"] == "section_text"
    assert row["section"] == "indications_and_usage"


def test_offsets_slice_to_mention_text_and_score_is_formatted() -> None:
    text = "relief of asthma and headache"
    records = [TextRecord("dailymed_spl_sections", "rec1", "section_text", text)]
    frame = resolve_mention_candidates(records, _matcher())
    # Normalized-string sort order: asthma before headache.
    assert frame.get_column("mention_text").to_list() == ["asthma", "headache"]
    assert frame.get_column("type").to_list() == ["disease", "phenotype"]
    for row in frame.iter_rows(named=True):
        start, end = int(row["mention_start"]), int(row["mention_end"])
        assert text[start:end] == row["mention_text"]
        assert row["score"] == "1.000000"


def test_resolve_is_deterministic() -> None:
    records = [
        TextRecord("dailymed_spl_sections", "rec1", "section_text", "headache and asthma"),
        TextRecord("faers_cases", "rec2", "indication", "headache"),
    ]
    first = resolve_mention_candidates(records, _matcher())
    for _ in range(3):
        assert resolve_mention_candidates(records, _matcher()).equals(first)


# --- text-record builders ------------------------------------------------------


def test_text_records_from_dailymed_sections() -> None:
    frame = pl.DataFrame({"source_record_id": ["b3:a"], "section_name": ["indications_and_usage"], "clean_text": ["treatment of asthma"]})
    records = text_records_from_dailymed_sections(frame)
    assert len(records) == 1
    record = records[0]
    assert record.source_table == "dailymed_spl_sections"
    assert record.source_record_id == "b3:a"
    assert record.text_field == "section_text"
    assert record.text == "treatment of asthma"
    assert record.section == "indications_and_usage"


def test_text_records_from_faers_cases_prefers_source_record_id() -> None:
    frame = pl.DataFrame({"source_record_id": ["b3:x"], "primaryid": ["1001"], "indication": ["asthma"]})
    records = text_records_from_faers_cases(frame)
    assert len(records) == 1
    record = records[0]
    assert record.source_table == "faers_cases"
    assert record.source_record_id == "b3:x"
    assert record.text_field == "indication"
    assert record.section == ""


# --- uncompressed TSV output ---------------------------------------------------


def test_write_mention_candidates_is_uncompressed_tsv(tmp_path: Path) -> None:
    records = [TextRecord("faers_cases", "rec1", "indication", "asthma")]
    frame = resolve_mention_candidates(records, _matcher())
    out = tmp_path / "mention_candidates.tsv"
    rows = write_mention_candidates(frame, out)
    assert rows == 1
    raw = out.read_bytes()
    # Uncompressed: plain TSV header, not a gzip magic number.
    assert not raw.startswith(b"\x1f\x8b")
    assert raw.split(b"\n", 1)[0].decode().split("\t") == MENTION_CANDIDATES_COLUMNS


# --- end-to-end transformer registering one artifact ---------------------------


def test_transformer_end_to_end_registers_one_artifact(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    sections = tmp_path / "spl_sections.parquet"
    pl.DataFrame({"clean_text": ["treatment of asthma"], "source_record_id": ["b3:a"], "section_name": ["indications_and_usage"]}).write_parquet(
        sections
    )
    sections_ref = _ref(sections)

    transformer = MentionCandidateTransformer(_matcher())
    (out_ref,) = transformer.transform([sections_ref], ctx)

    out_path = ctx.workdir / "data" / "tabular" / "mention_candidates.tsv"
    assert out_ref.uri == out_path
    assert out_path.exists()
    assert out_ref.media_type == "text/tab-separated-values"
    assert out_ref.rows == 1

    frame = pl.read_csv(out_path, separator="\t")
    assert frame.columns == MENTION_CANDIDATES_COLUMNS
    assert frame.get_column("mention_text").to_list() == ["asthma"]

    assert out_ref.manifest is not None
    manifest = read_manifest(out_ref.manifest)
    assert manifest.artifact_id == out_ref.blake3
    assert manifest.operation is not None
    assert manifest.operation.name == "ner_mention_candidates"
    assert manifest.table.schema_fingerprint is not None
    # The one input artifact that yielded records is recorded.
    assert manifest.inputs == [sections_ref.blake3]
