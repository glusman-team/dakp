"""Unit tests for mention-candidate emission (mention_candidates.tsv).

Covers: the PLAN.md column contract, unique-string resolution (occurrences deduped to one
representative + count), deterministic ranking, optional fullmap-backend composition, the
text-record builders for the DailyMed/FAERS interim contracts, uncompressed TSV output,
and an end-to-end transformer run over the real pipeline fixtures.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.extract import faers_ascii, spl_xml
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
from dakp_pipeline.ner.dictionary import DictionaryIndex
from dakp_pipeline.ner.lexical import LexicalMatcher
from dakp_pipeline.ner.mapping import MockFullmapBackend
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, faers

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_ONTOLOGY_TSV = _FIXTURE_ROOT / "ontology" / "disease_map.tsv"


def _matcher() -> LexicalMatcher:
    return LexicalMatcher(DictionaryIndex.from_tsv(_ONTOLOGY_TSV))


def _ctx(tmp_path: Path) -> TaskContext:
    return TaskContext(profile="mock", workdir=(tmp_path / "work"), fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params={})


# --- schema & unique-string resolution ----------------------------------------


def test_resolve_returns_contract_columns() -> None:
    frame = resolve_mention_candidates([], _matcher())
    assert frame.columns == MENTION_CANDIDATES_COLUMNS
    assert frame.height == 0


def test_unique_string_resolution_dedupes_occurrences() -> None:
    records = [
        TextRecord("dailymed_spl_sections", "rec1", "section_text", "treatment of hypercholesterolemia in adults", "indications_and_usage"),
        TextRecord("faers_cases", "rec2", "indication", "hypercholesterolemia"),
        TextRecord("faers_cases", "rec3", "indication", "Hypercholesterolemia"),  # same normalized string
    ]
    frame = resolve_mention_candidates(records, _matcher())
    # Three occurrences of one unique mention string -> one candidate row.
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["mention_text"] == "hypercholesterolemia"
    assert row["candidate_curie"] == "MONDO:0005154"
    assert row["candidate_source"] == "MONDO"
    assert row["semantic_group"] == "disease"
    assert row["rank"] == "1"
    assert "occurrences=3" in row["normalization_notes"]


def test_representative_occurrence_is_deterministic_first() -> None:
    records = [
        TextRecord("faers_cases", "rec2", "indication", "hypercholesterolemia"),
        TextRecord("dailymed_spl_sections", "rec1", "section_text", "hypercholesterolemia in adults", "indications_and_usage"),
    ]
    frame = resolve_mention_candidates(records, _matcher())
    row = frame.row(0, named=True)
    # "dailymed_spl_sections" sorts before "faers_cases" -> dailymed is the representative.
    assert row["source_table"] == "dailymed_spl_sections"
    assert row["source_record_id"] == "rec1"
    assert row["text_field"] == "section_text"
    assert "section:indications_and_usage" in row["normalization_notes"]


def test_offsets_in_emitted_rows_slice_to_mention_text() -> None:
    text = "relief of headache and pain"
    records = [TextRecord("dailymed_spl_sections", "rec1", "section_text", text)]
    frame = resolve_mention_candidates(records, _matcher())
    for row in frame.iter_rows(named=True):
        start, end = int(row["mention_start"]), int(row["mention_end"])
        assert text[start:end] == row["mention_text"]


def test_resolve_is_deterministic() -> None:
    records = [
        TextRecord("dailymed_spl_sections", "rec1", "section_text", "headache and pain and asthma"),
        TextRecord("faers_cases", "rec2", "indication", "headache"),
    ]
    first = resolve_mention_candidates(records, _matcher())
    for _ in range(3):
        assert resolve_mention_candidates(records, _matcher()).equals(first)


# --- backend composition -------------------------------------------------------


def test_backend_adds_fullmap_candidate_and_ranks_first() -> None:
    backend_frame = pl.DataFrame(
        {"text": ["hypercholesterolemia"], "curie": ["MONDO:0005154"], "name": ["hypercholesterolemia"], "category": ["Disease"], "src": ["fullmap"]}
    )
    backend = MockFullmapBackend.from_frame(backend_frame, source_col="src")
    records = [TextRecord("faers_cases", "rec1", "indication", "hypercholesterolemia")]
    frame = resolve_mention_candidates(records, _matcher(), backend=backend)

    # Dictionary (MONDO) + backend (fullmap) -> two candidates for the one mention string.
    assert frame.height == 2
    assert frame.get_column("rank").to_list() == ["1", "2"]
    # Canonical fullmap ranks ahead of the curated ontology candidate.
    assert frame.get_column("candidate_source").to_list() == ["fullmap", "MONDO"]
    assert frame.row(0, named=True)["normalization_notes"].startswith("fullmap-resolve")


def test_backend_sharing_curie_and_source_is_deduped() -> None:
    # Backend resolves to the same (curie, source) as the dictionary -> no duplicate row.
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)  # infers source MONDO/HPO from CURIEs
    records = [TextRecord("faers_cases", "rec1", "indication", "hypercholesterolemia")]
    frame = resolve_mention_candidates(records, _matcher(), backend=backend)
    assert frame.height == 1
    assert frame.get_column("candidate_source").to_list() == ["MONDO"]


# --- text-record builders ------------------------------------------------------


def test_text_records_from_dailymed_sections() -> None:
    frame = pl.DataFrame(
        {"source_record_id": ["b3:a", "b3:b"], "section_name": ["indications_and_usage", "contraindications"], "clean_text": ["headache text", "  "]}
    )
    records = text_records_from_dailymed_sections(frame)
    assert len(records) == 1  # blank section text skipped
    assert records[0].source_table == "dailymed_spl_sections"
    assert records[0].text_field == "section_text"
    assert records[0].section == "indications_and_usage"


def test_text_records_from_faers_cases_prefers_source_record_id() -> None:
    frame = pl.DataFrame({"source_record_id": ["b3:x"], "primaryid": ["1001"], "indication": ["back pain"]})
    records = text_records_from_faers_cases(frame)
    assert records[0].source_record_id == "b3:x"
    assert records[0].text_field == "indication"


def test_text_records_from_faers_cases_falls_back_to_primaryid() -> None:
    # The public faers_cases.tsv projection lacks source_record_id.
    frame = pl.DataFrame({"primaryid": ["1001"], "indication": ["headache"]})
    records = text_records_from_faers_cases(frame)
    assert records[0].source_record_id == "1001"


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


# --- end-to-end transformer over the pipeline fixtures -------------------------


def _extract_fixture_refs(ctx: TaskContext) -> tuple[ArtifactRef, ArtifactRef]:
    Workdir(ctx.workdir).create()
    dailymed_refs = spl_xml.extract(dailymed.fetch(ctx), ctx)
    faers_refs = faers_ascii.extract(faers.fetch(ctx), ctx)
    sections_ref = next(r for r in dailymed_refs if r.uri.name == "spl_sections.parquet")
    cases_ref = faers_refs[0]  # global deduped cases.parquet is returned first
    assert cases_ref.uri.name == "cases.parquet"
    return sections_ref, cases_ref


def test_transformer_end_to_end_over_fixtures(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    sections_ref, cases_ref = _extract_fixture_refs(ctx)
    transformer = MentionCandidateTransformer(_matcher())
    (out_ref,) = transformer.transform([sections_ref, cases_ref], ctx)

    out_path = ctx.workdir / "data" / "tabular" / "mention_candidates.tsv"
    assert out_ref.uri == out_path
    assert out_path.exists()

    frame = pl.read_csv(out_path, separator="\t")
    assert frame.columns == MENTION_CANDIDATES_COLUMNS
    # Fixture mentions: hypercholesterolemia, headache, pain (DailyMed indications + FAERS)
    # plus asthma (DailyMed contraindication section).
    assert sorted(frame.get_column("candidate_curie").to_list()) == ["HP:0002315", "MONDO:0004979", "MONDO:0005154", "MONDO:0020528"]
    # hypercholesterolemia occurs in both DailyMed and FAERS -> occurrences >= 2.
    chol = frame.filter(pl.col("candidate_curie") == "MONDO:0005154").row(0, named=True)
    assert "occurrences=" in chol["normalization_notes"]
    assert int(chol["normalization_notes"].split("occurrences=")[1].split(";")[0]) >= 2


def test_transformer_manifest_records_provenance(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    sections_ref, cases_ref = _extract_fixture_refs(ctx)
    transformer = MentionCandidateTransformer(_matcher())
    (out_ref,) = transformer.transform([sections_ref, cases_ref], ctx)

    assert out_ref.media_type == "text/tab-separated-values"
    assert out_ref.manifest is not None
    manifest = read_manifest(out_ref.manifest)
    assert manifest.artifact_id == out_ref.blake3
    assert manifest.operation is not None
    assert manifest.operation.name == "ner_mention_candidates"
    assert manifest.table.schema_fingerprint is not None
    # Both input artifacts (DailyMed sections + FAERS cases) are recorded.
    assert sections_ref.blake3 in manifest.inputs
    assert cases_ref.blake3 in manifest.inputs
