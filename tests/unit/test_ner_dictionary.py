"""Unit tests for the normalized dictionary index (PLAN.md Phase 4 fast baseline).

Covers: deterministic normalization, offset-map correctness (true offsets back into the
original text), index build from the pipeline ontology fixture, CURIE-prefix source
inference, category -> semantic-group mapping, alias indexing, and lookup determinism.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.ner.dictionary import DictionaryEntry, DictionaryIndex, infer_source, normalize_text, normalize_with_map, semantic_group_for

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_ONTOLOGY_TSV = _FIXTURE_ROOT / "ontology" / "disease_map.tsv"


# --- normalization -------------------------------------------------------------


def test_normalize_text_is_lowercase_punct_folded_and_trimmed() -> None:
    assert normalize_text("  Hypercholesterolemia,  (adult) ") == "hypercholesterolemia adult"
    assert normalize_text("Peptic-Ulcer Disease") == "peptic ulcer disease"
    assert normalize_text("") == ""


def test_normalize_text_strips_html_and_possessive() -> None:
    assert normalize_text("patient's <b>headache</b>") == "patient headache"
    assert normalize_text("Parkinson's disease") == "parkinson disease"


def test_normalize_text_is_deterministic() -> None:
    value = "Mixed CASE, <i>renal</i> failure's effect"
    assert normalize_text(value) == normalize_text(value)


def test_normalize_with_map_round_trips_offsets() -> None:
    text = "Adult  Hypercholesterolemia,"
    normalized, index_map = normalize_with_map(text)
    assert normalized == "adult hypercholesterolemia"
    assert len(index_map) == len(normalized)
    # Every normalized character maps back to the same character in the original text.
    reconstructed = "".join(text[i] for i in index_map)
    assert reconstructed.lower() == normalized


def test_normalize_with_map_offsets_point_at_original_surface() -> None:
    text = "relief of HEADACHE and pain"
    normalized, index_map = normalize_with_map(text)
    start = normalized.index("headache")
    end = start + len("headache")
    # Mapping the normalized span back yields the original (mixed-case) surface form.
    orig_start = index_map[start]
    orig_end = index_map[end - 1] + 1
    assert text[orig_start:orig_end] == "HEADACHE"


def test_normalize_with_map_handles_html_and_possessive_offsets() -> None:
    text = "the <b>asthma</b> patient's pain"
    normalized, index_map = normalize_with_map(text)
    assert normalized == "the asthma patient pain"
    start = normalized.index("asthma")
    assert text[index_map[start]] == "a"
    # "pain" still maps to the true trailing surface form.
    pstart = normalized.index("pain")
    assert text[index_map[pstart] : index_map[pstart] + 4] == "pain"


def test_normalize_with_map_agrees_with_normalize_text() -> None:
    """The offset-preserving normalizer must produce the same string as normalize_text,
    including the lone-'<' / unterminated-tag cases that previously diverged."""
    cases = [
        "plain text",
        "5 < 10 mg",  # lone '<' with no closing '>'
        "heart <b failure",  # unterminated tag
        "a < b and c > d",  # '<' ... '>' span
        "patient's <b>headache</b>",  # well-formed tag + possessive
        "Peptic-Ulcer Disease",
        "  leading and trailing  ",
        "<leading> tag",
        "trailing tag <x",
    ]
    for text in cases:
        normalized, index_map = normalize_with_map(text)
        assert normalized == normalize_text(text), text
        assert len(index_map) == len(normalized), text


def test_normalize_with_map_lone_angle_bracket_keeps_following_text() -> None:
    normalized, index_map = normalize_with_map("take 5 < 10 mg aspirin")
    assert normalized == "take 5 10 mg aspirin"
    # The text after the lone '<' is preserved and still maps back to the original.
    start = normalized.index("aspirin")
    assert index_map[start] == "take 5 < 10 mg aspirin".index("aspirin")


# --- source / category semantics ----------------------------------------------


def test_infer_source_from_curie_prefix() -> None:
    assert infer_source("MONDO:0005154") == "MONDO"
    assert infer_source("HP:0002315") == "HPO"
    assert infer_source("DRUGBANK:DB00001") == "DRUGBANK"
    assert infer_source("UNKNOWN:123") == "fullmap"
    assert infer_source("noprefix") == "fullmap"


def test_semantic_group_for_categories() -> None:
    assert semantic_group_for("Disease") == "disease"
    assert semantic_group_for("PhenotypicFeature") == "phenotype"
    assert semantic_group_for("SmallMolecule") == "drug"
    assert semantic_group_for("  disease ") == "disease"
    # Unknown categories fall back to the dominant DAKP object type.
    assert semantic_group_for("SomethingElse") == "disease"


# --- index build from the pipeline fixture ------------------------------------


def test_index_builds_from_ontology_fixture() -> None:
    index = DictionaryIndex.from_tsv(_ONTOLOGY_TSV)
    # 5 distinct normalized terms in disease_map.tsv.
    assert len(index) == 5
    assert "hypercholesterolemia" in index
    assert "peptic ulcer disease" in index
    assert "not-a-term" not in index


def test_index_lookup_returns_candidate_metadata() -> None:
    index = DictionaryIndex.from_tsv(_ONTOLOGY_TSV)
    (entry,) = index.lookup("hypercholesterolemia")
    assert entry.curie == "MONDO:0005154"
    assert entry.name == "hypercholesterolemia"
    assert entry.category == "Disease"
    assert entry.source == "MONDO"  # inferred from the MONDO: prefix
    assert entry.semantic_group == "disease"

    (hp,) = index.lookup("headache")
    assert hp.curie == "HP:0002315"
    assert hp.source == "HPO"
    assert hp.semantic_group == "phenotype"


def test_index_lookup_text_normalizes_before_lookup() -> None:
    index = DictionaryIndex.from_tsv(_ONTOLOGY_TSV)
    (entry,) = index.lookup_text("  Peptic Ulcer Disease ")
    assert entry.curie == "MONDO:0005194"


def test_index_lookup_missing_returns_empty_tuple() -> None:
    index = DictionaryIndex.from_tsv(_ONTOLOGY_TSV)
    assert index.lookup("zzz") == ()


def test_index_build_is_deterministic() -> None:
    a = DictionaryIndex.from_tsv(_ONTOLOGY_TSV)
    b = DictionaryIndex.from_tsv(_ONTOLOGY_TSV)
    assert a.normalized_terms() == b.normalized_terms()
    assert [e.curie for e in a.entries()] == [e.curie for e in b.entries()]


def test_index_deduplicates_identical_entries() -> None:
    entries = [
        DictionaryEntry("pain", "MONDO:0020528", "pain", "Disease", "MONDO", "pain"),
        DictionaryEntry("pain", "MONDO:0020528", "pain", "Disease", "MONDO", "pain"),
    ]
    index = DictionaryIndex.from_entries(entries)
    assert len(index.lookup("pain")) == 1


def test_index_orders_multiple_candidates_deterministically() -> None:
    # Two distinct CURIEs share one normalized string -> sorted by (source, category, curie).
    entries = [
        DictionaryEntry("fever", "HP:0001945", "fever", "PhenotypicFeature", "HPO", "fever"),
        DictionaryEntry("fever", "MONDO:0001234", "fever", "Disease", "MONDO", "fever"),
    ]
    index = DictionaryIndex.from_entries(entries)
    got = index.lookup("fever")
    assert [e.source for e in got] == ["HPO", "MONDO"]  # HPO sorts before MONDO


def test_index_skips_empty_normalized_keys() -> None:
    entries = [DictionaryEntry("", "MONDO:1", "x", "Disease", "MONDO", "!!!")]
    assert len(DictionaryIndex.from_entries(entries)) == 0


def test_from_frame_indexes_alias_columns_to_same_curie() -> None:
    frame = pl.DataFrame(
        {
            "text": ["myocardial infarction"],
            "alias": ["heart attack"],
            "curie": ["MONDO:0005015"],
            "name": ["myocardial infarction"],
            "category": ["Disease"],
        }
    )
    index = DictionaryIndex.from_frame(frame, alias_columns=["alias"])
    assert index.lookup("myocardial infarction")[0].curie == "MONDO:0005015"
    assert index.lookup("heart attack")[0].curie == "MONDO:0005015"


def test_from_frame_source_column_overrides_inference() -> None:
    frame = pl.DataFrame({"text": ["pain"], "curie": ["X:1"], "name": ["pain"], "category": ["Disease"], "src": ["BABEL"]})
    index = DictionaryIndex.from_frame(frame, source_col="src")
    assert index.lookup("pain")[0].source == "BABEL"
