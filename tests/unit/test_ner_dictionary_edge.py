"""Edge-case tests for ``dakp_pipeline.ner.dictionary`` (drive to 100% branch coverage).

Targets the uncovered empty-surface ``continue`` in ``DictionaryIndex.from_frame`` plus
adversarial normalization (HTML tags, lone ``<``, possessives, unicode), offset-map
round-trips, source/semantic inference fallbacks, and index dedup/ordering/aliasing.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.ner.dictionary import (
    DictionaryEntry,
    DictionaryIndex,
    infer_source,
    normalize_text,
    normalize_with_map,
    semantic_group_for,
)

# --- normalization --------------------------------------------------------------


def test_normalize_text_folds_html_possessives_and_punctuation() -> None:
    assert normalize_text("Peptic Ulcer Disease") == "peptic ulcer disease"
    assert normalize_text("<b>asthma</b>") == "asthma"
    assert normalize_text("patient's asthma") == "patient asthma"  # possessive 's dropped
    assert normalize_text("fever, chills; pain!") == "fever chills pain"
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text("<unclosed") == "unclosed"  # lone '<' treated as a boundary


def test_normalize_text_drops_non_ascii_alphanumerics() -> None:
    # Non-ASCII alphanumerics are folded to spaces (only [a-z0-9] survive).
    assert normalize_text("café pain") == "caf pain"
    assert normalize_text("naïve") == "na ve"


def test_normalize_with_map_offsets_roundtrip_to_original() -> None:
    for text in ("Hepatitis B!", "<b>asthma</b> and pain", "patient's liver disease", "  spaced  out  ", "a"):
        normalized, index_map = normalize_with_map(text)
        assert normalized == normalize_text(text)
        assert len(index_map) == len(normalized)
        # Every normalized character maps back to the identical original character.
        for norm_char, orig_idx in zip(normalized, index_map, strict=True):
            assert text[orig_idx].lower() == norm_char or norm_char == " "


def test_normalize_with_map_handles_empty_and_lone_angle_bracket() -> None:
    assert normalize_with_map("") == ("", [])
    normalized, index_map = normalize_with_map("<")
    assert normalized == ""  # a lone boundary space trims away
    assert isinstance(index_map, list)


# --- source / semantic-group inference ------------------------------------------


def test_infer_source_known_unknown_and_no_prefix() -> None:
    assert infer_source("MONDO:0004979") == "MONDO"
    assert infer_source("HP:0002315") == "HPO"
    assert infer_source("DRUGBANK:DB00001") == "DRUGBANK"
    assert infer_source("UNKNOWN:123") == "fullmap"  # unknown prefix -> fullmap
    assert infer_source("nocolon") == "fullmap"  # no ':' -> fullmap
    assert infer_source("") == "fullmap"


def test_semantic_group_for_known_unknown_and_whitespace() -> None:
    assert semantic_group_for("Disease") == "disease"
    assert semantic_group_for("PhenotypicFeature") == "phenotype"
    assert semantic_group_for("ChemicalEntity") == "drug"
    assert semantic_group_for("  SmallMolecule  ") == "drug"
    assert semantic_group_for("TotallyUnknown") == "disease"  # conservative fallback
    assert semantic_group_for("") == "disease"


# --- DictionaryIndex ------------------------------------------------------------


def _entry(normalized: str, curie: str, category: str = "Disease", source: str = "MONDO", name: str | None = None) -> DictionaryEntry:
    return DictionaryEntry(normalized, curie, name or normalized, category, source, normalized)


def test_index_skips_empty_keys_dedups_and_orders() -> None:
    entries = [
        _entry("asthma", "MONDO:2"),
        _entry("asthma", "MONDO:1"),  # same key, reordered on lookup
        _entry("asthma", "MONDO:1"),  # duplicate -> dropped
        _entry("", "MONDO:empty"),  # empty normalized key -> never indexed
    ]
    index = DictionaryIndex(entries)
    assert "" not in index
    assert len(index) == 1
    looked_up = index.lookup("asthma")
    assert [e.curie for e in looked_up] == ["MONDO:1", "MONDO:2"]  # sorted deterministically
    assert index.lookup("missing") == ()
    assert index.lookup_text("  ASTHMA  ") == looked_up  # normalized lookup


def test_index_entries_and_normalized_terms_are_deterministic() -> None:
    index = DictionaryIndex([_entry("pain", "MONDO:9"), _entry("asthma", "MONDO:1")])
    assert index.normalized_terms() == ("asthma", "pain")
    assert [e.normalized for e in index.entries()] == ["asthma", "pain"]


def test_index_entry_semantic_group_property() -> None:
    assert _entry("asthma", "MONDO:1", category="Disease").semantic_group == "disease"
    assert _entry("headache", "HP:1", category="PhenotypicFeature").semantic_group == "phenotype"
    assert _entry("aspirin", "DRUGBANK:1", category="Drug").semantic_group == "drug"


# --- from_frame: empty-surface skip + aliases + source override -----------------


def test_from_frame_skips_blank_surface_forms() -> None:
    frame = pl.DataFrame(
        {
            "text": ["asthma", "", "   "],
            "curie": ["MONDO:1", "MONDO:2", "MONDO:3"],
            "name": ["asthma", "blank", "spaces"],
            "category": ["Disease", "Disease", "Disease"],
        }
    )
    index = DictionaryIndex.from_frame(frame)
    # Only the non-blank surface form is indexed.
    assert index.normalized_terms() == ("asthma",)
    assert len(index) == 1


def test_from_frame_indexes_alias_columns_to_same_curie() -> None:
    frame = pl.DataFrame(
        {
            "text": ["myocardial infarction"],
            "alias_exact": ["heart attack"],
            "alias_related": [""],  # blank alias skipped
            "curie": ["MONDO:0005015"],
            "name": ["myocardial infarction"],
            "category": ["Disease"],
        }
    )
    index = DictionaryIndex.from_frame(frame, alias_columns=["alias_exact", "alias_related"])
    assert set(index.normalized_terms()) == {"myocardial infarction", "heart attack"}
    # Both surface forms resolve to the same CURIE.
    assert index.lookup_text("heart attack")[0].curie == "MONDO:0005015"


def test_from_frame_source_override_and_name_fallback() -> None:
    frame = pl.DataFrame(
        {
            "text": ["aspirin"],
            "curie": ["UNKNOWN:1"],  # would infer 'fullmap'...
            "name": [""],  # blank name -> falls back to the surface form
            "category": ["Drug"],
            "src": ["DRUGBANK"],
        }
    )
    index = DictionaryIndex.from_frame(frame, source_col="src")
    entry = index.lookup_text("aspirin")[0]
    assert entry.source == "DRUGBANK"  # explicit source column wins over inference
    assert entry.name == "aspirin"  # name fell back to the surface form


def test_from_frame_infers_source_from_curie_when_no_source_column() -> None:
    frame = pl.DataFrame({"text": ["asthma"], "curie": ["MONDO:1"], "name": ["asthma"], "category": ["Disease"]})
    entry = DictionaryIndex.from_frame(frame).lookup_text("asthma")[0]
    assert entry.source == "MONDO"
