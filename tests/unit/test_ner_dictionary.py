"""Unit tests for the normalized disease/phenotype gazetteer (span detection only).

Covers the public API of ``dakp_pipeline.ner.dictionary``: deterministic text normalization
(with offset maps back into the original text), raw-label canonicalization, and the immutable
``Gazetteer`` — construction, frame/TSV builders, and lookup/iteration queries.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.ner.dictionary import (
    CONTRAINDICATION_DISEASE_TYPES,
    TYPE_DISEASE,
    TYPE_PHENOTYPE,
    Gazetteer,
    canonical_type,
    normalize_text,
    normalize_with_map,
)

# --- constants -----------------------------------------------------------------


def test_entity_type_constants() -> None:
    assert TYPE_DISEASE == "disease"
    assert TYPE_PHENOTYPE == "phenotype"
    assert CONTRAINDICATION_DISEASE_TYPES == ("disease", "phenotype")


# --- normalize_text ------------------------------------------------------------


def test_normalize_text_lowercases_folds_punctuation_and_trims() -> None:
    assert normalize_text("  Hypercholesterolemia,  (adult) ") == "hypercholesterolemia adult"
    assert normalize_text("Peptic-Ulcer Disease") == "peptic ulcer disease"
    assert normalize_text("fever, chills; pain!") == "fever chills pain"


def test_normalize_text_strips_html_tags_and_possessives() -> None:
    assert normalize_text("patient's <b>headache</b>") == "patient headache"
    assert normalize_text("Parkinson's disease") == "parkinson disease"


def test_normalize_text_empty_and_whitespace_only() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_normalize_text_is_deterministic() -> None:
    value = "Mixed CASE, <i>renal</i> failure's effect"
    assert normalize_text(value) == normalize_text(value)


# --- normalize_with_map --------------------------------------------------------


def test_normalize_with_map_round_trips_offsets() -> None:
    text = "Adult  Hypercholesterolemia,"
    normalized, index_map = normalize_with_map(text)
    assert normalized == "adult hypercholesterolemia"
    assert len(index_map) == len(normalized)
    reconstructed = "".join(text[i] for i in index_map)
    assert reconstructed.lower() == normalized


def test_normalize_with_map_offsets_point_at_original_surface() -> None:
    text = "relief of HEADACHE and pain"
    normalized, index_map = normalize_with_map(text)
    start = normalized.index("headache")
    orig_start = index_map[start]
    orig_end = index_map[start + len("headache") - 1] + 1
    assert text[orig_start:orig_end] == "HEADACHE"


def test_normalize_with_map_handles_html_and_possessive_offsets() -> None:
    text = "the <b>asthma</b> patient's pain"
    normalized, index_map = normalize_with_map(text)
    assert normalized == "the asthma patient pain"
    assert text[index_map[normalized.index("asthma")]] == "a"
    pstart = normalized.index("pain")
    assert text[index_map[pstart] : index_map[pstart] + 4] == "pain"


def test_normalize_with_map_agrees_with_normalize_text() -> None:
    cases = [
        "plain text",
        "5 < 10 mg",
        "heart <b failure",
        "a < b and c > d",
        "a <b> c",
        "patient's <b>headache</b>",
        "Peptic-Ulcer Disease",
        "  leading and trailing  ",
        "<leading> tag",
        "trailing tag <x",
    ]
    for text in cases:
        normalized, index_map = normalize_with_map(text)
        assert normalized == normalize_text(text)
        assert len(index_map) == len(normalized)


def test_normalize_with_map_lone_angle_bracket_keeps_following_text() -> None:
    text = "take 5 < 10 mg aspirin"
    normalized, index_map = normalize_with_map(text)
    assert normalized == "take 5 10 mg aspirin"
    assert index_map[normalized.index("aspirin")] == text.index("aspirin")


# --- canonical_type ------------------------------------------------------------


def test_canonical_type_maps_aliases() -> None:
    assert canonical_type("disease") == TYPE_DISEASE
    assert canonical_type("Diseases") == TYPE_DISEASE
    assert canonical_type("phenotype") == TYPE_PHENOTYPE
    assert canonical_type("Phenotypes") == TYPE_PHENOTYPE
    assert canonical_type("PhenotypicFeature") == TYPE_PHENOTYPE
    assert canonical_type("phenotypic_feature") == TYPE_PHENOTYPE


def test_canonical_type_unknown_label_lowercased_fallback() -> None:
    assert canonical_type("SmallMolecule") == "smallmolecule"
    assert canonical_type("  Disease  ") == TYPE_DISEASE  # stripped before lookup


# --- Gazetteer construction ----------------------------------------------------


def test_gazetteer_normalizes_keys_and_canonicalizes_types() -> None:
    gaz = Gazetteer({"  Peptic Ulcer Disease ": "Diseases", "HEADACHE": "PhenotypicFeature"})
    assert gaz.type_for("peptic ulcer disease") == TYPE_DISEASE
    assert gaz.type_for("headache") == TYPE_PHENOTYPE


def test_gazetteer_last_inserted_type_wins_for_duplicate_keys() -> None:
    gaz = Gazetteer({"fever": "disease", "Fever": "phenotype"})
    assert len(gaz) == 1
    assert gaz.type_for("fever") == TYPE_PHENOTYPE


# --- builders ------------------------------------------------------------------


def test_from_frame_builds_and_canonicalizes_biolink_categories() -> None:
    frame = pl.DataFrame(
        {
            "text": ["myocardial infarction", "headache"],
            "type": ["Disease", "PhenotypicFeature"],
            "curie": ["MONDO:0005015", "HP:0002315"],  # ignored: span detection only
        }
    )
    gaz = Gazetteer.from_frame(frame)
    assert gaz.type_for("myocardial infarction") == TYPE_DISEASE
    assert gaz.type_for("headache") == TYPE_PHENOTYPE
    assert len(gaz) == 2


def test_from_frame_honors_custom_columns() -> None:
    frame = pl.DataFrame({"term": ["asthma"], "kind": ["disease"]})
    gaz = Gazetteer.from_frame(frame, text_col="term", type_col="kind")
    assert gaz.type_for("asthma") == TYPE_DISEASE


def test_from_tsv_reads_tab_separated_terms(tmp_path: Path) -> None:
    path = tmp_path / "terms.tsv"
    path.write_text("text\ttype\nasthma\tDisease\nfever\tphenotype\n")
    gaz = Gazetteer.from_tsv(path)
    assert gaz.type_for("asthma") == TYPE_DISEASE
    assert gaz.type_for("fever") == TYPE_PHENOTYPE


# --- queries -------------------------------------------------------------------


def test_type_for_hit_and_miss() -> None:
    gaz = Gazetteer({"asthma": "disease"})
    assert gaz.type_for("asthma") == TYPE_DISEASE
    assert gaz.type_for("missing") is None


def test_contains_true_and_false() -> None:
    gaz = Gazetteer({"asthma": "disease"})
    assert "asthma" in gaz
    assert "zzz" not in gaz


def test_len_counts_distinct_normalized_phrases() -> None:
    assert len(Gazetteer({})) == 0
    assert len(Gazetteer({"asthma": "disease", "Asthma": "disease", "fever": "phenotype"})) == 2


def test_normalized_terms_sorted() -> None:
    gaz = Gazetteer({"pain": "disease", "asthma": "disease", "fever": "phenotype"})
    assert gaz.normalized_terms() == ("asthma", "fever", "pain")


def test_items_yields_sorted_pairs() -> None:
    gaz = Gazetteer({"pain": "disease", "asthma": "disease", "fever": "phenotype"})
    assert list(gaz.items()) == [("asthma", "disease"), ("fever", "phenotype"), ("pain", "disease")]
