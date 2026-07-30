"""Edge-case tests for ``dakp_pipeline.ner.dictionary`` (drive to 100% branch coverage).

Targets every remaining branch: ``normalize_with_map``'s well-formed-tag vs lone-'<' paths,
possessive dropping, ASCII-alnum vs punctuation folding, whitespace-run collapse and end
trimming, the ``canonical_type`` alias table and its lowercased fallback, and ``Gazetteer``
empty-key / empty-row skipping in the constructors and builders.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.ner.dictionary import TYPE_DISEASE, TYPE_PHENOTYPE, Gazetteer, canonical_type, normalize_text, normalize_with_map

# --- normalize_with_map: empty / angle-bracket branches ------------------------


def test_normalize_with_map_empty_string() -> None:
    # Empty input never enters the scan or collapse loops and leaves nothing to trim.
    assert normalize_with_map("") == ("", [])


def test_normalize_with_map_lone_angle_bracket_trims_to_empty() -> None:
    # A lone '<' (no closing '>') emits one boundary space that trims away entirely.
    normalized, index_map = normalize_with_map("<")
    assert normalized == ""
    assert index_map == []


def test_normalize_with_map_well_formed_tag_maps_boundary_to_open_bracket() -> None:
    text = "a <b> c"
    normalized, index_map = normalize_with_map(text)
    assert normalized == "a c"
    # Text after the tag survives and still maps back to its true original offset.
    assert index_map[normalized.index("c")] == text.index("c")


# --- normalize_with_map: possessive branches -----------------------------------


def test_normalize_with_map_drops_possessive_apostrophe_s_both_cases() -> None:
    normalized, index_map = normalize_with_map("cat's dog'S tail")
    assert normalized == "cat dog tail"
    assert len(index_map) == len(normalized)


def test_normalize_with_map_apostrophe_at_end_is_punctuation() -> None:
    # Trailing apostrophe has no following char -> falls through to the punctuation rule.
    normalized, _ = normalize_with_map("patients'")
    assert normalized == "patients"


def test_normalize_with_map_apostrophe_not_followed_by_s_is_punctuation() -> None:
    normalized, _ = normalize_with_map("o'clock")
    assert normalized == "o clock"


# --- normalization: ascii-alnum vs punctuation / non-ascii ---------------------


def test_normalize_text_folds_non_ascii_and_punctuation_to_spaces() -> None:
    assert normalize_text("café pain") == "caf pain"
    assert normalize_text("naïve") == "na ve"
    assert normalize_text("a&b") == "a b"


def test_normalize_with_map_collapses_runs_and_trims_both_ends() -> None:
    normalized, index_map = normalize_with_map("  a   b  ")
    assert normalized == "a b"
    # Leading run trimmed; the single interior space maps to the run's FIRST char (index 3).
    assert index_map == [2, 3, 6]


def test_normalize_with_map_trailing_punctuation_trims() -> None:
    # Trailing punctuation becomes a boundary space that the trailing-trim step pops.
    normalized, index_map = normalize_with_map("abc,")
    assert normalized == "abc"
    assert index_map == [0, 1, 2]


# --- canonical_type: every alias + fallbacks -----------------------------------


def test_canonical_type_every_alias_and_fallbacks() -> None:
    assert canonical_type("disease") == TYPE_DISEASE
    assert canonical_type("diseases") == TYPE_DISEASE
    assert canonical_type("phenotype") == TYPE_PHENOTYPE
    assert canonical_type("phenotypes") == TYPE_PHENOTYPE
    assert canonical_type("phenotypicfeature") == TYPE_PHENOTYPE
    assert canonical_type("phenotypic_feature") == TYPE_PHENOTYPE
    assert canonical_type("CHEMICAL") == "chemical"  # unknown label -> lowercased
    assert canonical_type("") == ""  # empty -> empty fallback
    assert canonical_type("   ") == ""  # whitespace-only -> empty fallback


# --- Gazetteer: empty-key skip + empty construction ----------------------------


def test_gazetteer_skips_surfaces_that_normalize_to_empty() -> None:
    gaz = Gazetteer({"!!!": "disease", "asthma": "disease"})
    assert len(gaz) == 1
    assert gaz.type_for("asthma") == TYPE_DISEASE
    assert "" not in gaz


def test_empty_gazetteer_iteration_and_queries() -> None:
    gaz = Gazetteer({})
    assert gaz.normalized_terms() == ()
    assert list(gaz.items()) == []
    assert len(gaz) == 0
    assert gaz.type_for("anything") is None
    assert "anything" not in gaz


# --- from_frame: blank/null surface + type skipping, biolink categories --------


def test_from_frame_skips_blank_and_null_surface_and_type_rows() -> None:
    frame = pl.DataFrame({"text": ["asthma", "", "   ", None, "fever", "headache"], "type": ["Disease", "disease", "phenotype", "disease", "", None]})
    gaz = Gazetteer.from_frame(frame)
    # Only the row with a non-blank surface AND a non-blank type survives.
    assert gaz.normalized_terms() == ("asthma",)
    assert gaz.type_for("asthma") == TYPE_DISEASE


def test_from_frame_accepts_biolink_categories() -> None:
    frame = pl.DataFrame({"text": ["asthma", "headache"], "type": ["Disease", "PhenotypicFeature"]})
    gaz = Gazetteer.from_frame(frame)
    assert gaz.type_for("asthma") == TYPE_DISEASE
    assert gaz.type_for("headache") == TYPE_PHENOTYPE


def test_from_frame_handles_empty_frame() -> None:
    frame = pl.DataFrame(schema={"text": pl.String, "type": pl.String})
    assert len(Gazetteer.from_frame(frame)) == 0


# --- from_tsv: builder kwargs passthrough --------------------------------------


def test_from_tsv_accepts_builder_kwargs(tmp_path: Path) -> None:
    path = tmp_path / "custom.tsv"
    path.write_text("term\tkind\nasthma\tDisease\n")
    gaz = Gazetteer.from_tsv(path, text_col="term", type_col="kind")
    assert gaz.type_for("asthma") == TYPE_DISEASE
