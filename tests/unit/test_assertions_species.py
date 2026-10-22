"""Unit tests for the label-type-gated species derivation (assertions/species.py).

Species is part of DAKP edge identity, so these tests pin the three properties that keep
animal-label assertions from folding onto human edges: non-animal sources are always
human, animal rows take their species from the text (sentence first, label-wide second),
and an animal row with no species is unattributable (empty) rather than defaulted to
human.
"""

from __future__ import annotations

import re

import pytest

from dakp_pipeline.assertions import species
from dakp_pipeline.assertions.species import HUMAN_TAXON, SPECIES_LEXICON, label_species, row_species, species_in_text

DOG = "NCBITaxon:9615"
CAT = "NCBITaxon:9685"
CATTLE = "NCBITaxon:9913"
HORSE = "NCBITaxon:9796"
CHICKEN = "NCBITaxon:9031"


@pytest.mark.parametrize("label_type", [None, "", "34391-4", "34390-5", "  "])
def test_non_animal_sources_are_always_human(label_type: str | None) -> None:
    """Human labels, unknown label types, and non-DailyMed sources are human by construction.

    A species word on a human label ("in rats") is study context, so even a sentence full
    of animal words must not move a human-label row off 9606.
    """
    assert row_species(label_type, "Carcinogenicity studies in rats and dogs.", frozenset({DOG})) == (HUMAN_TAXON,)


@pytest.mark.parametrize("label_type", ["50578-4", "50577-6"])
def test_animal_row_takes_sentence_species_first(label_type: str) -> None:
    """The row's own sentence wins over the label-wide species set."""
    assert row_species(label_type, "For the treatment of pain in horses.", frozenset({DOG, CAT})) == (HORSE,)


def test_animal_row_falls_back_to_label_wide_species() -> None:
    """A sentence naming no species inherits the label's species (title/indication text)."""
    assert row_species("50578-4", "Contraindicated in animals with renal disease.", frozenset({DOG})) == (DOG,)


def test_animal_row_multi_species_sentence_yields_each_species_sorted() -> None:
    """ "dogs and cats" asserts the relationship for both organisms: one entry per species."""
    assert row_species("50577-6", "For use in dogs and cats.", frozenset()) == (DOG, CAT)


def test_animal_row_without_any_species_is_unattributable() -> None:
    """No species anywhere -> empty tuple; the caller drops it rather than defaulting to human.

    Defaulting would put an animal assertion on the human edge, the exact fold this
    feature exists to prevent.
    """
    assert row_species("50578-4", "Contraindicated in animals with renal disease.", frozenset()) == ()


def test_human_words_never_attribute_an_animal_row() -> None:
    """ "Not for use in humans" on an animal label is a user-safety warning, not the organism."""
    assert species_in_text("Not for use in humans. Keep out of reach of children.") == frozenset()
    assert row_species("50578-4", "Not for use in humans.", frozenset()) == ()


def test_species_in_text_matching_rules() -> None:
    """Whole-word, case-insensitive, flexible inner whitespace, longest entry first."""
    assert species_in_text("CATTLE and Swine") == frozenset({CATTLE, "NCBITaxon:9823"})
    assert species_in_text("honey  bees") == frozenset({"NCBITaxon:7460"})
    assert species_in_text("laying hens") == frozenset({CHICKEN})
    assert species_in_text("concatenate dogma catalog") == frozenset()  # no substring hits
    assert species_in_text(None) == frozenset()
    assert species_in_text("") == frozenset()


def test_ambiguous_group_and_non_species_words_are_excluded() -> None:
    """Group words and words with a dominant non-species reading never match.

    Batch resolution against the production fullmap showed these are unsafe: kids/steer
    resolve to diseases/phenomena, and group words (poultry, fish, livestock) name no
    single organism.
    """
    for word in ("poultry", "fish", "livestock", "animals", "birds", "kids", "man", "steer", "people", "human", "humans"):
        assert word not in SPECIES_LEXICON
        assert species_in_text(f"for use in {word}") == frozenset()


def test_label_species_unions_section_texts() -> None:
    """The label-wide set is the union over every mined section (None-safe)."""
    assert label_species(["For dogs only.", None, "Safe in cats.", ""]) == frozenset({DOG, CAT})
    assert label_species([]) == frozenset()


def test_lexicon_values_are_species_level_ncbitaxon_curies() -> None:
    """Every lexicon value is an NCBITaxon CURIE; keys are lowercase surface forms.

    The values ride the species_context_qualifier verbatim and are re-verified against a
    fullmap by tests/eval/species_lexicon_audit.py.
    """
    assert all(re.fullmatch(r"NCBITaxon:\d+", curie) for curie in SPECIES_LEXICON.values())
    assert all(key == key.lower().strip() for key in SPECIES_LEXICON)
    assert HUMAN_TAXON not in SPECIES_LEXICON.values()
    assert species.__all__ == ["HUMAN_TAXON", "SPECIES_LEXICON", "animal_row_species", "label_species", "row_species", "species_in_text"]
