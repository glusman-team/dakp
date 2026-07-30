"""Edge-case tests for ``dakp_pipeline.ner.lexical`` (drive to 100% branch coverage).

Targets the private helpers directly (``_find_word_bounded`` empty-needle guard + boundary
positions, ``_overlaps_any`` short-circuit branches, ``_mention_sort_key``), the frozen/
hashable :class:`Mention`, and every remaining ``match`` branch: empty gazetteer, the
empty/blank/ignored whole-field guard, custom + empty ignore lists, the opt-in synonym path
(including an identity synonym and a multi-synonym ``;``-joined note), and per-mention
ignore filtering.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dakp_pipeline.ner.dictionary import Gazetteer
from dakp_pipeline.ner.lexical import (
    DEFAULT_IGNORE_TERMS,
    DIRECT_SCORE,
    LEGACY_SYNONYMS,
    SYNONYM_SCORE,
    LexicalMatcher,
    Mention,
    _find_word_bounded,
    _mention_sort_key,
    _overlaps_any,
)


def _matcher(terms: dict[str, str], **kwargs: object) -> LexicalMatcher:
    return LexicalMatcher(Gazetteer(terms), **kwargs)  # type: ignore[arg-type]


# --- _find_word_bounded helper --------------------------------------------------


def test_find_word_bounded_empty_needle_returns_empty() -> None:
    assert _find_word_bounded("asthma pain", "") == []


def test_find_word_bounded_multiple_occurrences() -> None:
    assert _find_word_bounded("pain and pain", "pain") == [0, 9]


def test_find_word_bounded_boundary_positions() -> None:
    assert _find_word_bounded("pain a", "pain") == [0]  # bounded by start-of-string + space
    assert _find_word_bounded("a pain", "pain") == [2]  # bounded by space + end-of-string
    assert _find_word_bounded("pain", "pain") == [0]  # whole string is the word


def test_find_word_bounded_rejects_inner_substrings() -> None:
    assert _find_word_bounded("painting", "pain") == []  # trailing chars -> no boundary after
    assert _find_word_bounded("repainting", "pain") == []  # leading chars -> no boundary before


# --- _overlaps_any helper -------------------------------------------------------


def test_overlaps_any_empty_covered_is_false() -> None:
    assert _overlaps_any(0, 4, []) is False


def test_overlaps_any_true_overlap() -> None:
    assert _overlaps_any(5, 10, [(0, 6)]) is True


def test_overlaps_any_new_span_after_covered_is_false() -> None:
    # start >= cov_end short-circuits the ``and`` (covered span is entirely before).
    assert _overlaps_any(10, 15, [(0, 6)]) is False


def test_overlaps_any_new_span_before_covered_is_false() -> None:
    # start < cov_end but cov_start >= end (covered span is entirely after).
    assert _overlaps_any(0, 4, [(9, 20)]) is False


# --- _mention_sort_key helper ---------------------------------------------------


def test_mention_sort_key_is_start_end_type_text() -> None:
    mention = Mention(text="asthma", start=3, end=9, type="disease", score=DIRECT_SCORE)
    assert _mention_sort_key(mention) == (3, 9, "disease", "asthma")


# --- Mention dataclass ----------------------------------------------------------


def test_mention_default_optional_fields_are_empty() -> None:
    mention = Mention(text="x", start=0, end=1, type="disease", score=DIRECT_SCORE)
    assert (mention.normalized, mention.notes, mention.section) == ("", "", "")


def test_mention_is_frozen() -> None:
    mention = Mention(text="asthma", start=0, end=6, type="disease", score=DIRECT_SCORE)
    with pytest.raises(FrozenInstanceError):
        mention.text = "changed"  # type: ignore[misc]


def test_mention_is_hashable_and_value_equal() -> None:
    a = Mention(text="asthma", start=0, end=6, type="disease", score=DIRECT_SCORE)
    b = Mention(text="asthma", start=0, end=6, type="disease", score=DIRECT_SCORE)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# --- match: empty/blank/ignored guard -------------------------------------------


def test_match_empty_gazetteer_yields_nothing() -> None:
    assert LexicalMatcher(Gazetteer({})).match("asthma") == []


def test_match_empty_and_blank_text_yield_nothing() -> None:
    matcher = _matcher({"asthma": "disease"})
    assert matcher.match("") == []
    assert matcher.match("   \t\n ") == []


def test_is_ignored_text_true_and_false() -> None:
    matcher = _matcher({"asthma": "disease"})
    assert matcher.is_ignored_text("Off label use")  # normalized before lookup
    assert not matcher.is_ignored_text("asthma")


def test_match_whole_field_ignore_suppresses_even_with_inner_term() -> None:
    matcher = _matcher({"pain": "phenotype"}, ignore_terms=["pain"])
    # Whole-field ignore wins despite "pain" being a matchable term.
    assert matcher.match("pain") == []


def test_match_empty_custom_ignore_list_ignores_nothing() -> None:
    matcher = _matcher({"asthma": "disease"}, ignore_terms=[])
    # An empty custom ignore list replaces the defaults and suppresses nothing.
    assert not matcher.is_ignored_text("off label use")
    assert [m.text for m in matcher.match("asthma")] == ["asthma"]


def test_match_per_mention_ignore_returns_none_for_that_span_only() -> None:
    matcher = _matcher({"prophylaxis": "disease", "asthma": "disease"})
    mentions = matcher.match("asthma and prophylaxis")
    assert [m.normalized for m in mentions] == ["asthma"]


# --- match: synonyms ------------------------------------------------------------


def test_match_without_synonyms_skips_synonym_path() -> None:
    matcher = _matcher({"heart failure": "disease"})  # synonyms=None
    assert matcher.match("cardiac failure") == []


def test_match_identity_synonym_is_not_noted() -> None:
    # A synonym mapping a token to itself performs no substitution (replacement == token).
    matcher = _matcher({"asthma": "disease"}, synonyms={"asthma": "asthma"})
    mentions = matcher.match("asthma")
    assert mentions[0].score == DIRECT_SCORE
    assert mentions[0].notes == "exact"


def test_match_synonym_keeps_original_offsets_and_lower_score() -> None:
    matcher = _matcher({"heart failure": "disease"}, synonyms=LEGACY_SYNONYMS)
    text = "cardiac failure"
    mentions = matcher.match(text)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.text == "cardiac failure"
    assert text[m.start : m.end] == m.text
    assert m.score == SYNONYM_SCORE
    assert m.notes == "synonym:cardiac>heart"
    # A token with no synonym entry ("failure") matches unchanged alongside the substitution.
    assert m.normalized == "cardiac failure"


def test_match_multiple_synonyms_join_notes_with_semicolon() -> None:
    matcher = _matcher({"kidney heart failure": "disease"}, synonyms=LEGACY_SYNONYMS)
    text = "renal cardiac failure"
    mentions = matcher.match(text)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.text == "renal cardiac failure"
    assert text[m.start : m.end] == m.text
    assert m.score == SYNONYM_SCORE
    assert m.notes == "synonym:cardiac>heart;synonym:renal>kidney"  # sorted + ';' joined


# --- constants ------------------------------------------------------------------


def test_default_ignore_terms_are_normalized() -> None:
    assert isinstance(DEFAULT_IGNORE_TERMS, frozenset)
    assert "product used for unknown indication" in DEFAULT_IGNORE_TERMS
    assert all(term == term.strip().lower() for term in DEFAULT_IGNORE_TERMS)


def test_legacy_synonyms_mapping() -> None:
    assert dict(LEGACY_SYNONYMS) == {"cardiac": "heart", "renal": "kidney", "hepatic": "liver"}
