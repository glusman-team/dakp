"""Edge-case tests for ``dakp_pipeline.ner.lexical`` (drive to 100% branch coverage).

Covers the empty-needle guard of the private ``_find_word_bounded`` helper plus adversarial
matching: blank/ignored text, whole-field vs per-mention ignore, opt-in synonyms (with the
lower synonym score and preserved original offsets), greedy longest-first matching, multiple
candidate entries per span, section propagation, and determinism.
"""

from __future__ import annotations

from dakp_pipeline.ner.dictionary import DictionaryEntry, DictionaryIndex
from dakp_pipeline.ner.lexical import DEFAULT_IGNORE_TERMS, DIRECT_SCORE, LEGACY_SYNONYMS, SYNONYM_SCORE, LexicalMatcher, _find_word_bounded


def _entry(normalized: str, curie: str, category: str = "Disease", source: str = "MONDO", name: str | None = None) -> DictionaryEntry:
    return DictionaryEntry(normalized, curie, name or normalized, category, source, normalized)


def _index(*entries: DictionaryEntry) -> DictionaryIndex:
    return DictionaryIndex.from_entries(list(entries))


# --- _find_word_bounded helper --------------------------------------------------


def test_find_word_bounded_empty_needle_returns_empty() -> None:
    assert _find_word_bounded("asthma pain", "") == []


def test_find_word_bounded_multiple_and_boundary_positions() -> None:
    assert _find_word_bounded("pain and pain", "pain") == [0, 9]
    assert _find_word_bounded("painting", "pain") == []  # not word-bounded
    assert _find_word_bounded("a pain", "pain") == [2]  # bounded by start-space + end-of-string


# --- empty / blank / ignored text -----------------------------------------------


def test_match_empty_and_blank_text_yields_nothing() -> None:
    matcher = LexicalMatcher(_index(_entry("asthma", "MONDO:1")))
    assert matcher.match("") == []
    assert matcher.match("   \t\n ") == []


def test_match_whole_field_ignore_term_suppresses_record() -> None:
    matcher = LexicalMatcher(_index(_entry("asthma", "MONDO:1")))
    # A whole indication string in the legacy ignore set yields no mentions.
    assert matcher.is_ignored_text("Off label use")
    assert matcher.match("off label use") == []
    # A non-ignored field is not suppressed.
    assert not matcher.is_ignored_text("asthma")


def test_match_custom_ignore_terms_override_defaults() -> None:
    matcher = LexicalMatcher(_index(_entry("asthma", "MONDO:1")), ignore_terms=["asthma"])
    # Custom ignore list REPLACES the defaults: 'asthma' is now ignored whole-field...
    assert matcher.match("asthma") == []
    # ...and a legacy default ignore term is no longer ignored.
    assert not matcher.is_ignored_text("off label use")


def test_match_per_mention_ignore_filter() -> None:
    # 'prophylaxis' is a default ignore term; as an individual mention inside a larger field
    # it is dropped while a real disease mention survives.
    matcher = LexicalMatcher(_index(_entry("asthma", "MONDO:1"), _entry("prophylaxis", "MONDO:2")))
    mentions = matcher.match("asthma and prophylaxis")
    assert [m.normalized for m in mentions] == ["asthma"]


# --- synonyms -------------------------------------------------------------------


def test_match_synonym_substitution_keeps_original_offsets_and_lower_score() -> None:
    index = _index(_entry("heart failure", "MONDO:1", name="heart failure"))
    matcher = LexicalMatcher(index, synonyms=LEGACY_SYNONYMS)
    text = "cardiac failure"
    mentions = matcher.match(text)
    assert len(mentions) == 1
    mention = mentions[0]
    # The reported surface form/offsets stay the ORIGINAL words ('cardiac failure')...
    assert mention.mention_text == "cardiac failure"
    assert text[mention.mention_start : mention.mention_end] == mention.mention_text
    # ...while matching via the synonymized concept ('heart').
    assert mention.score == SYNONYM_SCORE
    assert "synonym:cardiac>heart" in mention.notes


def test_match_direct_score_and_exact_notes_without_synonyms() -> None:
    matcher = LexicalMatcher(_index(_entry("asthma", "MONDO:1")))
    mentions = matcher.match("asthma")
    assert mentions[0].score == DIRECT_SCORE
    assert mentions[0].notes == "exact"


def test_match_synonym_that_is_identity_is_not_noted() -> None:
    # A synonym mapping a token to itself produces no substitution note (replacement == token).
    index = _index(_entry("asthma", "MONDO:1"))
    matcher = LexicalMatcher(index, synonyms={"asthma": "asthma"})
    mentions = matcher.match("asthma")
    assert mentions[0].score == DIRECT_SCORE
    assert mentions[0].notes == "exact"


# --- greedy matching / multiple entries / section / determinism -----------------


def test_match_greedy_longest_first_prevents_nested_match() -> None:
    index = _index(_entry("peptic ulcer disease", "MONDO:1"), _entry("ulcer", "MONDO:2"))
    mentions = LexicalMatcher(index).match("peptic ulcer disease")
    assert [m.normalized for m in mentions] == ["peptic ulcer disease"]


def test_match_returns_one_mention_per_candidate_entry_for_a_span() -> None:
    # Two CURIEs share the normalized string 'asthma' -> one mention per entry for the span.
    index = _index(_entry("asthma", "MONDO:1"), _entry("asthma", "HP:9", source="HPO"))
    mentions = LexicalMatcher(index).match("asthma")
    assert sorted(m.entry.curie for m in mentions) == ["HP:9", "MONDO:1"]
    # All share the same span offsets.
    assert {(m.mention_start, m.mention_end) for m in mentions} == {(0, 6)}


def test_match_propagates_section_and_is_deterministic() -> None:
    matcher = LexicalMatcher(_index(_entry("asthma", "MONDO:1"), _entry("pain", "MONDO:2")))
    text = "pain and asthma and pain"
    first = [(m.mention_start, m.mention_end, m.entry.curie, m.section) for m in matcher.match(text, section="indications")]
    assert all(m.section == "indications" for m in matcher.match(text, section="indications"))
    for _ in range(5):
        again = [(m.mention_start, m.mention_end, m.entry.curie, m.section) for m in matcher.match(text, section="indications")]
        assert again == first
    # Sorted by (start, end, curie, source).
    starts = [m.mention_start for m in matcher.match(text)]
    assert starts == sorted(starts)


def test_default_ignore_terms_are_normalized() -> None:
    assert "off label use" in DEFAULT_IGNORE_TERMS
    assert all(term == term.strip().lower() for term in DEFAULT_IGNORE_TERMS)
