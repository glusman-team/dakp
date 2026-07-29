"""Unit tests for the deterministic lexical mention matcher.

Covers: true offsets into the original text (``mention_text == text[start:end]``),
word-boundary matching (no substring false positives), greedy longest-phrase-first
matching, whole-field + per-mention ignore-list suppression, opt-in synonym fallback with
preserved surface offsets, deterministic scoring/ordering, and section-context capture.
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.ner.dictionary import DictionaryEntry, DictionaryIndex
from dakp_pipeline.ner.lexical import DEFAULT_IGNORE_TERMS, DIRECT_SCORE, LEGACY_SYNONYMS, SYNONYM_SCORE, LexicalMatcher

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_ONTOLOGY_TSV = _FIXTURE_ROOT / "ontology" / "disease_map.tsv"


def _fixture_matcher(**kwargs: object) -> LexicalMatcher:
    return LexicalMatcher(DictionaryIndex.from_tsv(_ONTOLOGY_TSV), **kwargs)  # type: ignore[arg-type]


def _entry(normalized: str, curie: str, category: str = "Disease", source: str = "MONDO") -> DictionaryEntry:
    return DictionaryEntry(normalized, curie, normalized, category, source, normalized)


# --- span offsets --------------------------------------------------------------


def test_match_finds_mention_with_true_offsets() -> None:
    matcher = _fixture_matcher()
    text = "Examplestatin is indicated for the treatment of hypercholesterolemia in adults."
    mentions = matcher.match(text)
    assert [m.mention_text for m in mentions] == ["hypercholesterolemia"]
    m = mentions[0]
    # The half-open offsets slice back to the exact surface form.
    assert text[m.mention_start : m.mention_end] == m.mention_text == "hypercholesterolemia"
    assert m.entry.curie == "MONDO:0005154"
    assert m.semantic_group == "disease"
    assert m.score == DIRECT_SCORE
    assert m.notes == "exact"


def test_match_offsets_survive_mixed_case_and_punctuation() -> None:
    matcher = _fixture_matcher()
    text = "Relief of HEADACHE, and mild-to-moderate Pain."
    mentions = matcher.match(text)
    by_text = {m.mention_text: m for m in mentions}
    assert set(by_text) == {"HEADACHE", "Pain"}
    for m in mentions:
        assert text[m.mention_start : m.mention_end] == m.mention_text
    assert by_text["HEADACHE"].entry.curie == "HP:0002315"
    assert by_text["Pain"].normalized == "pain"  # normalized surface, original case preserved


def test_multiword_phrase_offsets_span_the_full_surface() -> None:
    matcher = _fixture_matcher()
    text = "History of peptic ulcer disease."
    mentions = matcher.match(text)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.mention_text == "peptic ulcer disease"
    assert text[m.mention_start : m.mention_end] == "peptic ulcer disease"
    assert m.entry.curie == "MONDO:0005194"


def test_match_returns_empty_for_no_terms_or_blank_text() -> None:
    matcher = _fixture_matcher()
    assert matcher.match("gastroesophageal reflux only") == []  # not in the fixture dictionary
    assert matcher.match("") == []
    assert matcher.match("   ") == []


# --- word boundaries -----------------------------------------------------------


def test_no_substring_false_positive() -> None:
    matcher = _fixture_matcher()
    # "pain" must not match inside "painting"; "headache" must not match "headaches".
    assert matcher.match("painting and repainting") == []
    mentions = matcher.match("headaches are not headache")
    assert [m.mention_text for m in mentions] == ["headache"]


def test_lone_angle_bracket_does_not_drop_following_text() -> None:
    # SPL dosage text like "5 < 10 mg" must not swallow the rest of the field.
    index = DictionaryIndex.from_entries([_entry("aspirin", "DRUGBANK:DB00945", category="Drug", source="DRUGBANK")])
    matcher = LexicalMatcher(index)
    mentions = matcher.match("take 5 < 10 mg aspirin daily")
    assert [m.mention_text for m in mentions] == ["aspirin"]
    assert mentions[0].semantic_group == "drug"
    text = "take 5 < 10 mg aspirin daily"
    assert text[mentions[0].mention_start : mentions[0].mention_end] == "aspirin"


# --- greedy longest-phrase-first ----------------------------------------------


def test_longest_phrase_wins_over_inner_term() -> None:
    index = DictionaryIndex.from_entries([_entry("peptic ulcer disease", "MONDO:1"), _entry("ulcer", "MONDO:2")])
    matcher = LexicalMatcher(index)
    mentions = matcher.match("peptic ulcer disease")
    # The longer phrase covers the span; "ulcer" does not re-match inside it.
    assert [m.mention_text for m in mentions] == ["peptic ulcer disease"]


def test_adjacent_terms_both_match() -> None:
    matcher = _fixture_matcher()
    mentions = matcher.match("asthma and headache")
    assert sorted(m.mention_text for m in mentions) == ["asthma", "headache"]


# --- ignore-list ---------------------------------------------------------------


def test_whole_field_ignore_suppresses_record() -> None:
    matcher = _fixture_matcher()
    assert matcher.is_ignored_text("product used for unknown indication")
    # An indication string that IS an ignore term yields no mentions.
    assert matcher.match("product used for unknown indication") == []
    # A non-ignored field still matches (ignore is exact whole-field, not substring).
    assert [m.mention_text for m in matcher.match("pain")]


def test_whole_field_ignore_beats_inner_match() -> None:
    # Suppression wins even when the ignored field contains a matchable dictionary term.
    index = DictionaryIndex.from_entries([_entry("pain", "MONDO:2")])
    matcher = LexicalMatcher(index, ignore_terms=["pain relief"])
    assert matcher.match("pain relief") == []  # whole field ignored despite "pain"
    assert [m.mention_text for m in matcher.match("pain")] == ["pain"]  # not ignored


def test_per_mention_ignore_filters_individual_term() -> None:
    # A dictionary term that is also an ignore term is filtered at the mention level.
    index = DictionaryIndex.from_entries([_entry("prophylaxis", "MONDO:9"), _entry("pain", "MONDO:2")])
    matcher = LexicalMatcher(index)
    assert "prophylaxis" in DEFAULT_IGNORE_TERMS
    mentions = matcher.match("used for prophylaxis of pain")
    assert [m.mention_text for m in mentions] == ["pain"]  # prophylaxis dropped


def test_custom_ignore_terms_override_default() -> None:
    index = DictionaryIndex.from_entries([_entry("pain", "MONDO:2")])
    matcher = LexicalMatcher(index, ignore_terms=["pain"])
    assert matcher.match("pain") == []  # custom ignore list replaces the default


# --- synonyms (opt-in) ---------------------------------------------------------


def test_synonym_fallback_preserves_original_surface_and_offsets() -> None:
    index = DictionaryIndex.from_entries([_entry("heart failure", "MONDO:0004")])
    matcher = LexicalMatcher(index, synonyms=LEGACY_SYNONYMS)
    text = "severe cardiac failure"
    mentions = matcher.match(text)
    assert len(mentions) == 1
    m = mentions[0]
    # Surface form + offsets stay the original words; concept is the synonym target.
    assert m.mention_text == "cardiac failure"
    assert text[m.mention_start : m.mention_end] == "cardiac failure"
    assert m.normalized == "cardiac failure"
    assert m.entry.curie == "MONDO:0004"
    assert m.notes == "synonym:cardiac>heart"
    assert m.score == SYNONYM_SCORE


def test_no_synonym_by_default() -> None:
    index = DictionaryIndex.from_entries([_entry("heart failure", "MONDO:0004")])
    matcher = LexicalMatcher(index)  # no synonyms
    assert matcher.match("severe cardiac failure") == []


# --- determinism / ordering / section context ----------------------------------


def test_match_is_deterministic_across_calls() -> None:
    matcher = _fixture_matcher()
    text = "headache and pain and asthma and hypercholesterolemia"
    first = [(m.mention_start, m.mention_end, m.entry.curie, m.score) for m in matcher.match(text)]
    for _ in range(5):
        again = [(m.mention_start, m.mention_end, m.entry.curie, m.score) for m in matcher.match(text)]
        assert again == first


def test_mentions_sorted_by_offset_then_curie() -> None:
    matcher = _fixture_matcher()
    mentions = matcher.match("pain then headache")
    assert [m.mention_text for m in mentions] == ["pain", "headache"]
    starts = [m.mention_start for m in mentions]
    assert starts == sorted(starts)


def test_multiple_candidates_for_one_span_sorted_by_source() -> None:
    index = DictionaryIndex.from_entries(
        [_entry("fever", "HP:1", category="PhenotypicFeature", source="HPO"), _entry("fever", "MONDO:2", source="MONDO")]
    )
    matcher = LexicalMatcher(index)
    mentions = matcher.match("fever")
    assert [m.entry.source for m in mentions] == ["HPO", "MONDO"]
    # Same span for both candidates.
    assert mentions[0].mention_start == mentions[1].mention_start
    assert mentions[0].mention_end == mentions[1].mention_end


def test_section_context_is_preserved() -> None:
    matcher = _fixture_matcher()
    mentions = matcher.match("asthma", section="indications_and_usage")
    assert mentions[0].section == "indications_and_usage"


def test_offset_invariant_holds_for_every_mention() -> None:
    matcher = _fixture_matcher()
    text = "Hypercholesterolemia, headache, pain; asthma. peptic ulcer disease"
    for m in matcher.match(text):
        assert text[m.mention_start : m.mention_end] == m.mention_text
        assert m.normalized == m.mention_text.lower()
