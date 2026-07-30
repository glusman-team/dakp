"""Unit tests for the deterministic lexical mention matcher (``ner/lexical.py``).

The matcher is a gazetteer span-detector: it locates disease/phenotype mention **spans** in
source text and emits a :class:`Mention` that is a text span + entity type ONLY (no ontology
CURIE). Covers: true offsets into the original text (``mention.text == text[start:end]``),
word-boundary matching (no substring false positives), greedy longest-phrase-first matching,
repeat non-overlapping occurrences, whole-field + per-mention ignore suppression, opt-in
synonym fallback with preserved surface offsets, deterministic scoring/ordering, and section
passthrough.
"""

from __future__ import annotations

from dakp_pipeline.ner.dictionary import Gazetteer
from dakp_pipeline.ner.lexical import DEFAULT_IGNORE_TERMS, DIRECT_SCORE, LEGACY_SYNONYMS, SYNONYM_SCORE, LexicalMatcher


def _matcher(terms: dict[str, str], **kwargs: object) -> LexicalMatcher:
    return LexicalMatcher(Gazetteer(terms), **kwargs)  # type: ignore[arg-type]


# --- span offsets --------------------------------------------------------------


def test_match_finds_mention_with_true_offsets() -> None:
    matcher = _matcher({"hypercholesterolemia": "disease"})
    text = "Examplestatin is indicated for the treatment of hypercholesterolemia in adults."
    mentions = matcher.match(text)
    assert [m.text for m in mentions] == ["hypercholesterolemia"]
    m = mentions[0]
    # The half-open offsets slice back to the exact surface form.
    assert text[m.start : m.end] == m.text == "hypercholesterolemia"
    assert m.type == "disease"
    assert m.normalized == "hypercholesterolemia"
    assert m.score == DIRECT_SCORE
    assert m.notes == "exact"
    assert m.section == ""


def test_match_offsets_survive_mixed_case_and_punctuation() -> None:
    matcher = _matcher({"headache": "phenotype", "pain": "phenotype"})
    text = "Relief of HEADACHE, and mild-to-moderate Pain."
    mentions = matcher.match(text)
    by_text = {m.text: m for m in mentions}
    assert set(by_text) == {"HEADACHE", "Pain"}
    for m in mentions:
        assert text[m.start : m.end] == m.text
    # Original case is preserved on the surface; ``normalized`` is the lowercased key.
    assert by_text["Pain"].normalized == "pain"
    assert by_text["HEADACHE"].normalized == "headache"


def test_multiword_phrase_offsets_span_the_full_surface() -> None:
    matcher = _matcher({"peptic ulcer disease": "disease"})
    text = "History of peptic ulcer disease."
    mentions = matcher.match(text)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.text == "peptic ulcer disease"
    assert text[m.start : m.end] == "peptic ulcer disease"
    assert m.type == "disease"


def test_match_returns_empty_for_unknown_or_blank_text() -> None:
    matcher = _matcher({"asthma": "disease"})
    assert matcher.match("gastroesophageal reflux only") == []  # term not in the gazetteer
    assert matcher.match("") == []
    assert matcher.match("   ") == []


# --- word boundaries -----------------------------------------------------------


def test_no_substring_false_positive() -> None:
    matcher = _matcher({"pain": "phenotype", "headache": "phenotype"})
    # "pain" must NOT match inside "painting"/"repainting".
    assert matcher.match("painting and repainting") == []
    # "headache" must NOT match inside "headaches".
    mentions = matcher.match("headaches are not headache")
    assert [m.text for m in mentions] == ["headache"]


# --- greedy longest-phrase-first ----------------------------------------------


def test_longest_phrase_wins_over_inner_term() -> None:
    matcher = _matcher({"peptic ulcer disease": "disease", "ulcer": "disease"})
    mentions = matcher.match("peptic ulcer disease")
    # The longer phrase covers the span; "ulcer" does not re-match inside it.
    assert [m.text for m in mentions] == ["peptic ulcer disease"]


def test_adjacent_terms_both_match() -> None:
    matcher = _matcher({"asthma": "disease", "headache": "phenotype"})
    mentions = matcher.match("asthma and headache")
    assert sorted(m.text for m in mentions) == ["asthma", "headache"]


def test_shorter_term_before_longer_phrase_both_match() -> None:
    # A shorter term occurring BEFORE an already-covered longer phrase is not an overlap.
    matcher = _matcher({"peptic ulcer disease": "disease", "pain": "phenotype"})
    mentions = matcher.match("pain and peptic ulcer disease")
    assert [(m.text, m.type) for m in mentions] == [("pain", "phenotype"), ("peptic ulcer disease", "disease")]


def test_repeat_non_overlapping_occurrences_all_match() -> None:
    matcher = _matcher({"pain": "phenotype"})
    mentions = matcher.match("pain and pain")
    assert [(m.text, m.start, m.end) for m in mentions] == [("pain", 0, 4), ("pain", 9, 13)]


# --- ignore-list ---------------------------------------------------------------


def test_whole_field_default_ignore_suppresses_record() -> None:
    matcher = _matcher({"pain": "phenotype"})
    assert "product used for unknown indication" in DEFAULT_IGNORE_TERMS
    assert matcher.is_ignored_text("product used for unknown indication")
    # A whole indication string in the legacy ignore set yields no mentions.
    assert matcher.match("product used for unknown indication") == []
    # A non-ignored field still matches (ignore is exact whole-field, not substring).
    assert [m.text for m in matcher.match("pain")] == ["pain"]


def test_whole_field_custom_ignore_terms_replace_defaults() -> None:
    matcher = _matcher({"pain": "phenotype"}, ignore_terms=["pain relief"])
    # Suppression wins even when the ignored field contains a matchable term.
    assert matcher.match("pain relief") == []
    assert [m.text for m in matcher.match("pain")] == ["pain"]
    # A custom ignore list REPLACES the defaults.
    assert not matcher.is_ignored_text("off label use")


def test_per_mention_ignore_filters_individual_term() -> None:
    # "prophylaxis" is a default ignore term; as an individual mention inside a larger field
    # it is dropped while a real disease mention survives.
    matcher = _matcher({"prophylaxis": "disease", "pain": "phenotype"})
    assert "prophylaxis" in DEFAULT_IGNORE_TERMS
    mentions = matcher.match("used for prophylaxis of pain")
    assert [m.text for m in mentions] == ["pain"]


# --- synonyms (opt-in) ---------------------------------------------------------


def test_synonym_fallback_preserves_original_surface_and_offsets() -> None:
    matcher = _matcher({"heart failure": "disease"}, synonyms=LEGACY_SYNONYMS)
    text = "severe cardiac failure"
    mentions = matcher.match(text)
    assert len(mentions) == 1
    m = mentions[0]
    # Surface form + offsets stay the ORIGINAL words; the concept is the synonym target.
    assert m.text == "cardiac failure"
    assert text[m.start : m.end] == "cardiac failure"
    assert m.normalized == "cardiac failure"
    assert m.notes.startswith("synonym:")
    assert "cardiac>heart" in m.notes
    assert m.score == SYNONYM_SCORE
    assert m.score < DIRECT_SCORE


def test_no_synonym_by_default_and_exact_notes() -> None:
    no_syn = _matcher({"heart failure": "disease"})  # no synonyms
    assert no_syn.match("severe cardiac failure") == []
    # A direct (non-synonym) match records exact notes and the full direct score.
    direct = _matcher({"asthma": "disease"}).match("asthma")
    assert direct[0].notes == "exact"
    assert direct[0].score == DIRECT_SCORE


# --- determinism / ordering / section context ----------------------------------


def test_section_context_is_preserved() -> None:
    matcher = _matcher({"asthma": "disease"})
    mentions = matcher.match("asthma", section="indications_and_usage")
    assert mentions[0].section == "indications_and_usage"


def test_match_is_deterministic_across_calls() -> None:
    matcher = _matcher({"asthma": "disease", "headache": "phenotype", "pain": "phenotype"})
    text = "headache and pain and asthma"
    first = [(m.start, m.end, m.type, m.text, m.score) for m in matcher.match(text)]
    for _ in range(5):
        again = [(m.start, m.end, m.type, m.text, m.score) for m in matcher.match(text)]
        assert again == first


def test_mentions_sorted_by_offset() -> None:
    matcher = _matcher({"asthma": "disease", "headache": "phenotype", "pain": "phenotype"})
    mentions = matcher.match("pain then headache then asthma")
    keys = [(m.start, m.end, m.type, m.text) for m in mentions]
    assert keys == sorted(keys)
    assert [m.text for m in mentions] == ["pain", "headache", "asthma"]


def test_offset_invariant_holds_for_every_mention() -> None:
    matcher = _matcher({"headache": "phenotype", "pain": "phenotype", "asthma": "disease", "peptic ulcer disease": "disease"})
    text = "Headache, pain; asthma. peptic ulcer disease"
    mentions = matcher.match(text)
    assert len(mentions) == 4
    for m in mentions:
        assert text[m.start : m.end] == m.text
        assert m.normalized == m.text.lower()
