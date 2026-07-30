"""Edge-case tests for ``dakp_pipeline.ner.backends`` (drive to 100% branch coverage).

Targets the lines the main suite leaves uncovered:

* the empty-needle guard of the private ``_find_word_bounded`` helper;
* the ``MockNERBackend`` skip-an-empty-surface-form branch and the greedy-overlap ``continue``;
* the *successful* lazy-import path of ``GLiNERBackend`` / ``SciSpacyBackend`` (exercised with
  fake ``gliner`` / ``spacy`` modules injected into ``sys.modules`` and a stubbed
  ``ensure_model`` — still no ``[ner]`` extra installed);
* adversarial empty / whitespace / very-short text, all entity types, unknown-backend errors,
  and ``extract_contraindication_diseases`` edge cases.

Everything here passes with the ``[ner]`` extra NOT installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline.ner import backends
from dakp_pipeline.ner.backends import (
    CONTRAINDICATION_DISEASE_TYPES,
    TYPE_CHEMICAL,
    TYPE_DISEASE,
    TYPE_DRUG,
    TYPE_PHENOTYPE,
    DictionaryNERBackend,
    EntitySpan,
    GLiNERBackend,
    MockNERBackend,
    SciSpacyBackend,
    _find_word_bounded,
    canonical_type,
    extract_contraindication_diseases,
    get_backend,
)
from dakp_pipeline.ner.dictionary import DictionaryEntry, DictionaryIndex
from dakp_pipeline.ner.model_cache import ModelRef

# --- _find_word_bounded helper --------------------------------------------------


def test_find_word_bounded_empty_needle_returns_empty() -> None:
    # The empty-needle guard (defensive: callers only pass non-empty normalized terms).
    assert _find_word_bounded("anything at all", "") == []
    assert _find_word_bounded("", "") == []


def test_find_word_bounded_boundaries_and_repeat_occurrences() -> None:
    assert _find_word_bounded("pain pain", "pain") == [0, 5]
    # Not word-bounded inside a longer word.
    assert _find_word_bounded("painting", "pain") == []
    # Boundary at end-of-string and after a space.
    assert _find_word_bounded("a pain", "pain") == [2]


# --- MockNERBackend: empty-surface skip + greedy overlap ------------------------


def test_mock_backend_skips_surface_forms_that_normalize_to_empty() -> None:
    # "!!!" and "" and "<b>" all normalize to "" -> skipped (not indexed), so they never match.
    backend = MockNERBackend({"!!!": "disease", "": "phenotype", "<b>": "disease", "pain": "disease"})
    spans = backend.extract("!!! <b> pain", ["disease", "phenotype"])
    assert [s.text for s in spans] == ["pain"]


def test_mock_backend_greedy_longest_first_skips_overlapping_shorter_term() -> None:
    # "peptic ulcer disease" is matched first (longest); "ulcer" / "pain" inside it are skipped.
    backend = MockNERBackend({"peptic ulcer disease": "disease", "ulcer": "disease", "disease": "disease"})
    text = "peptic ulcer disease"
    spans = backend.extract(text, ["disease"])
    assert [s.text for s in spans] == ["peptic ulcer disease"]
    assert spans[0].start == 0 and spans[0].end == len(text)


def test_mock_backend_non_overlapping_repeat_and_adjacent_terms() -> None:
    backend = MockNERBackend({"pain": "disease", "fever": "phenotype"})
    text = "pain fever pain"
    spans = backend.extract(text, [])  # empty types = no filter
    assert [(s.text, s.type) for s in spans] == [("pain", "disease"), ("fever", "phenotype"), ("pain", "disease")]


def test_mock_backend_all_entity_types_and_custom_score() -> None:
    backend = MockNERBackend(
        {"asthma": TYPE_DISEASE, "rash": TYPE_PHENOTYPE, "aspirin": TYPE_CHEMICAL, "ibuprofen": TYPE_DRUG}, score=0.5
    )
    text = "asthma rash aspirin ibuprofen"
    spans = backend.extract(text, [])
    assert {s.type for s in spans} == {TYPE_DISEASE, TYPE_PHENOTYPE, TYPE_CHEMICAL, TYPE_DRUG}
    assert all(s.score == 0.5 for s in spans)
    # Filtering to a single type keeps only that type.
    assert [s.text for s in backend.extract(text, [TYPE_DRUG])] == ["ibuprofen"]


def test_mock_backend_very_short_and_blank_text() -> None:
    backend = MockNERBackend({"a": "disease", "pain": "disease"})
    assert backend.extract("", []) == []
    assert backend.extract("   \t\n ", []) == []
    # A single-character vocabulary term can match a single-character text.
    assert [s.text for s in backend.extract("a", [TYPE_DISEASE])] == ["a"]


def test_mock_backend_normalizes_case_and_punctuation_but_reports_original_offsets() -> None:
    backend = MockNERBackend({"hepatitis b": "disease"})
    text = "Hepatitis-B!"
    spans = backend.extract(text, [TYPE_DISEASE])
    assert len(spans) == 1
    assert spans[0].text == text[spans[0].start : spans[0].end]


# --- DictionaryNERBackend: synonyms / ignore / empty text -----------------------


def _index(*entries: DictionaryEntry) -> DictionaryIndex:
    return DictionaryIndex.from_entries(list(entries))


def test_dictionary_backend_empty_text_and_no_matches() -> None:
    backend = DictionaryNERBackend(_index(DictionaryEntry("asthma", "MONDO:1", "asthma", "Disease", "MONDO", "asthma")))
    assert backend.extract("", []) == []
    assert backend.extract("   ", []) == []
    assert backend.extract("nothing relevant here", [TYPE_DISEASE]) == []


def test_dictionary_backend_honors_ignore_terms_and_synonyms() -> None:
    index = _index(
        DictionaryEntry("heart failure", "MONDO:1", "heart failure", "Disease", "MONDO", "heart failure"),
        DictionaryEntry("asthma", "MONDO:2", "asthma", "Disease", "MONDO", "asthma"),
    )
    # Synonym 'cardiac' -> 'heart' lets "cardiac failure" resolve the "heart failure" term,
    # while keeping the original surface offsets.
    backend = DictionaryNERBackend(index, synonyms={"cardiac": "heart"})
    text = "cardiac failure"
    spans = backend.extract(text, [TYPE_DISEASE])
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == spans[0].text
    assert spans[0].score < 1.0  # synonym match is lower confidence than a direct match

    # A whole-field ignore term suppresses everything.
    ignored = DictionaryNERBackend(index, ignore_terms=["asthma"])
    assert ignored.extract("asthma", [TYPE_DISEASE]) == []


# --- GLiNERBackend: successful lazy-import path (fake gliner) -------------------


class _FakeGLiNERModel:
    def __init__(self, predictions: list[dict[str, Any]]) -> None:
        self._predictions = predictions
        self.calls: list[tuple[str, list[str], float]] = []

    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.0) -> list[dict[str, Any]]:
        self.calls.append((text, labels, threshold))
        return self._predictions


class _FakeGLiNER:
    loaded_from: list[str] = []
    model = _FakeGLiNERModel([])

    @staticmethod
    def from_pretrained(path: str) -> _FakeGLiNERModel:
        _FakeGLiNER.loaded_from.append(path)
        return _FakeGLiNER.model


def _install_fake_gliner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, predictions: list[dict[str, Any]]) -> None:
    _FakeGLiNER.loaded_from = []
    _FakeGLiNER.model = _FakeGLiNERModel(predictions)
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)

    def _fake_ensure_model(model_id: str, **kwargs: Any) -> ModelRef:
        return ModelRef(model_id=model_id, source="huggingface", path=tmp_path, b3="b3:deadbeef", manifest=tmp_path / "manifest.json")

    monkeypatch.setattr(backends, "ensure_model", _fake_ensure_model)


def test_gliner_backend_loads_and_extracts_with_fake_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    predictions = [
        {"text": "liver disease", "start": 0, "end": 13, "label": "DISEASE", "score": 0.9},
        {"text": "rash", "start": 18, "end": 22, "label": "PhenotypicFeature", "score": 0.8},
        {"text": "aspirin", "start": 27, "end": 34, "label": "CHEMICAL", "score": 0.7},
    ]
    _install_fake_gliner(monkeypatch, tmp_path, predictions)

    backend = GLiNERBackend(model_id="acme/ner", threshold=0.42, cache_dir=tmp_path)
    spans = backend.extract("liver disease, rash, aspirin", [TYPE_DISEASE, TYPE_PHENOTYPE])

    # The chemical span is filtered out; labels are canonicalized; deterministic order.
    assert [(s.text, s.type) for s in spans] == [("liver disease", TYPE_DISEASE), ("rash", TYPE_PHENOTYPE)]
    assert _FakeGLiNER.loaded_from == [str(tmp_path)]
    # Requested types are passed straight through as zero-shot labels, with the threshold.
    _text, labels, threshold = _FakeGLiNER.model.calls[0]
    assert labels == [TYPE_DISEASE, TYPE_PHENOTYPE]
    assert threshold == 0.42

    # A second extract reuses the cached model (from_pretrained called exactly once).
    backend.extract("liver disease", [TYPE_DISEASE])
    assert len(_FakeGLiNER.loaded_from) == 1


def test_gliner_backend_empty_types_fall_back_to_contraindication_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner(monkeypatch, tmp_path, [])
    backend = GLiNERBackend()
    backend.extract("anything", [])  # empty types -> CONTRAINDICATION_DISEASE_TYPES labels
    _text, labels, _threshold = _FakeGLiNER.model.calls[0]
    assert labels == list(CONTRAINDICATION_DISEASE_TYPES)


# --- SciSpacyBackend: successful lazy-import path (fake spacy) ------------------


class _FakeEnt:
    def __init__(self, text: str, start_char: int, end_char: int, label_: str) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label_


class _FakeDoc:
    def __init__(self, ents: list[_FakeEnt]) -> None:
        self.ents = ents


class _FakeNLP:
    def __init__(self, ents: list[_FakeEnt]) -> None:
        self._ents = ents
        self.calls: list[str] = []

    def __call__(self, text: str) -> _FakeDoc:
        self.calls.append(text)
        return _FakeDoc(self._ents)


def test_scispacy_backend_loads_and_extracts_with_fake_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    nlp = _FakeNLP([_FakeEnt("liver disease", 0, 13, "DISEASE"), _FakeEnt("aspirin", 15, 22, "CHEMICAL")])
    loaded: list[str] = []
    module = types.ModuleType("spacy")

    def _load(model_id: str) -> _FakeNLP:
        loaded.append(model_id)
        return nlp

    module.load = _load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", module)

    backend = SciSpacyBackend(model_id="en_ner_bc5cdr_md")
    spans = backend.extract("liver disease, aspirin", [TYPE_DISEASE])

    # BC5CDR labels are canonicalized (DISEASE -> disease); chemical filtered out.
    assert [(s.text, s.type) for s in spans] == [("liver disease", TYPE_DISEASE)]
    assert all(s.score == 1.0 for s in spans)
    assert loaded == ["en_ner_bc5cdr_md"]

    # Second extract reuses the cached nlp (spacy.load called exactly once).
    backend.extract("liver disease", [TYPE_DISEASE])
    assert len(loaded) == 1
    assert len(nlp.calls) == 2


def test_scispacy_backend_empty_types_keeps_all_canonicalized_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    nlp = _FakeNLP([_FakeEnt("liver disease", 0, 13, "DISEASE"), _FakeEnt("aspirin", 15, 22, "CHEMICAL")])
    module = types.ModuleType("spacy")
    module.load = lambda model_id: nlp  # type: ignore[attr-defined,method-assign]
    monkeypatch.setitem(sys.modules, "spacy", module)

    spans = SciSpacyBackend().extract("liver disease, aspirin", [])  # no filter
    assert sorted(s.type for s in spans) == [TYPE_CHEMICAL, TYPE_DISEASE]


# --- factory + high-level helper edge cases -------------------------------------


def test_get_backend_empty_and_whitespace_names() -> None:
    with pytest.raises(ValueError, match="unknown ner_backend"):
        get_backend("")
    with pytest.raises(ValueError, match="unknown ner_backend"):
        get_backend("   ")
    # Whitespace-padded known names are accepted (case-insensitive).
    assert isinstance(get_backend("  GLiNER  "), GLiNERBackend)
    assert isinstance(get_backend(" scispacy "), SciSpacyBackend)


def test_extract_contraindication_diseases_empty_text_and_no_matches() -> None:
    backend = MockNERBackend({"asthma": "disease"})
    assert extract_contraindication_diseases("", backend) == []
    assert extract_contraindication_diseases("no diseases here", backend) == []
    # Only disease/phenotype types are requested: a drug-only vocabulary yields nothing.
    drug_only = MockNERBackend({"aspirin": "drug"})
    assert extract_contraindication_diseases("aspirin", drug_only) == []


def test_canonical_type_unknown_and_whitespace_labels() -> None:
    assert canonical_type("") == ""
    assert canonical_type("   ") == ""
    assert canonical_type("  DISEASES  ") == TYPE_DISEASE
    assert canonical_type("Phenotypic_Feature") == TYPE_PHENOTYPE
    assert canonical_type("SmallMolecule") == TYPE_DRUG


def test_entity_span_is_frozen_and_hashable() -> None:
    span = EntitySpan(text="asthma", start=0, end=6, type=TYPE_DISEASE, score=1.0)
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass rejects assignment
        span.score = 0.5  # type: ignore[misc]
    assert isinstance(hash(span), int)
