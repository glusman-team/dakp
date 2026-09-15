"""Schema tests for the prompt-engineered NER vocabulary and its two-channel routing (US-007).

Why this file exists: the label vocabulary is now a *prompt*, not a list of words. Two things can
silently regress it, and neither shows up as an exception:

1. **Descriptions stop reaching the model.** ``MODEL_LABELS`` is a ``{label: description}``
   mapping handed to gliner2 verbatim; if a refactor flattens it to names, extraction quality
   drops (measured: plain ``["disease", "phenotype"]`` mistyped "pregnancy" as a disease at 0.82
   and emitted "childbearing potential" at 0.40) while every call still succeeds. The stub model
   here captures ``entity_types`` and asserts it equals :data:`MODEL_LABELS`.
2. **An unmodelled qualifier enters the pipeline.** ``MENTION_TYPES`` is a closed vocabulary;
   ``severity_qualifier`` is the concrete counter-example (a plausible label DAKP has no column
   for). It must be dropped at the span adapter, and it must not exist anywhere in ``src/``,
   ``tables/`` or the gold fixture.

Channel routing is tested at both levels: the pure span routers (fast, hermetic) and the full
``extract`` / ``extract_batch`` paths through the fake ``gliner2`` module shared with
``test_ner_edge.py`` (the established injection pattern — extended here, not duplicated).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
from test_ner_edge import _FakeAutoExtractor, _FakeExtractorModel, _install_fake_gliner2, _span

from dakp_pipeline.assertions import object_mentions
from dakp_pipeline.ner.dictionary import (
    MENTION_TYPES,
    OBJECT_TYPES,
    QUALIFIER_TYPES,
    TYPE_ANATOMICAL_ENTITY,
    TYPE_BIOLOGICAL_SEX,
    TYPE_DISEASE,
    TYPE_FREQUENCY_QUALIFIER,
    TYPE_ORGANISM_TAXON,
    TYPE_PHENOTYPE,
    TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS,
    TYPE_TEMPORAL_CONTEXT_QUALIFIER,
    TYPE_TEMPORAL_INTERVAL_QUALIFIER,
    canonical_type,
)
from dakp_pipeline.ner.ner import (
    MODEL_LABEL_NAMES,
    MODEL_LABELS,
    QUALIFIER_ACCEPT_THRESHOLD,
    QUALIFIER_GAZETTEER,
    QUALIFIER_GAZETTEER_NOTES,
    QUALIFIER_NOTES,
    DiseaseNER,
    Mention,
    _ModelSpan,
    _spans_from_result,
)

#: The reviewed schema: model label -> canonical ``Mention.type``, in prompt order.
_EXPECTED_LABEL_TO_TYPE: tuple[tuple[str, str], ...] = (
    ("biolink:Disease", TYPE_DISEASE),
    ("biolink:PhenotypicFeature", TYPE_PHENOTYPE),
    ("biolink:AnatomicalEntity", TYPE_ANATOMICAL_ENTITY),
    ("biolink:BiologicalSex", TYPE_BIOLOGICAL_SEX),
    ("biolink:PopulationOfIndividualOrganisms", TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS),
    ("biolink:OrganismTaxon", TYPE_ORGANISM_TAXON),
    ("frequency_qualifier", TYPE_FREQUENCY_QUALIFIER),
    ("temporal_context_qualifier", TYPE_TEMPORAL_CONTEXT_QUALIFIER),
    ("temporal_interval_qualifier", TYPE_TEMPORAL_INTERVAL_QUALIFIER),
)

#: The reviewed cap on a label description: long enough for exemplars, short enough that nine of
#: them plus the text still fit the checkpoint's prompt budget.
_MAX_DESCRIPTION_CHARS = 220

#: A label DAKP has no column for — the vocabulary filter's counter-example.
_UNMODELLED_QUALIFIER = "severity_qualifier"


def _backend(**kwargs: Any) -> DiseaseNER:
    """A production-mode backend with an empty gazetteer, so only the model channel is under test."""
    kwargs.setdefault("gazetteer", {})
    return DiseaseNER(offline=False, **kwargs)


# --- REQ-SCH-1: the label vocabulary is Biolink category IDs WITH descriptions ----


def test_label_vocabulary_is_the_reviewed_biolink_schema_in_prompt_order() -> None:
    """Labels are Biolink category IDs (not bare words) in the reviewed order, and each one
    canonicalizes onto exactly the ``Mention.type`` the shapers expect.

    Order is the prompt order gliner2 renders, so it is part of the reviewed artifact — a
    reordered mapping is a different prompt and a different measurement.
    """
    assert tuple(MODEL_LABELS) == tuple(label for label, _ in _EXPECTED_LABEL_TO_TYPE)
    assert tuple(MODEL_LABELS) == MODEL_LABEL_NAMES
    assert [(label, canonical_type(label)) for label in MODEL_LABEL_NAMES] == list(_EXPECTED_LABEL_TO_TYPE)
    # Every label lands inside the closed vocabulary, or the adapter would drop all of it.
    assert {canonical_type(label) for label in MODEL_LABEL_NAMES} == set(MENTION_TYPES)
    # ``disease_context_qualifier`` deliberately gets NO label: it shares ``biolink:Disease`` and
    # is derived from the object channel by the contraindication shaper's context logic.
    assert "disease_context_qualifier" not in MODEL_LABELS
    assert canonical_type("disease_context_qualifier") not in set(MENTION_TYPES)


def test_every_label_carries_a_bounded_span_description_not_a_task_instruction() -> None:
    """Descriptions say *what span to draw*, with inline positive exemplars, and stay ≤220 chars.

    gliner2 supports descriptions for entity labels but NOT few-shot input->output pairs
    (``processor._process_entities`` offers only ``["none", "descriptions"]`` modes), so an
    arrow-style example would be rendered as prose and could not work as intended. Imperative
    task instructions ("Extract every disease...") are equally out: they re-describe the task the
    label already names and cost prompt budget.
    """
    for label, description in MODEL_LABELS.items():
        assert isinstance(description, str), label
        assert description, label
        assert description.strip() == description, label
        assert len(description) <= _MAX_DESCRIPTION_CHARS, f"{label} is {len(description)} chars"
        assert "\n" not in description, label  # one prompt line per label
        assert "e.g." in description, f"{label} has no positive exemplars"
        assert "->" not in description, f"{label} looks like a few-shot pair"
        assert "→" not in description, f"{label} looks like a few-shot pair"
        assert not description.lower().startswith(("extract", "find", "identify", "return", "label", "tag")), label


def test_label_material_is_a_deterministic_module_constant() -> None:
    """The mention-cache fingerprint folds the label material, so it must be reproducible.

    A description built from anything non-deterministic (a timestamp, a dict comprehension over a
    set, an env var) would change the fingerprint between runs and silently invalidate every
    cached mention. Construction must also COPY the vocabulary, so a caller mutating the mapping
    it passed in cannot change what an already-built backend requests.
    """
    from dakp_pipeline.ner.mention_cache import config_fingerprint

    assert config_fingerprint(_backend(model_id="acme/x")) == config_fingerprint(_backend(model_id="acme/x"))

    custom = dict(MODEL_LABELS)
    backend = _backend(model_labels=custom)
    custom["biolink:Disease"] = "mutated after construction"
    assert backend._model_labels == MODEL_LABELS
    assert MODEL_LABELS["biolink:Disease"] != "mutated after construction"


def test_both_call_paths_receive_the_described_vocabulary_verbatim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``extract_entities`` (per window) and ``batch_extract_entities`` (worker hot path) must both
    be handed the label->description mapping, not just its keys.

    This is the regression guard for REQ-SCH-1: flattening ``MODEL_LABELS`` to
    ``list(MODEL_LABELS)`` anywhere would keep every test that only checks mention types green
    while quietly dropping the descriptions from the prompt.
    """
    text = "Contraindicated in patients with pulmonary hypertension."
    model = _FakeExtractorModel([_span(text, "pulmonary hypertension", "biolink:Disease", 0.99)])
    _install_fake_gliner2(monkeypatch, tmp_path, [], model=model)
    backend = _backend()

    assert [(m.text, m.type) for m in backend.extract(text)] == [("pulmonary hypertension", TYPE_DISEASE)]
    assert [(m.text, m.type) for m in backend.extract_batch([text])[0]] == [("pulmonary hypertension", TYPE_DISEASE)]

    assert len(model.calls) == 2  # one per-text call and one batched call
    for _window, entity_types, _threshold in model.calls:
        assert entity_types == MODEL_LABELS
        assert isinstance(entity_types, dict)
        assert all(entity_types.values())


# --- REQ-SCH-2: canonical type mapping --------------------------------------------


def test_canonical_type_folds_every_label_dialect_onto_one_vocabulary() -> None:
    """Model labels, bare Biolink names and the legacy lowercase dialect all canonicalize to the
    same value — otherwise one checkpoint swap would silently change ``Mention.type`` downstream."""
    assert canonical_type("biolink:Disease") == canonical_type("Disease") == canonical_type("disease") == TYPE_DISEASE
    assert canonical_type("biolink:PhenotypicFeature") == canonical_type("PhenotypicFeature") == TYPE_PHENOTYPE
    assert canonical_type("phenotype") == canonical_type("phenotypic_feature") == canonical_type("phenotypes") == TYPE_PHENOTYPE
    assert canonical_type("biolink:AnatomicalEntity") == TYPE_ANATOMICAL_ENTITY
    assert canonical_type("biolink:BiologicalSex") == TYPE_BIOLOGICAL_SEX
    assert canonical_type("biolink:PopulationOfIndividualOrganisms") == TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS
    assert canonical_type("biolink:OrganismTaxon") == TYPE_ORGANISM_TAXON
    # Field-named qualifiers map to themselves (they have no Biolink category of that name).
    for qualifier in (TYPE_FREQUENCY_QUALIFIER, TYPE_TEMPORAL_CONTEXT_QUALIFIER, TYPE_TEMPORAL_INTERVAL_QUALIFIER):
        assert canonical_type(qualifier) == qualifier
    assert canonical_type("  biolink:disease  ") == TYPE_DISEASE  # stripped + case-insensitive


def test_canonical_type_leaves_unknown_labels_outside_the_vocabulary() -> None:
    """Unknown labels keep the lowercased fallback (unchanged behavior) — which is exactly what
    puts them outside ``MENTION_TYPES`` so the adapter drops them instead of inventing a type."""
    assert canonical_type(_UNMODELLED_QUALIFIER) == _UNMODELLED_QUALIFIER
    assert canonical_type(_UNMODELLED_QUALIFIER) not in set(MENTION_TYPES)
    assert canonical_type("biolink:SmallMolecule") == "smallmolecule" not in set(MENTION_TYPES)
    assert canonical_type("CHEMICAL") == "chemical"
    assert canonical_type("") == ""


def test_closed_sex_qualifier_gazetteer_is_unambiguous_and_separate_from_objects() -> None:
    """Common sex terms should be deterministic qualifiers without expanding object matching.

    The lexicon deliberately excludes broad demographic phrases: a conservative, reviewable
    closed list prevents ``women`` from becoming an assertion object while avoiding a silent
    population-ontology policy in the NER layer.
    """
    assert QUALIFIER_GAZETTEER == {
        "female": TYPE_BIOLOGICAL_SEX,
        "females": TYPE_BIOLOGICAL_SEX,
        "male": TYPE_BIOLOGICAL_SEX,
        "males": TYPE_BIOLOGICAL_SEX,
        "men": TYPE_BIOLOGICAL_SEX,
        "woman": TYPE_BIOLOGICAL_SEX,
        "women": TYPE_BIOLOGICAL_SEX,
    }
    assert set(QUALIFIER_GAZETTEER.values()).isdisjoint(OBJECT_TYPES)


def test_lexical_sex_qualifiers_work_offline_and_preserve_word_boundaries() -> None:
    """Offline mode must emit sex qualifiers, but ``male`` must never match inside ``female``.

    This protects the review-requested easy deterministic path and the matcher boundary invariant
    that prevents one sex term from duplicating or corrupting another.
    """
    backend = DiseaseNER(offline=True, gazetteer={})
    mentions = backend.extract("Female patients and men; male, women, females, males, woman.")
    assert [(m.text, m.type, m.notes) for m in mentions] == [
        ("Female", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("men", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("male", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("women", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("females", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("males", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("woman", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
    ]


def test_lexical_qualifier_wins_only_exact_model_duplicate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A known sex surface beats an exact model duplicate but never suppresses object spans.

    The qualifier channels are independent: the same lexical hit should not delete an overlapping
    disease, and a differently bounded model qualifier must still survive for caller review.
    """
    text = "Women with asthma"
    model = _FakeExtractorModel(
        [
            _span(text, "Women", "biolink:BiologicalSex", 0.99),
            _span(text, "Women with", "biolink:PopulationOfIndividualOrganisms", 0.8),
            _span(text, "asthma", "biolink:Disease", 0.99),
        ]
    )
    _install_fake_gliner2(monkeypatch, tmp_path, [], model=model)
    mentions = _backend().extract(text)
    assert [(m.text, m.type, m.notes) for m in mentions] == [
        ("Women", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("Women with", TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS, QUALIFIER_NOTES),
        ("asthma", TYPE_DISEASE, "gliner"),
    ]


def test_type_vocabulary_splits_into_two_disjoint_channels() -> None:
    """The channel split is the routing rule, so the two tuples must stay disjoint and complete."""
    assert OBJECT_TYPES == (TYPE_DISEASE, TYPE_PHENOTYPE)
    assert len(QUALIFIER_TYPES) == 7
    assert MENTION_TYPES == OBJECT_TYPES + QUALIFIER_TYPES
    assert set(OBJECT_TYPES).isdisjoint(QUALIFIER_TYPES)
    assert len(set(MENTION_TYPES)) == len(MENTION_TYPES)  # no duplicate canonical types


# --- REQ-SCH-2: the vocabulary filter (negative tests) ------------------------------


def test_spans_from_result_drops_every_label_outside_the_vocabulary() -> None:
    """A checkpoint may emit anything; DAKP emits only ``MENTION_TYPES``. Both an unmodelled
    qualifier (``severity_qualifier``) and an unrelated category (``CHEMICAL``) are dropped here,
    before any channel routing, so they cannot reach a shaper through either path."""
    window = "severe asthma in rats"
    result = {
        "entities": {
            "biolink:Disease": [{"text": "asthma", "confidence": 0.9, "start": 7, "end": 13}],
            "biolink:OrganismTaxon": [{"text": "rats", "confidence": 0.8, "start": 17, "end": 21}],
            _UNMODELLED_QUALIFIER: [{"text": "severe", "confidence": 0.99, "start": 0, "end": 6}],
            "CHEMICAL": [{"text": "asthma", "confidence": 0.99, "start": 7, "end": 13}],
        }
    }
    spans = _spans_from_result(result, window)
    assert [(span.type, span.start, span.end) for span in spans] == [("Disease", 7, 13), ("OrganismTaxon", 17, 21)]
    assert _UNMODELLED_QUALIFIER not in {span.type for span in spans}


def test_unmodelled_qualifier_never_becomes_a_mention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """NEGATIVE end-to-end proof: a high-confidence ``severity_qualifier`` span is dropped on both
    the per-text and the batched path, while the object span in the same result survives.

    Without the ``MENTION_TYPES`` filter this would emit ``Mention(type="severity_qualifier")``,
    which no DAKP column models — and a shaper would then either drop it silently or, worse,
    assert it as an object.
    """
    text = "Contraindicated in severe asthma."
    predictions = [_span(text, "severe", _UNMODELLED_QUALIFIER, 0.99), _span(text, "asthma", "biolink:Disease", 0.99)]
    model = _FakeExtractorModel(predictions)
    _install_fake_gliner2(monkeypatch, tmp_path, [], model=model)
    backend = _backend()

    single = backend.extract(text)
    batched = backend.extract_batch([text])[0]

    assert [(m.text, m.type, m.notes) for m in single] == [("asthma", TYPE_DISEASE, "gliner")]
    assert batched == single
    assert all(_UNMODELLED_QUALIFIER not in (m.type, m.notes) for m in [*single, *batched])
    assert {m.type for m in [*single, *batched]} <= set(MENTION_TYPES)


def test_shipped_code_tables_and_gold_fixture_never_mention_the_unmodelled_qualifier() -> None:
    """The closed vocabulary is only real if nothing shipped re-introduces the dropped label.

    A stray constant, table column or gold annotation naming ``severity_qualifier`` would mean
    DAKP claims a type it cannot produce (or produces one it never reviewed), so the whole tree
    that ships — ``src/``, ``tables/`` and the gold fixture — is grepped. Test files are excluded
    on purpose: they are where the drop is asserted.
    """
    root = Path(__file__).resolve().parents[2]
    paths = [
        path
        for path in itertools.chain((root / "src").rglob("*"), (root / "tables").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    paths.append(root / "tests" / "eval" / "ner_gold.json")
    assert paths, "no shipped files found — the glob is broken, this test would pass vacuously"
    offenders = [str(path.relative_to(root)) for path in paths if _UNMODELLED_QUALIFIER in path.read_text("utf-8", errors="ignore")]
    assert offenders == []


# --- REQ-SCH-3: one call, two channels ---------------------------------------------


def test_channel_spans_route_by_canonical_type_and_remap_offsets() -> None:
    """The router is the whole two-channel design: object types go to the merge, everything else in
    the vocabulary goes to the qualifier channel, and window-relative offsets become full-text ones."""
    text = "asthma in women"
    backend = _backend()
    spans = [
        _ModelSpan(start=0, end=6, type=TYPE_DISEASE, score=0.9),
        _ModelSpan(start=10, end=15, type=TYPE_BIOLOGICAL_SEX, score=0.9),
        _ModelSpan(start=10, end=15, type=TYPE_FREQUENCY_QUALIFIER, score=0.9),
    ]

    objects, qualifiers = backend._channel_spans(text, spans, window_start=100)

    assert [(span.type, span.start, span.end) for span in objects] == [(TYPE_DISEASE, 100, 106)]
    assert [(span.type, span.start, span.end) for span in qualifiers] == [(TYPE_BIOLOGICAL_SEX, 110, 115), (TYPE_FREQUENCY_QUALIFIER, 110, 115)]
    assert backend._channel_spans(text, [], 0) == ([], [])


def test_one_call_emits_both_channels_with_distinct_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ONE inference call carries every label, and the two channels are distinguishable afterwards:
    ``gliner`` / ``gliner:extends`` / ``exact`` are object provenance, ``gliner:qualifier`` is the
    qualifier channel. Cross-label overlap is allowed, so a qualifier coexists with the maximal
    object span over the same words (de-overlapping is per label, by design)."""
    text = "Contraindicated in patients with pulmonary hypertension in women."
    predictions = [_span(text, "pulmonary hypertension", "biolink:Disease", 0.99), _span(text, "women", "biolink:BiologicalSex", 1.0)]
    _install_fake_gliner2(monkeypatch, tmp_path, predictions)

    mentions = _backend(gazetteer={"hypertension": "disease"}).extract(text)

    assert [(m.text, m.type, m.notes) for m in mentions] == [
        ("pulmonary hypertension", TYPE_DISEASE, "gliner:extends"),
        ("women", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
    ]
    assert mentions == sorted(mentions, key=lambda m: (m.start, m.end, m.type, m.text))


def test_qualifier_channel_keeps_the_head_that_object_hedge_trimming_would_strip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The measured rationale for routing instead of merging: ``_trim_hedges`` turns
    ``history of hypertension`` into ``hypertension``, destroying the temporal qualifier, while the
    object channel still trims its own span. Same words, one call, two correct answers."""
    text = "Contraindicated in patients with a history of hypertension."
    predictions = [
        _span(text, "history of hypertension", "temporal_context_qualifier", 0.7),
        _span(text, "history of hypertension", "biolink:Disease", 0.9),
    ]
    _install_fake_gliner2(monkeypatch, tmp_path, predictions)

    mentions = _backend().extract(text)

    assert {(m.text, m.type, m.notes) for m in mentions} == {
        ("history of hypertension", TYPE_TEMPORAL_CONTEXT_QUALIFIER, QUALIFIER_NOTES),
        ("hypertension", TYPE_DISEASE, "gliner"),
    }


def test_population_qualifier_survives_the_object_population_filter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``_POPULATION_PHRASES`` holds exactly the surfaces the population/sex labels exist to
    extract, so it must stay object-channel-only: as a qualifier the phrase is a mention, as an
    object span it is still the false positive it was always dropped for."""
    text = "Contraindicated in women of childbearing potential."
    phrase = "women of childbearing potential"

    _install_fake_gliner2(monkeypatch, tmp_path, [_span(text, phrase, "biolink:PopulationOfIndividualOrganisms", 0.86)])
    qualifier = _backend().extract(text)
    assert [(m.text, m.type, m.notes) for m in qualifier] == [
        ("women", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        (phrase, TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS, QUALIFIER_NOTES),
    ]

    _install_fake_gliner2(monkeypatch, tmp_path, [_span(text, phrase, "biolink:PhenotypicFeature", 0.86)])
    # Object channel only: population descriptors never become assertion objects. The newly added
    # deterministic sex qualifier is a separate channel and is explicitly retained alongside.
    assert [m for m in _backend().extract(text) if m.type in OBJECT_TYPES] == []
    assert [m.text for m in _backend().extract(text) if m.type not in OBJECT_TYPES] == ["women"]


def test_qualifier_floor_abstains_below_and_accepts_at_the_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Qualifiers narrow an existing true object edge, so they are asserted at their own higher
    floor (:data:`QUALIFIER_ACCEPT_THRESHOLD`) — not at the object acceptance profile, and not at
    the generation floor that produced them."""
    text = "Take one tablet once daily."
    below = QUALIFIER_ACCEPT_THRESHOLD - 0.01

    _install_fake_gliner2(monkeypatch, tmp_path, [_span(text, "once daily", "frequency_qualifier", below)])
    assert _backend().extract(text) == []

    _install_fake_gliner2(monkeypatch, tmp_path, [_span(text, "once daily", "frequency_qualifier", QUALIFIER_ACCEPT_THRESHOLD)])
    at_floor = _backend().extract(text)
    assert [(m.text, m.type, m.score) for m in at_floor] == [("once daily", TYPE_FREQUENCY_QUALIFIER, QUALIFIER_ACCEPT_THRESHOLD)]

    # A precision-first object profile does not raise the qualifier floor with it.
    _install_fake_gliner2(monkeypatch, tmp_path, [_span(text, "once daily", "frequency_qualifier", QUALIFIER_ACCEPT_THRESHOLD)])
    strict = DiseaseNER.for_indications(offline=False, gazetteer={})
    assert [(m.text, m.notes) for m in strict.extract(text)] == [("once daily", QUALIFIER_NOTES)]


def test_qualifier_mentions_dedupe_on_start_end_type_keeping_the_best_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """gliner2 can return the same span under one label twice; the qualifier channel has no
    de-overlap pass, so exact duplicates collapse deterministically to the highest score while a
    genuinely different type over the same words is kept."""
    text = "Take one tablet once daily."
    predictions = [
        _span(text, "once daily", "frequency_qualifier", 0.7),
        _span(text, "once daily", "frequency_qualifier", 0.9),
        _span(text, "once daily", "temporal_interval_qualifier", 0.8),
    ]
    _install_fake_gliner2(monkeypatch, tmp_path, predictions)

    mentions = _backend().extract(text)

    assert [(m.text, m.type, m.score) for m in mentions] == [
        ("once daily", TYPE_FREQUENCY_QUALIFIER, 0.9),
        ("once daily", TYPE_TEMPORAL_INTERVAL_QUALIFIER, 0.8),
    ]
    assert all(m.normalized == "once daily" for m in mentions)


def test_extract_batch_routes_qualifiers_exactly_like_single_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The worker hot path batches every window of every text into ONE call; per-text channel
    routing must be identical to the per-text path, or multi-GPU mining would produce different
    assertions than a single-text run."""
    text = "Contraindicated in severe asthma in women."
    predictions = [
        _span(text, "asthma", "biolink:Disease", 0.95),
        _span(text, "women", "biolink:PopulationOfIndividualOrganisms", 0.9),
        _span(text, "severe", _UNMODELLED_QUALIFIER, 0.99),
    ]
    model = _FakeExtractorModel(predictions)
    _install_fake_gliner2(monkeypatch, tmp_path, [], model=model)
    backend = _backend()

    batched = backend.extract_batch([text, "", "   ", text])
    single = backend.extract(text)

    assert batched[1] == []
    assert batched[2] == []
    assert batched[0] == batched[3] == single
    assert [(m.text, m.type, m.notes) for m in single] == [
        ("asthma", TYPE_DISEASE, "gliner"),
        ("women", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("women", TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS, QUALIFIER_NOTES),
    ]
    assert all(call[1] == MODEL_LABELS for call in model.calls)


def test_offline_mode_emits_closed_qualifier_lexicon_but_no_model_qualifiers() -> None:
    """Offline extraction retains only reviewed lexical qualifiers, never model-only fields.

    This makes unambiguous sex wording available without making temporal/frequency extraction
    non-deterministic or promoting any qualifier into an assertion object.
    """
    mentions = DiseaseNER().extract("Contraindicated in women of childbearing potential with asthma, once daily.")
    assert [(m.text, m.type, m.notes) for m in mentions] == [
        ("women", TYPE_BIOLOGICAL_SEX, QUALIFIER_GAZETTEER_NOTES),
        ("asthma", TYPE_DISEASE, "exact"),
    ]


# --- REQ-OP-10: shapers never treat a qualifier as an object ------------------------


def test_object_mentions_narrows_mixed_channel_output_to_objects() -> None:
    """``extract`` now returns mixed channels, so every shaper narrows through this filter; a
    qualifier asserted as an object would emit nonsense edges such as ``contraindicated_in women``."""
    disease = Mention(text="asthma", start=0, end=6, type=TYPE_DISEASE, score=1.0, notes="exact")
    phenotype = Mention(text="headache", start=7, end=15, type=TYPE_PHENOTYPE, score=1.0, notes="gliner")
    population = Mention(text="women", start=16, end=21, type=TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS, score=1.0, notes=QUALIFIER_NOTES)
    sex = Mention(text="female", start=22, end=28, type=TYPE_BIOLOGICAL_SEX, score=1.0, notes=QUALIFIER_NOTES)
    anatomical = Mention(text="liver", start=29, end=34, type=TYPE_ANATOMICAL_ENTITY, score=1.0, notes=QUALIFIER_NOTES)
    taxon = Mention(text="rats", start=35, end=39, type=TYPE_ORGANISM_TAXON, score=1.0, notes=QUALIFIER_NOTES)
    frequency = Mention(text="once daily", start=40, end=50, type=TYPE_FREQUENCY_QUALIFIER, score=1.0, notes=QUALIFIER_NOTES)
    temporal = Mention(text="history of", start=51, end=61, type=TYPE_TEMPORAL_CONTEXT_QUALIFIER, score=1.0, notes=QUALIFIER_NOTES)
    interval = Mention(text="for 14 days", start=62, end=73, type=TYPE_TEMPORAL_INTERVAL_QUALIFIER, score=1.0, notes=QUALIFIER_NOTES)

    mixed = [disease, population, phenotype, sex, anatomical, taxon, frequency, temporal, interval]

    assert object_mentions(mixed) == [disease, phenotype]  # order preserved, qualifiers dropped
    assert object_mentions([]) == []
    assert object_mentions([population]) == []


def test_stub_model_returns_category_id_result_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Guard on the test double itself: the fake must mirror gliner2's real result shape with
    category-ID keys, or every routing test above would be asserting against a dialect the real
    checkpoint never emits."""
    text = "asthma"
    _install_fake_gliner2(monkeypatch, tmp_path, [{"start": 0, "end": 6, "label": "biolink:Disease", "score": 0.9}])
    result = _FakeAutoExtractor.model.extract_entities(text, MODEL_LABELS)
    assert list(result["entities"]) == ["biolink:Disease"]
    assert result["entities"]["biolink:Disease"][0] == {"text": "asthma", "confidence": 0.9, "start": 0, "end": 6}
