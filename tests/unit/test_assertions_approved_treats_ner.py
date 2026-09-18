"""Tests for the NER channels in ``dakp_pipeline.assertions.approved_treats``.

Covers: the mention-based corroboration channel (rule 4) — a section NER mention word-contained
in the candidate corroborates it where dictionary + verbatim miss — the DailyMed fallback
mention candidates, the once-per-section mining (sequential + multi-GPU dispatch), and the
shaper's injected-vs-default NER resolution.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dakp_pipeline.assertions import approved_treats
from dakp_pipeline.assertions.approved_treats import (
    ApprovedTreatsShaper,
    _candidate_mention,
    _doc_key,
    _indication_observations,
    _mine_indication_mentions,
    _sentence_local,
    build_approved_treats_rows,
)
from dakp_pipeline.assertions.evidence import DailyMedEvidence
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.ner import ner as ner_module
from dakp_pipeline.ner.model_cache import ModelRef
from dakp_pipeline.ner.ner import DiseaseNER, Mention


def _supported_evidence(section_text: str) -> DailyMedEvidence:
    """NDA 12345 approved on SET-A with one indication section."""
    return DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}},
        approval_display={"12345": "012345"},
        set_ingredient={"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF")},
        active_ingredients_by_set={"SET-A": [("Examplestatin", "UNII:QFX8B1R4QF")]},
        indication_docs={"SET-A": [("SET-A#34067-9", section_text)]},
    )


def _cases(indication: str) -> pl.DataFrame:
    return pl.DataFrame({"nda": ["012345"], "indication": [indication], "drugname": ["Examplestatin"], "ingredient": ["Examplestatin"]})


_MAPPING = {"12345": {"EXAMPLESTATIN"}}


# --- rule 4: the NER mention channel --------------------------------------------


def test_ner_mention_contained_in_candidate_corroborates() -> None:
    # The label names the general condition ('breast cancer'); the FAERS candidate is the more
    # specific report ('hormone receptor positive breast cancer'). Dictionary and verbatim miss;
    # the NER mention (word-contained IN the candidate) corroborates.
    ev = _supported_evidence("Examplestatin is indicated for breast cancer.")
    ner = DiseaseNER(gazetteer={"breast cancer": "disease"})
    rows = build_approved_treats_rows(_cases("Hormone receptor positive breast cancer"), ev, _MAPPING, {}, ner=ner)
    assert [row["object_text"] for row in rows] == ["Hormone receptor positive breast cancer"]


def test_ner_mention_equal_to_candidate_corroborates() -> None:
    # Same normalized text on both sides (mention == candidate) corroborates.
    ev = _supported_evidence("Examplestatin is indicated for diabetic ketoacidosis.")
    ner = DiseaseNER(gazetteer={"diabetic ketoacidosis": "disease"})
    rows = build_approved_treats_rows(_cases("diabetic ketoacidosis"), ev, _MAPPING, {}, ner=ner)
    assert [row["object_text"] for row in rows] == ["diabetic ketoacidosis"]


def test_unrelated_mention_does_not_corroborate() -> None:
    # The section's mentions are unrelated to the candidate -> still dropped.
    ev = _supported_evidence("Examplestatin is indicated for breast cancer.")
    ner = DiseaseNER(gazetteer={"breast cancer": "disease"})
    assert build_approved_treats_rows(_cases("diabetic ketoacidosis"), ev, _MAPPING, {}, ner=ner) == []


def test_mention_channel_off_without_ner() -> None:
    # No backend -> lexical-only behavior (the historical direct-call surface).
    ev = _supported_evidence("Examplestatin is indicated for breast cancer.")
    assert build_approved_treats_rows(_cases("Hormone receptor positive breast cancer"), ev, _MAPPING, {}) == []


# --- DailyMed fallback: NER mention candidates ------------------------------------


def test_observations_skip_documents_and_sentences_without_candidate() -> None:
    """Observation mining must ignore unsupported documents and sentences rather than inventing hosts."""
    from dakp_pipeline.assertions.approved_treats import _indication_observations

    evidence = DailyMedEvidence(indication_docs={"SET-A": [("DOC-A", "other condition.")]})
    candidate = {"object_text": "asthma", "object_category": "Disease"}
    assert _indication_observations(evidence, ["SET-A"], candidate, {}, None) == []


def test_sentence_and_candidate_helpers_abstain_on_empty_or_missing_text() -> None:
    """Blank and absent candidate surfaces must not manufacture lexical hosts or spans."""
    from dakp_pipeline.assertions.approved_treats import _candidate_mention, _sentence_spans

    assert _sentence_spans("  .\n\n  asthma.  ") == [(2, 3, "."), (7, 14, "asthma.")]
    assert _sentence_spans("  ") == []
    assert _sentence_spans("  first.\n\n  second  ") == [(2, 8, "first."), (12, 18, "second")]
    assert _sentence_spans("\n\n asthma.") == [(3, 10, "asthma.")]
    assert _candidate_mention("migraine", {"object_text": "migraine", "object_category": "Disease"}, 0) is not None
    assert _candidate_mention("migraine", {"object_text": "mi graine", "object_category": "Disease"}, 0) is None
    assert _candidate_mention("No condition here", {"object_text": "", "object_category": "Disease"}, 0) is None
    assert _candidate_mention("No condition here", {"object_text": "migraine", "object_category": "Disease"}, 0) is None


def test_candidate_fallback_preserves_original_offsets_and_qualifier_attachment() -> None:
    sentence = "Examplestatin is indicated for heart--   failure in women."
    candidate = {"object_text": "heart failure", "object_category": "Disease"}
    mention = _candidate_mention(sentence, candidate, 0)
    assert mention is not None
    assert mention.text == "heart--   failure"
    assert mention.text == sentence[mention.start : mention.end]
    qualifier_start = sentence.index("women")
    observations = _indication_observations(
        DailyMedEvidence(indication_docs={"SET-A": [("DOC-A", sentence)]}),
        ["SET-A"],
        candidate,
        {},
        {("SET-A", "DOC-A"): [Mention("women", qualifier_start, qualifier_start + 5, "BiologicalSex", 0.9)]},
    )
    assert observations[0]["qualifiers"] == {"sex_text": "women"}


def test_sentence_local_clips_a_mention_that_straddles_the_sentence() -> None:
    """A span crossing a sentence boundary is clipped to the part inside, surface included.

    GLiNER predicts over token windows that tile the whole section, so a mined span can straddle
    two sentences; ``attach_qualifiers_with_scores`` requires sentence-relative offsets INSIDE the
    sentence and raises on anything else.
    """
    sentence = "indicated for asthma."
    straddling = Mention("asthma. Use", sentence.index("asthma"), sentence.index("asthma") + 11, "Disease", 1.0)
    local = _sentence_local(straddling, 0, len(sentence), sentence)
    assert (local.start, local.end, local.text) == (14, len(sentence), "asthma.")
    assert local.text == sentence[local.start : local.end]  # the mention contract still holds
    # A span starting before the sentence clips its head the same way.
    leading = Mention("for asthma", -3, 17, "Disease", 1.0)
    assert _sentence_local(leading, 4, len(sentence), sentence[4:]).start == 0


def test_indication_observations_clips_a_straddling_mention_instead_of_raising() -> None:
    """Regression guard for run 10: a boundary-straddling mention failed the task AFTER mining.

    ``shape_treatment_tables`` mined 59,441 sections (38 min, cached) and then died on
    ``ValueError: mention offsets must be sentence-relative and within sentence bounds`` because
    the overlap filter kept a straddling span and the rebasing shifted it to ``end > len(sentence)``.
    """
    text = "Examplestatin is indicated for asthma. Use with caution in women."
    straddling = Mention("asthma. Use", text.index("asthma"), text.index("Use") + 3, "Disease", 1.0)
    qualifier = Mention("women", text.index("women"), text.index("women") + 5, "BiologicalSex", 0.9)
    observations = _indication_observations(
        DailyMedEvidence(indication_docs={"SET-A": [("SET-A#34067-9", text)]}),
        ["SET-A"],
        {"object_text": "asthma", "object_category": "Disease"},
        {},
        {("SET-A", "SET-A#34067-9"): [straddling, qualifier]},
    )
    # Only the first sentence names the candidate; its host is the clipped span, and the second
    # sentence's qualifier is not attached to a host that is not there.
    assert [observation["context"] for observation in observations] == ["indication"]
    assert observations[0]["qualifiers"] == {}


def test_duplicate_indication_documents_are_mined_under_their_own_key() -> None:
    """Two indication sections of ONE SPL document get two mining-map entries, not one.

    ``doc_id`` is the SPL DOCUMENT id, so ``(set_id, doc_id)`` collides for a document carrying
    several sections (20,467 real pairs do). One shared entry hands a section mentions mined from
    a DIFFERENT text — offsets into the wrong string, which is how run 12's contraindication
    shaper died. The first section keeps the historical bare key; later ones are ordinal-suffixed.
    """
    assert _doc_key("DOC-A", 0) == "DOC-A"
    assert _doc_key("DOC-A", 1) == "DOC-A#2"
    evidence = DailyMedEvidence(indication_docs={"SET-A": [("DOC-A", "indicated for asthma"), ("DOC-A", "indicated for diabetes")]})
    mined, _ema_mined = _mine_indication_mentions(evidence, DiseaseNER(gazetteer={"asthma": "disease", "diabetes": "disease"}), None)
    assert set(mined) == {("SET-A", "DOC-A"), ("SET-A", "DOC-A#2")}
    assert [mention.text for mention in mined[("SET-A", "DOC-A")]] == ["asthma"]
    assert [mention.text for mention in mined[("SET-A", "DOC-A#2")]] == ["diabetes"]
    assert _ema_mined == {}


def test_indication_observations_reads_each_duplicate_document_section_separately() -> None:
    """The reading side resolves the same ordinal keys, so a qualifier stays with its own section."""
    first = "Examplestatin is indicated for asthma."
    second = "Examplestatin is indicated for asthma in women."
    evidence = DailyMedEvidence(indication_docs={"SET-A": [("DOC-A", first), ("DOC-A", second)]})
    mentions = {
        ("SET-A", "DOC-A"): [],  # the first section carries no qualifier
        ("SET-A", "DOC-A#2"): [
            Mention("asthma", second.index("asthma"), second.index("asthma") + 6, "Disease", 1.0),
            Mention("women", second.index("women"), second.index("women") + 5, "BiologicalSex", 0.9),
        ],
    }
    observations = _indication_observations(evidence, ["SET-A"], {"object_text": "asthma", "object_category": "Disease"}, {}, mentions)
    assert [observation["qualifiers"] for observation in observations] == [{}, {"sex_text": "women"}]
    assert [observation["doc_id"] for observation in observations] == ["DOC-A", "DOC-A"]  # rows keep the real doc_id


def test_patient_template_merges_qualifier_from_nonzero_host() -> None:
    """A patient-template qualifier must survive when its selected host is not object index zero.

    This guards the prior ``attached.get(0)`` bug: the pre-marker disease and patient-template
    disease are both candidate mentions, but only the post-marker host may receive ``women``.
    """
    sentence = "Not for asthma alone; indicated in patients with asthma in women."
    asthma = [index for index in range(len(sentence)) if sentence.startswith("asthma", index)]
    mentions = {
        ("SET-A", "SET-A#34067-9"): [
            Mention("asthma", asthma[0], asthma[0] + 6, "Disease", 1.0),
            Mention("asthma", asthma[1], asthma[1] + 6, "Disease", 1.0),
            Mention("women", sentence.index("women"), sentence.index("women") + 5, "BiologicalSex", 0.9),
        ]
    }
    observations = _indication_observations(
        DailyMedEvidence(indication_docs={"SET-A": [("SET-A#34067-9", sentence)]}), ["SET-A"], {"object_text": "asthma"}, {}, mentions
    )
    assert observations[0]["qualifiers"] == {"sex_text": "women"}


def test_dailymed_offsets_rebase_before_attachment() -> None:
    """Document-relative NER spans must attach like sentence-relative spans after rebasing.

    This guards the position-dependent false negative caused by passing a sentence at offset 500
    into a helper that compares offsets against sentence-local marker geometry.
    """
    sentence = "Drug is used in women."
    prefix = "x" * 500 + "\n"
    text = prefix + sentence
    base = len(prefix)
    mentions = {
        ("SET-A", "SET-A#34067-9"): [Mention("Drug", base, base + 4, "Disease", 1.0), Mention("women", base + 16, base + 21, "BiologicalSex", 0.9)]
    }
    observations = _indication_observations(
        DailyMedEvidence(indication_docs={"SET-A": [("SET-A#34067-9", text)]}),
        ["SET-A"],
        {"object_text": "Drug", "object_category": "Disease"},
        {},
        mentions,
    )
    assert observations[0]["qualifiers"] == {"sex_text": "women"}


def test_indication_observation_merges_duplicate_host_qualifiers_by_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate model hosts must keep the highest qualifier score within one source sentence."""
    text = "Examplestatin is indicated for asthma."
    host_start = text.index("asthma")
    monkeypatch.setattr(
        approved_treats,
        "attach_qualifiers_with_scores",
        lambda *_args: (
            {0: {"sex_text": "men"}, 1: {"sex_text": "women"}, 2: {"sex_text": "adults"}},
            {(0, "sex_text"): (0.8, "men"), (1, "sex_text"): (0.9, "women"), (2, "sex_text"): (0.7, "adults")},
        ),
    )
    observations = _indication_observations(
        _supported_evidence(text),
        ["SET-A"],
        {"object_text": "asthma", "object_category": "Disease"},
        {},
        {("SET-A", "SET-A#34067-9"): [Mention("asthma", host_start, host_start + 6, "Disease", 1.0)]},
    )
    assert observations[0]["qualifiers"] == {"sex_text": "women"}


def test_dailymed_context_and_qualifiers_aggregate_model_metadata() -> None:
    """DailyMed sentence context and qualifiers must survive aggregation, with the best model vote.

    This protects the one-pass NER observation path: unrelated sentences cannot supply a host,
    while repeated support documents may upgrade context metadata deterministically.
    """
    text = "Examplestatin is indicated for asthma in women once daily."
    ev = _supported_evidence(text)
    host_start = text.index("asthma")
    mentions = {
        ("SET-A", "SET-A#34067-9"): [
            Mention("asthma", host_start, host_start + 6, "Disease", 1.0, context_model="prevention", context_model_score=0.8),
            Mention("women", text.index("women"), text.index("women") + 5, "BiologicalSex", 0.9),
        ]
    }
    observations = _indication_observations(ev, ["SET-A"], {"object_text": "asthma"}, {}, mentions)
    assert observations[0]["context"] == "indication"
    assert observations[0]["qualifiers"] == {"sex_text": "women"}
    assert observations[0]["model"] == "prevention"


def test_approved_qualifier_merge_keeps_highest_score_and_deterministic_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = [
        {
            "set_id": "SET-A",
            "doc_id": "DOC-A",
            "context": "indication",
            "qualifiers": {"sex_text": "men"},
            "qualifier_scores": {"sex_text": (0.8, "men")},
            "model": "",
            "model_score": 0.0,
        },
        {
            "set_id": "SET-A",
            "doc_id": "DOC-B",
            "context": "indication",
            "qualifiers": {"sex_text": "women"},
            "qualifier_scores": {"sex_text": (0.9, "women")},
            "model": "",
            "model_score": 0.0,
        },
        {
            "set_id": "SET-A",
            "doc_id": "DOC-C",
            "context": "indication",
            "qualifiers": {"sex_text": "adults"},
            "qualifier_scores": {"sex_text": (0.9, "adults")},
            "model": "",
            "model_score": 0.0,
        },
    ]
    monkeypatch.setattr(approved_treats, "_condition_corroborated_sets", lambda *args: ["SET-A"])
    monkeypatch.setattr(approved_treats, "_indication_observations", lambda *args: observations)
    rows = build_approved_treats_rows(_cases("asthma"), _supported_evidence("asthma"), _MAPPING, {})
    assert rows[0]["sex_text"] == "women"  # score wins; lexical value wins the equal-score tie
    assert "qualifier_scores" not in rows[0]  # scores stay internal to the score-free row schema


def test_approved_aggregation_handles_empty_observations_and_upgrades_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The aggregator must drop empty observations and retain the highest context-model score."""
    first = {"set_id": "SET-A", "doc_id": "DOC-A", "context": "indication", "qualifiers": {}, "model": "indication", "model_score": 0.2}
    second = {"set_id": "SET-A", "doc_id": "DOC-B", "context": "indication", "qualifiers": {}, "model": "prevention", "model_score": 0.9}
    monkeypatch.setattr(approved_treats, "_condition_corroborated_sets", lambda *args: ["SET-A"])
    monkeypatch.setattr(approved_treats, "_indication_observations", lambda *args: [first, second])
    cases = _cases("asthma")
    ev = _supported_evidence("asthma")
    ev.indication_docs["SET-A"].append(("DOC-B", "asthma"))
    rows = build_approved_treats_rows(cases, ev, _MAPPING, {}, ner=None)
    assert rows[0]["assertion_context_model"] == "prevention"
    monkeypatch.setattr(approved_treats, "_indication_observations", lambda *args: [])
    assert rows[0]["assertion_context_model_score"] == "0.9"
    assert build_approved_treats_rows(cases, ev, _MAPPING, {}, ner=None) == []


def test_dailymed_fallback_yields_ner_mention_candidates() -> None:
    # No FAERS cases: an on-label condition absent from the dictionary becomes a candidate via
    # its NER mention (text-only object; rule 4 holds by construction).
    ev = _supported_evidence("Examplestatin is indicated for diabetic ketoacidosis.")
    ner = DiseaseNER(gazetteer={"diabetic ketoacidosis": "disease"})
    rows = build_approved_treats_rows(None, ev, _MAPPING, {}, ner=ner)
    assert [row["object_text"] for row in rows] == ["diabetic ketoacidosis"]
    assert rows[0]["object_curie"] == ""  # text-first: Tablassert/fullmap resolves


def test_dailymed_fallback_mention_matching_dictionary_term_is_not_duplicated(disease_map: dict[str, dict[str, str]]) -> None:
    # The offline gazetteer term coincides with the dictionary match -> one candidate, not two.
    ev = _supported_evidence("Examplestatin is indicated for hypercholesterolemia.")
    ner = DiseaseNER(gazetteer={"hypercholesterolemia": "disease"})
    rows = build_approved_treats_rows(None, ev, _MAPPING, disease_map, ner=ner)
    assert [row["object_text"] for row in rows] == ["hypercholesterolemia"]
    assert rows[0]["object_curie"] == "MONDO:0005154"  # dictionary CURIE survives


def test_dailymed_fallback_duplicate_mention_across_docs_yields_once() -> None:
    # Two documents on the same set both mine the same mention -> one candidate, not two.
    ev = _supported_evidence("Examplestatin is indicated for diabetic ketoacidosis.")
    ev.indication_docs["SET-A"].append(("SET-A#34067-9b", "Examplestatin is indicated for diabetic ketoacidosis."))
    ner = DiseaseNER(gazetteer={"diabetic ketoacidosis": "disease"})
    rows = build_approved_treats_rows(None, ev, _MAPPING, {}, ner=ner)
    assert [row["object_text"] for row in rows] == ["diabetic ketoacidosis"]


def test_dailymed_fallback_blank_mention_is_skipped() -> None:
    # A mention whose text normalizes to nothing never becomes a candidate.
    ev = _supported_evidence("Examplestatin is indicated for hypercholesterolemia.")

    class _BlankMentionNER(DiseaseNER):
        def extract(self, text: str, **kwargs: Any) -> Any:
            from dakp_pipeline.ner.lexical import Mention

            return [Mention(text="!!!", start=0, end=3, type="Disease", score=0.9)]

    assert build_approved_treats_rows(None, ev, _MAPPING, {}, ner=_BlankMentionNER()) == []


# --- once-per-section mining ------------------------------------------------------


def test_mine_indication_mentions_empty_docs_mine_nothing() -> None:
    assert _mine_indication_mentions(DailyMedEvidence(), DiseaseNER(gazetteer={"asthma": "disease"}), None) == ({}, {})


def test_mine_indication_mentions_sequential_offline() -> None:
    ev = DailyMedEvidence(indication_docs={"SET-A": [("SET-A#34067-9", "indicated for asthma")]})
    spl_mentions, ema_mentions = _mine_indication_mentions(ev, DiseaseNER(gazetteer={"asthma": "disease"}), None)
    assert [m.text for m in spl_mentions[("SET-A", "SET-A#34067-9")]] == ["asthma"]
    assert ema_mentions == {}


def test_production_ner_dispatches_multi_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production NER + devices + >1 section: mining goes through _mine_multi_gpu."""
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}},
        approval_display={"12345": "012345"},
        set_ingredient={"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF")},
        active_ingredients_by_set={"SET-A": [("Examplestatin", "UNII:QFX8B1R4QF")]},
        indication_docs={"SET-A": [("SET-A#a", "indicated for asthma"), ("SET-A#b", "indicated for asthma")]},
    )
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"})

    called: list[dict[str, Any]] = []

    def fake_multi_gpu(work_items: Any, ner_arg: Any, devs: Any) -> dict[tuple[str, str], Any]:
        called.append({"items": len(work_items), "devices": tuple(devs)})
        offline = DiseaseNER(gazetteer=ner_arg._gazetteer)
        return {(s, d): offline.extract(t) for s, d, t in work_items}

    monkeypatch.setattr(approved_treats, "_mine_multi_gpu", fake_multi_gpu)

    cases = pl.DataFrame({"nda": ["012345"], "indication": ["asthma"], "drugname": ["Examplestatin"], "ingredient": ["Examplestatin"]})
    rows = build_approved_treats_rows(cases, ev, _MAPPING, {}, ner=ner, devices=("cuda:0", "cuda:1"))
    assert called == [{"items": 2, "devices": ("cuda:0", "cuda:1")}]
    assert [row["object_text"] for row in rows] == ["asthma"]


def test_production_ner_single_section_stays_sequential(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A single section is mined inline even with devices available (no pool for one item)."""
    # Keep the inline production extract hermetic (no torch/model download): a fake gliner2
    # module plus a stubbed ensure_model, the same seam test_ner_edge.py uses.
    fake_model = types.SimpleNamespace(extract_entities=lambda text, entity_types, threshold=0.5, **_kwargs: {"entities": {}})
    module = types.ModuleType("gliner2")
    module.AutoExtractor = type("AutoExtractor", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_model)})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    monkeypatch.setattr(
        ner_module,
        "ensure_model",
        lambda model_id, **kw: ModelRef(
            model_id=model_id, source="huggingface", path=tmp_path, b3="b3:deadbeef", manifest=tmp_path / "manifest.json"
        ),
    )
    ev = _supported_evidence("indicated for asthma")
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"})
    monkeypatch.setattr(approved_treats, "_mine_multi_gpu", lambda *args: (_ for _ in ()).throw(AssertionError("must not dispatch")))
    cases = pl.DataFrame({"nda": ["012345"], "indication": ["asthma"], "drugname": ["Examplestatin"], "ingredient": ["Examplestatin"]})
    rows = build_approved_treats_rows(cases, ev, _MAPPING, {}, ner=ner, devices=("cuda:0", "cuda:1"))
    assert [row["object_text"] for row in rows] == ["asthma"]


# --- shaper NER resolution ---------------------------------------------------------


def test_shaper_uses_injected_ner(ctx: TaskContext, dailymed_refs: Any, drugsfda_refs: Any) -> None:
    """An injected params['ner'] backend is used as-is (no default construction)."""
    import dataclasses

    ner = DiseaseNER(gazetteer={"asthma": "disease"})
    injected_ctx = dataclasses.replace(ctx, params={**ctx.params, "ner": ner})
    refs = ApprovedTreatsShaper().transform([*dailymed_refs, *drugsfda_refs], injected_ctx)
    assert len(refs) == 1
    assert refs[0].uri.name == "approved_treats_assertions.tsv"
