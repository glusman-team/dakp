"""Tests for the NER channels in ``dakp_pipeline.assertions.approved_treats``.

Covers: the mention-based corroboration channel (rule 4) — a section NER mention word-contained
in the candidate corroborates it where dictionary + verbatim miss — the DailyMed fallback
mention candidates, the once-per-section mining (sequential + multi-GPU dispatch), and the
shaper's injected-vs-default NER resolution.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from dakp_pipeline.assertions import approved_treats
from dakp_pipeline.assertions.approved_treats import ApprovedTreatsShaper, _mine_indication_mentions, build_approved_treats_rows
from dakp_pipeline.assertions.evidence import DailyMedEvidence
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.ner.ner import DiseaseNER


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

            return [Mention(text="!!!", start=0, end=3, type="disease", score=0.9)]

    assert build_approved_treats_rows(None, ev, _MAPPING, {}, ner=_BlankMentionNER()) == []


# --- once-per-section mining ------------------------------------------------------


def test_mine_indication_mentions_empty_docs_mine_nothing() -> None:
    assert _mine_indication_mentions(DailyMedEvidence(), DiseaseNER(gazetteer={"asthma": "disease"}), None) == {}


def test_mine_indication_mentions_sequential_offline() -> None:
    ev = DailyMedEvidence(indication_docs={"SET-A": [("SET-A#34067-9", "indicated for asthma")]})
    mined = _mine_indication_mentions(ev, DiseaseNER(gazetteer={"asthma": "disease"}), None)
    assert [m.text for m in mined[("SET-A", "SET-A#34067-9")]] == ["asthma"]


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


def test_production_ner_single_section_stays_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single section is mined inline even with devices available (no pool for one item)."""
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
