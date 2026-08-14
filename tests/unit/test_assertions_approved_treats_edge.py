"""Edge-case tests for ``dakp_pipeline.assertions.approved_treats`` (drive to 100% branch coverage).

Targets: ``_subject_for_sets`` singleton adoption + multi-ingredient/missing-ingredient
fallthrough to the FAERS fallback, ``_object_attrs`` no-match, the FAERS-candidate skips on
missing NDA/indication, the empty-subject drop, the multi-NDA subject-CURIE back-fill, the
condition-corroboration gate (dictionary CURIE/text match, verbatim substring, provenance
restriction, drop counter), and the DailyMed-fallback skips for an un-approved set and a
duplicate pair.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from dakp_pipeline.assertions.approved_treats import _faers_candidates, _object_attrs, _subject_for_sets, build_approved_treats_rows
from dakp_pipeline.assertions.evidence import DailyMedEvidence

# --- _subject_for_sets: singleton adoption + fallthrough ------------------------


def test_subject_for_sets_skips_sets_without_ingredient_then_matches() -> None:
    ev = DailyMedEvidence(set_ingredient={"SET-Y": ("DrugY", "UNII:Y")}, active_ingredients_by_set={"SET-Y": [("DrugY", "UNII:Y")]})
    # SET-X has no ingredient (loop continues), SET-Y is a singleton -> its (name, UNII) returned.
    assert _subject_for_sets(ev, ["SET-X", "SET-Y"], "fallback") == ("DrugY", "UNII:Y")


def test_subject_for_sets_falls_back_when_no_set_has_ingredient() -> None:
    ev = DailyMedEvidence(set_ingredient={})
    # No set resolves -> the stripped FAERS fallback subject, with an empty CURIE.
    assert _subject_for_sets(ev, ["SET-X", "SET-Y"], "  FallbackDrug  ") == ("FallbackDrug", "")
    assert _subject_for_sets(ev, [], "OnlyFallback") == ("OnlyFallback", "")


def test_subject_for_sets_skips_multi_ingredient_set() -> None:
    # A combination-product set (two actives) is NOT trusted for drug identity: adopting one
    # component would over-attribute the treatment -> fall through to the FAERS fallback.
    ev = DailyMedEvidence(
        set_ingredient={"SET-COMBO": ("ComponentA", "UNII:A")},
        active_ingredients_by_set={"SET-COMBO": [("ComponentA", "UNII:A"), ("ComponentB", "UNII:B")]},
    )
    assert _subject_for_sets(ev, ["SET-COMBO"], "ComboFallback") == ("ComboFallback", "")


# --- _object_attrs: no disease-map match ----------------------------------------


def test_object_attrs_no_match_returns_text_default_category(disease_map: dict[str, dict[str, str]]) -> None:
    assert _object_attrs("zzz_unknown_condition", disease_map) == ("", "zzz_unknown_condition", "Disease")


def test_object_attrs_match_returns_first_baseline_match(disease_map: dict[str, dict[str, str]]) -> None:
    curie, name, category = _object_attrs("hypercholesterolemia", disease_map)
    assert curie == "MONDO:0005154"
    assert name == "hypercholesterolemia"
    assert category == "Disease"


# --- _faers_candidates: missing NDA / indication / duplicates -------------------


def test_faers_candidates_skip_missing_nda_or_indication_and_dedup(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame(
        {
            "nda": ["", "123", "012345", "012345"],
            "indication": ["hypercholesterolemia", "", "hypercholesterolemia", "hypercholesterolemia"],
            "drugname": ["a", "b", "c", "c"],
            "ingredient": ["", "", "", ""],
        }
    )
    cands = list(_faers_candidates(cases, disease_map))
    # row0: empty NDA -> skipped; row1: empty indication -> skipped; row2 yielded; row3 dup -> skipped.
    assert len(cands) == 1
    assert cands[0]["norm_nda"] == "12345"
    assert cands[0]["object_text"] == "hypercholesterolemia"
    assert cands[0]["fallback_subject"] == "c"  # ingredient empty -> falls back to drugname


def test_faers_candidates_uses_nda_raw_and_ingredient_columns(disease_map: dict[str, dict[str, str]]) -> None:
    # A frame shaped with the legacy 'nda_raw' / 'ingredient' aliases instead of 'nda'/'drugname'.
    cases = pl.DataFrame({"nda_raw": ["012345"], "indication": ["hypercholesterolemia"], "ingredient": ["Examplestatin"]})
    cands = list(_faers_candidates(cases, disease_map))
    assert len(cands) == 1
    assert cands[0]["norm_nda"] == "12345"
    assert cands[0]["fallback_subject"] == "Examplestatin"


# --- build_approved_treats_rows: empty-subject skip (77) ------------------------


def test_candidate_with_no_resolvable_subject_is_skipped(disease_map: dict[str, dict[str, str]]) -> None:
    # The approved set has NO active ingredient and the FAERS row has no fallback subject
    # -> _subject_for_sets returns ("", "") -> the candidate is dropped.
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}},
        approval_display={"12345": "012345"},
        set_ingredient={},  # no ingredient for SET-A
        indication_docs={"SET-A": [("SET-A#34067-9", "hypercholesterolemia")]},
    )
    mapping = {"12345": {"EXAMPLESTATIN"}}
    cases = pl.DataFrame({"nda": ["012345"], "indication": ["hypercholesterolemia"], "drugname": [""], "ingredient": [""]})
    assert build_approved_treats_rows(cases, ev, mapping, disease_map) == []


# --- build_approved_treats_rows: multi-NDA subject-CURIE back-fill (97) ---------


def test_second_nda_backfills_subject_curie_for_shared_key(disease_map: dict[str, dict[str, str]]) -> None:
    # Two NDAs resolve the SAME subject text 'DrugX'; the first set lacks a UNII, the second
    # provides one -> the aggregate's subject_curie is back-filled from the second candidate.
    ev = DailyMedEvidence(
        approval_sets={"11111": {"SET-A"}, "22222": {"SET-B"}},
        approval_display={"11111": "011111", "22222": "022222"},
        set_ingredient={"SET-A": ("DrugX", ""), "SET-B": ("DrugX", "UNII:X")},
        active_ingredients_by_set={"SET-A": [("DrugX", "")], "SET-B": [("DrugX", "UNII:X")]},
        indication_docs={"SET-A": [("SET-A#34067-9", "condY")], "SET-B": [("SET-B#34067-9", "condY")]},
    )
    mapping = {"11111": {"DRUGX"}, "22222": {"DRUGX"}}
    cases = pl.DataFrame(
        {"nda": ["011111", "022222"], "indication": ["condY", "condY"], "drugname": ["DrugX", "DrugX"], "ingredient": ["DrugX", "DrugX"]}
    )
    rows = build_approved_treats_rows(cases, ev, mapping, disease_map)
    assert len(rows) == 1
    assert rows[0]["subject_text"] == "DrugX"
    assert rows[0]["subject_curie"] == "UNII:X"  # back-filled from the second NDA's set
    assert rows[0]["approval_ids"] == "011111|022222"
    assert (
        rows[0]["supporting_spl_sets"]
        == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A|https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-B"
    )


# --- DailyMed fallback: un-approved set (178) + duplicate pair (184) ------------


def test_dailymed_fallback_skips_set_with_no_approval(disease_map: dict[str, dict[str, str]]) -> None:
    # SET-Z has an indication section but NO approval maps to it -> no NDAs -> skipped.
    ev = DailyMedEvidence(
        approval_sets={},  # nothing approved
        indication_docs={"SET-Z": [("SET-Z#34067-9", "hypercholesterolemia")]},
    )
    assert build_approved_treats_rows(None, ev, {}, disease_map) == []


def test_dailymed_fallback_dedups_repeated_indication_docs(disease_map: dict[str, dict[str, str]]) -> None:
    # The same (NDA, disease) pair appears in TWO indication docs of the same set -> the
    # second occurrence is skipped (key already seen), yielding exactly one row.
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}},
        approval_display={"12345": "012345"},
        set_ingredient={"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF")},
        active_ingredients_by_set={"SET-A": [("Examplestatin", "UNII:QFX8B1R4QF")]},
        indication_docs={"SET-A": [("SET-A#doc1", "hypercholesterolemia"), ("SET-A#doc2", "hypercholesterolemia")]},
    )
    mapping = {"12345": {"EXAMPLESTATIN"}}
    rows = build_approved_treats_rows(None, ev, mapping, disease_map)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "hypercholesterolemia"
    assert (
        rows[0]["supporting_spl_documents"]
        == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#doc1|https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#doc2"
    )  # both docs still aggregated


def test_empty_faers_frame_yields_no_rows(disease_map: dict[str, dict[str, str]]) -> None:
    ev = DailyMedEvidence()
    cases = pl.DataFrame({"nda": [], "indication": [], "drugname": [], "ingredient": []})
    assert build_approved_treats_rows(cases, ev, {}, disease_map) == []


# --- condition-in-label corroboration (rule 4) ----------------------------------


def _supported_evidence(section_text: str, *, second_set_text: str | None = None) -> DailyMedEvidence:
    """NDA 12345 approved on SET-A (and SET-B when ``second_set_text`` is given)."""
    approval_sets = {"12345": {"SET-A"}}
    set_ingredient = {"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF")}
    active = {"SET-A": [("Examplestatin", "UNII:QFX8B1R4QF")]}
    docs = {"SET-A": [("SET-A#34067-9", section_text)]}
    if second_set_text is not None:
        approval_sets["12345"].add("SET-B")
        set_ingredient["SET-B"] = ("Examplestatin", "UNII:QFX8B1R4QF")
        active["SET-B"] = [("Examplestatin", "UNII:QFX8B1R4QF")]
        docs["SET-B"] = [("SET-B#34067-9", second_set_text)]
    return DailyMedEvidence(
        approval_sets=approval_sets,
        approval_display={"12345": "012345"},
        set_ingredient=set_ingredient,
        active_ingredients_by_set=active,
        indication_docs=docs,
    )


def _cases(indication: str) -> pl.DataFrame:
    return pl.DataFrame({"nda": ["012345"], "indication": [indication], "drugname": ["Examplestatin"], "ingredient": ["Examplestatin"]})


def test_candidate_dropped_when_condition_absent_from_label(disease_map: dict[str, dict[str, str]]) -> None:
    # Every other gate passes, but the label's indication section names only 'asthma' — the
    # candidate condition 'hypercholesterolemia' matches NEITHER the dictionary hit (different
    # CURIE) nor the section text verbatim -> dropped (the legacy supportInDailyMed gate).
    ev = _supported_evidence("Examplestatin is indicated for asthma.")
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    try:
        rows = build_approved_treats_rows(_cases("hypercholesterolemia"), ev, {"12345": {"EXAMPLESTATIN"}}, disease_map)
    finally:
        logger.remove(sink_id)
    assert rows == []
    assert any("dropped_no_label_term_support = 1" in line for line in lines)


def test_candidate_dropped_when_no_dictionary_entry_and_no_verbatim_mention(disease_map: dict[str, dict[str, str]]) -> None:
    # 'cephalalgia' is not in the dictionary (candidate carries no CURIE): the section's
    # dictionary hit ('headache') is text-unequal and the verbatim substring check misses too.
    ev = _supported_evidence("Examplestatin is indicated for headache.")
    assert build_approved_treats_rows(_cases("cephalalgia"), ev, {"12345": {"EXAMPLESTATIN"}}, disease_map) == []


def test_candidate_kept_via_verbatim_substring_match() -> None:
    # Empty dictionary: corroboration rests on the word-bounded verbatim mention alone
    # (case/punctuation-insensitive normalized space, mirroring the LexicalMatcher).
    ev = _supported_evidence("Indicated for the treatment of Diabetic Ketoacidosis in adults.")
    rows = build_approved_treats_rows(_cases("diabetic ketoacidosis"), ev, {"12345": {"EXAMPLESTATIN"}}, {})
    assert [row["object_text"] for row in rows] == ["diabetic ketoacidosis"]
    assert rows[0]["supporting_spl_sets"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A"


def test_candidate_kept_via_dictionary_curie_match_with_different_surface_form() -> None:
    # The section names the same concept under a different surface form: the CURIE match (not a
    # verbatim mention) corroborates the candidate.
    disease_map = {
        "hypercholesterolemia": {"curie": "MONDO:0005154", "name": "hypercholesterolemia", "category": "Disease"},
        "high cholesterol": {"curie": "MONDO:0005154", "name": "hypercholesterolemia", "category": "Disease"},
    }
    ev = _supported_evidence("Indicated in adults with high cholesterol.")
    rows = build_approved_treats_rows(_cases("hypercholesterolemia"), ev, {"12345": {"EXAMPLESTATIN"}}, disease_map)
    assert [row["object_text"] for row in rows] == ["hypercholesterolemia"]


def test_candidate_kept_via_normalized_text_equality_without_curie() -> None:
    # Neither side carries a CURIE (dictionary entry with an empty curie): normalized-text
    # equality between the section's dictionary hit and the candidate corroborates it.
    disease_map = {"type 2 diabetes": {"curie": "", "name": "type 2 diabetes", "category": "Disease"}}
    ev = _supported_evidence("Indicated for TYPE 2 DIABETES.")
    rows = build_approved_treats_rows(_cases("Type 2 Diabetes"), ev, {"12345": {"EXAMPLESTATIN"}}, disease_map)
    assert [row["object_text"] for row in rows] == ["Type 2 Diabetes"]


def test_provenance_limited_to_condition_mentioning_sets(disease_map: dict[str, dict[str, str]]) -> None:
    # SET-A's label mentions the condition, SET-B's does not -> the row is kept, but only SET-A
    # (and its document) is cited as supporting SPL evidence.
    ev = _supported_evidence("Indicated for hypercholesterolemia.", second_set_text="Indicated for asthma.")
    rows = build_approved_treats_rows(_cases("hypercholesterolemia"), ev, {"12345": {"EXAMPLESTATIN"}}, disease_map)
    assert len(rows) == 1
    assert rows[0]["supporting_spl_sets"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A"
    assert rows[0]["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#34067-9"


def test_candidate_with_unnormalizable_condition_text_is_dropped(disease_map: dict[str, dict[str, str]]) -> None:
    # An indication that normalizes to the empty string can never be corroborated -> dropped.
    ev = _supported_evidence("Indicated for hypercholesterolemia.")
    assert build_approved_treats_rows(_cases("!!!"), ev, {"12345": {"EXAMPLESTATIN"}}, disease_map) == []


def test_multi_ingredient_supporting_set_yields_faers_fallback_subject(disease_map: dict[str, dict[str, str]]) -> None:
    # The only corroborating set is a combination product -> no DailyMed (name, UNII) adoption;
    # the subject is the FAERS-reported ingredient text with an empty CURIE.
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-COMBO"}},
        approval_display={"12345": "012345"},
        set_ingredient={"SET-COMBO": ("ComponentA", "UNII:A")},
        active_ingredients_by_set={"SET-COMBO": [("ComponentA", "UNII:A"), ("ComponentB", "UNII:B")]},
        indication_docs={"SET-COMBO": [("SET-COMBO#34067-9", "Indicated for hypercholesterolemia.")]},
    )
    rows = build_approved_treats_rows(_cases("hypercholesterolemia"), ev, {"12345": {"COMPONENTA"}}, disease_map)
    assert len(rows) == 1
    assert rows[0]["subject_text"] == "Examplestatin"  # FAERS fallback (ingredient column)
    assert rows[0]["subject_curie"] == ""
