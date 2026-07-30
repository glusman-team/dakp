"""Edge-case tests for ``dakp_pipeline.assertions.approved_treats`` (drive to 100% branch coverage).

Targets the uncovered lines: the empty-subject ``continue`` (77), the multi-NDA subject-CURIE
back-fill (97), ``_subject_for_sets`` loop-fallthrough + fallback (130->129, 132),
``_object_attrs`` no-match (141), the FAERS-candidate skip on missing NDA/indication (151),
and the DailyMed-fallback skips for an un-approved set (178) and a duplicate pair (184).
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.assertions.approved_treats import _faers_candidates, _object_attrs, _subject_for_sets, build_approved_treats_rows
from dakp_pipeline.assertions.evidence import DailyMedEvidence

# --- _subject_for_sets: loop fallthrough + fallback -----------------------------


def test_subject_for_sets_skips_sets_without_ingredient_then_matches() -> None:
    ev = DailyMedEvidence(set_ingredient={"SET-Y": ("DrugY", "UNII:Y")})
    # SET-X has no ingredient (loop continues -> 130->129), SET-Y does -> returned.
    assert _subject_for_sets(ev, ["SET-X", "SET-Y"], "fallback") == ("DrugY", "UNII:Y")


def test_subject_for_sets_falls_back_when_no_set_has_ingredient() -> None:
    ev = DailyMedEvidence(set_ingredient={})
    # No set resolves -> the stripped FAERS fallback subject, with an empty CURIE.
    assert _subject_for_sets(ev, ["SET-X", "SET-Y"], "  FallbackDrug  ") == ("FallbackDrug", "")
    assert _subject_for_sets(ev, [], "OnlyFallback") == ("OnlyFallback", "")


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
    assert rows[0]["supporting_spl_sets"] == "SET-A|SET-B"


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
        indication_docs={"SET-A": [("SET-A#doc1", "hypercholesterolemia"), ("SET-A#doc2", "hypercholesterolemia")]},
    )
    mapping = {"12345": {"EXAMPLESTATIN"}}
    rows = build_approved_treats_rows(None, ev, mapping, disease_map)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "hypercholesterolemia"
    assert rows[0]["supporting_spl_documents"] == "SET-A#doc1|SET-A#doc2"  # both docs still aggregated


def test_empty_faers_frame_yields_no_rows(disease_map: dict[str, dict[str, str]]) -> None:
    ev = DailyMedEvidence()
    cases = pl.DataFrame({"nda": [], "indication": [], "drugname": [], "ingredient": []})
    assert build_approved_treats_rows(cases, ev, {}, disease_map) == []
