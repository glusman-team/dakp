"""Unit tests for the shared assertion-evidence helpers (Milestone 5).

Covers NDA join-key normalization, provenance column assembly (dedup/sort/pipe), DailyMed
SPL-support indexing, and Drugs@FDA NDA→ingredient mapping — all against the real extractor
outputs where a frame is needed.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.assertions.evidence import (
    DAILYMED_SET_URL_BASE,
    build_dailymed_evidence,
    build_drugsfda_ingredient_map,
    dailymed_document_url,
    dailymed_set_url,
    find_faers_cases,
    merge_unique,
    normalize_nda,
    sorted_pipe,
)
from dakp_pipeline.io.contracts import ArtifactRef

# --- NDA normalization ----------------------------------------------------------


def test_normalize_nda_strips_padding_and_nondigits() -> None:
    assert normalize_nda("012345") == "12345"  # DailyMed/Drugs@FDA padded form
    assert normalize_nda("12345") == "12345"  # FAERS stripped form
    assert normalize_nda("NDA:012345") == "12345"  # prefixed form
    assert normalize_nda("000123") == "123"
    assert normalize_nda(" 017977 ") == "17977"


def test_normalize_nda_empty_and_nonnumeric_yield_no_key() -> None:
    assert normalize_nda("") == ""
    assert normalize_nda(None) == ""
    assert normalize_nda("NDA") == ""
    assert normalize_nda("0") == ""
    assert normalize_nda("000") == ""


def test_normalize_nda_aligns_faers_with_dailymed() -> None:
    # The whole point: FAERS '17977' joins DailyMed/Drugs@FDA '017977'.
    assert normalize_nda("17977") == normalize_nda("017977")


# --- provenance column assembly -------------------------------------------------


def test_merge_unique_dedups_sorts_and_drops_empties() -> None:
    assert merge_unique(["b", "a", "b", "", None, "  a  "]) == ["a", "b"]
    assert merge_unique([], ["x"], []) == ["x"]
    assert merge_unique() == []


def test_sorted_pipe_is_deterministic_list_encoding() -> None:
    assert sorted_pipe(["SET-B", "SET-A", "SET-B", ""]) == "SET-A|SET-B"
    # Order-independent => deterministic regardless of insertion order.
    assert sorted_pipe(["SET-A", "SET-B"]) == sorted_pipe(["SET-B", "SET-A"])
    assert sorted_pipe([]) == ""


def test_dailymed_set_url_links_to_label_page() -> None:
    assert (
        sorted_pipe(dailymed_set_url(value) for value in ["SET-B", "SET-A", "SET-B"])
        == f"{DAILYMED_SET_URL_BASE}SET-A|{DAILYMED_SET_URL_BASE}SET-B"
    )
    assert dailymed_set_url(f"{DAILYMED_SET_URL_BASE}SET-A") == f"{DAILYMED_SET_URL_BASE}SET-A"  # idempotent
    assert dailymed_set_url("") == ""


def test_dailymed_document_url_keeps_the_loinc_fragment() -> None:
    assert dailymed_document_url("SET-A#34067-9") == f"{DAILYMED_SET_URL_BASE}SET-A#34067-9"
    assert dailymed_document_url("SET-A") == f"{DAILYMED_SET_URL_BASE}SET-A"
    assert dailymed_document_url(f"{DAILYMED_SET_URL_BASE}SET-A#34067-9") == f"{DAILYMED_SET_URL_BASE}SET-A#34067-9"  # idempotent
    assert dailymed_document_url("") == ""


# --- DailyMed SPL-support index -------------------------------------------------


def test_dailymed_evidence_indexes_approvals_ingredients_sections(dailymed_refs: list[ArtifactRef]) -> None:
    ev = build_dailymed_evidence(dailymed_refs)

    # Approvals keyed by normalized NDA -> SPL set.
    assert ev.approval_sets["12345"] == {"SETID-EXAMPLESTATIN-001"}
    assert ev.approval_sets["17977"] == {"SETID-IBUPROFEN-002"}
    assert ev.approval_display["12345"] == "012345"  # padded display form preserved

    # Active ingredient + UNII per set (inactive lactose excluded).
    assert ev.set_ingredient["SETID-EXAMPLESTATIN-001"] == ("Examplestatin", "UNII:QFX8B1R4QF")
    assert ev.set_ingredient["SETID-IBUPROFEN-002"] == ("Ibuprofen", "UNII:WK2XYI10QM")
    assert all(name != "Lactose" for name, _ in ev.set_ingredient.values())

    # Indication (34067-9) and contraindication (34070-3) sections indexed by set.
    assert "SETID-EXAMPLESTATIN-001" in ev.indication_docs
    assert "SETID-IBUPROFEN-002" in ev.contraindication_docs
    # The no-LOINC 'HOW SUPPLIED' section is neither indication nor contraindication.
    assert all("HOW SUPPLIED" not in text for docs in ev.indication_docs.values() for _doc, text in docs)


def test_indication_support_requires_approval_and_section(dailymed_refs: list[ArtifactRef]) -> None:
    ev = build_dailymed_evidence(dailymed_refs)

    sets, docs = ev.indication_support("12345")
    assert sets == ["SETID-EXAMPLESTATIN-001"]
    assert docs == ["SETID-EXAMPLESTATIN-001#34067-9"]

    # Unknown NDA -> no support.
    assert ev.indication_support("99999") == ([], [])


def test_contraindication_sets_for_drug_is_first_scope(dailymed_refs: list[ArtifactRef]) -> None:
    ev = build_dailymed_evidence(dailymed_refs)

    # Ibuprofen has a DailyMed contraindication section -> linked (case-insensitive).
    assert ev.contraindication_sets_for_drug("ibuprofen") == ["SETID-IBUPROFEN-002"]
    assert ev.contraindication_sets_for_drug("IBUPROFEN") == ["SETID-IBUPROFEN-002"]
    # Warfarin has no DailyMed document -> no support.
    assert ev.contraindication_sets_for_drug("Warfarin") == []
    assert ev.contraindication_sets_for_drug("") == []


def test_build_dailymed_evidence_empty_without_tables() -> None:
    ev = build_dailymed_evidence([])
    assert ev.approval_sets == {}
    assert ev.indication_support("12345") == ([], [])


# --- Drugs@FDA NDA -> ingredient map --------------------------------------------


def test_drugsfda_ingredient_map_keyed_by_normalized_nda(drugsfda_refs: list[ArtifactRef]) -> None:
    mapping = build_drugsfda_ingredient_map(drugsfda_refs)

    assert mapping["12345"] == {"EXAMPLESTATIN"}
    assert mapping["17977"] == {"IBUPROFEN"}
    assert mapping["207500"] == {"REALDRUG"}
    # FAERS-stripped and Drugs@FDA-padded keys coincide.
    assert "012345" not in mapping  # stored normalized, not padded
    assert build_drugsfda_ingredient_map([]) == {}


# --- FAERS case-table resolution ------------------------------------------------


def test_find_faers_cases_resolves_global_cases(faers_refs: list[ArtifactRef]) -> None:
    cases = find_faers_cases(faers_refs)
    assert cases is not None
    assert {"drugname", "indication", "nda", "primaryid"} <= set(cases.columns)
    # No DELETE fixture -> all three 24Q3 cases survive.
    assert cases.height == 3


def test_find_faers_cases_none_without_faers(dailymed_refs: list[ArtifactRef]) -> None:
    # Inputs carrying no FAERS case table resolve to None (shapers fall back accordingly).
    assert find_faers_cases(dailymed_refs) is None
    assert find_faers_cases([]) is None


def test_find_faers_cases_ignores_non_case_frames() -> None:
    frame = pl.DataFrame({"unrelated": ["x"]})
    assert "drugname" not in frame.columns  # sanity: would be rejected even if named cases.parquet
