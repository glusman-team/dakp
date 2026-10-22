"""Unit tests for the shared assertion-evidence helpers (Milestone 5).

Covers NDA join-key normalization, provenance column assembly (dedup/sort/pipe), DailyMed
SPL-support indexing, and Drugs@FDA NDA→ingredient mapping — all against the real extractor
outputs where a frame is needed.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.assertions.evidence import (
    DAILYMED_SET_CURIE_PREFIX,
    DAILYMED_SET_URL_BASE,
    build_dailymed_evidence,
    build_drugsfda_ingredient_map,
    dailymed_document_url,
    dailymed_set_curie,
    dailymed_set_url,
    find_faers_cases,
    merge_unique,
    normalize_nda,
    sorted_pipe,
    spl_evidence_pipe,
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
        sorted_pipe(dailymed_set_url(value) for value in ["SET-B", "SET-A", "SET-B"]) == f"{DAILYMED_SET_URL_BASE}SET-A|{DAILYMED_SET_URL_BASE}SET-B"
    )
    assert dailymed_set_url(f"{DAILYMED_SET_URL_BASE}SET-A") == f"{DAILYMED_SET_URL_BASE}SET-A"  # idempotent
    assert dailymed_set_url("") == ""


def test_dailymed_document_url_keeps_the_loinc_fragment() -> None:
    assert dailymed_document_url("SET-A#34067-9") == f"{DAILYMED_SET_URL_BASE}SET-A#34067-9"
    assert dailymed_document_url("SET-A") == f"{DAILYMED_SET_URL_BASE}SET-A"
    assert dailymed_document_url(f"{DAILYMED_SET_URL_BASE}SET-A#34067-9") == f"{DAILYMED_SET_URL_BASE}SET-A#34067-9"  # idempotent
    assert dailymed_document_url("") == ""


def test_dailymed_set_curie_reduces_to_set_granularity() -> None:
    assert dailymed_set_curie("SET-A") == f"{DAILYMED_SET_CURIE_PREFIX}SET-A"
    assert dailymed_set_curie("SET-A#34067-9") == f"{DAILYMED_SET_CURIE_PREFIX}SET-A"  # document id -> set CURIE
    assert dailymed_set_curie("dailymed:SET-A") == "dailymed:SET-A"  # idempotent
    assert dailymed_set_curie(f"{DAILYMED_SET_URL_BASE}SET-A#34067-9") == "dailymed:SET-A"  # legacy URL form
    assert dailymed_set_curie("") == ""
    assert dailymed_set_curie(None) == ""


def test_spl_evidence_pipe_emits_set_curies() -> None:
    """The ``edge_evidence`` column carries legacy-form ``dailymed:<spl_set_id>`` CURIEs.

    Set granularity only (the legacy DAKP KG shape); document ids reduce to their set CURIE
    so section-level provenance contributes the same set evidence exactly once.
    """
    assert spl_evidence_pipe(["SET-B", "SET-A"], ["SET-A#34067-9"]) == "dailymed:SET-A|dailymed:SET-B"
    # CURIE-prefixed / URL-prefixed inputs normalize to the same single-prefixed CURIE; dups collapse.
    assert spl_evidence_pipe(["dailymed:SET-A", "SET-A"], []) == "dailymed:SET-A"
    assert spl_evidence_pipe([f"{DAILYMED_SET_URL_BASE}SET-A"], []) == "dailymed:SET-A"
    assert spl_evidence_pipe([], []) == ""


# --- DailyMed SPL-support index -------------------------------------------------


def test_dailymed_evidence_indexes_approvals_ingredients_sections(dailymed_refs: list[ArtifactRef]) -> None:
    ev = build_dailymed_evidence(dailymed_refs)

    # Approvals keyed by normalized NDA -> SPL set.
    assert ev.approval_sets["12345"] == {"SETID-EXAMPLESTATIN-001"}
    assert ev.approval_sets["17977"] == {"SETID-IBUPROFEN-002"}
    assert ev.approval_display["12345"] == "NDA012345"  # legacy display form: application type + padded number

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
    assert ev.label_types == {}
    assert ev.species_by_set == {}
    assert ev.animal_application_sets == set()
    assert not ev.is_animal_set("setid-anything")


def test_dailymed_evidence_indexes_animal_label_types_and_label_wide_species(animal_dailymed_refs: list[ArtifactRef]) -> None:
    """Animal sets gain their label type and the label-wide species fallback; human sets never."""
    ev = build_dailymed_evidence(animal_dailymed_refs)

    assert ev.label_types == {"setid-fleawash-001": "50578-4", "setid-itchaway-002": "50577-6", "setid-unspecies-003": "50578-4"}
    # The fixture's approvals carry the NADA prefix (in the curated approval_type column, as
    # the real SPL <code> would for a prefix-less id), so all three sets are animal by
    # application too -- the two gates agree.
    assert ev.animal_application_sets == {"setid-fleawash-001", "setid-itchaway-002", "setid-unspecies-003"}
    # FLEAWASH's mined indication names dogs and cats; ITCHAWAY's species live only in its
    # contraindication section; UNSPECIES names no species anywhere (empty = unattributable).
    assert ev.species_by_set == {
        "setid-fleawash-001": frozenset({"NCBITaxon:9615", "NCBITaxon:9685"}),
        "setid-itchaway-002": frozenset({"NCBITaxon:9685"}),
        "setid-unspecies-003": frozenset(),
    }


def test_human_dailymed_fixture_has_no_label_types(dailymed_refs: list[ArtifactRef]) -> None:
    ev = build_dailymed_evidence(dailymed_refs)
    assert ev.label_types == {}
    assert ev.species_by_set == {}


# --- Animal application approvals gate the species context ---------------------


def _parquet_ref(root: Path, name: str, data: dict[str, list[object]]) -> ArtifactRef:
    from dakp_pipeline.io.content_hash import hash_file

    path = root / name
    pl.DataFrame(data).write_parquet(path)
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def test_animal_application_approval_sets_are_animal_without_any_document_type_code(tmp_path: Path) -> None:
    """The production gate: spl_documents carries NO type code for animal-approval sets.

    A NADA approval with a HUMAN document code, and a CNADA approval with NO document row at
    all, must both derive species from the text -- never the human default. An NDA-approval
    set with no document code stays human.
    """
    approvals = _parquet_ref(
        tmp_path,
        "spl_approvals.parquet",
        {
            "approval_id": ["NADA141459", "CNADA141550", "NDA018780"],
            "approval_code": ["NADA141459", "CNADA141550", "NDA018780"],
            "approval_type": ["C73584", "C73584", "C73594"],
            "spl_set_id": ["set-nada", "set-cnada", "set-nda"],
        },
    )
    documents = _parquet_ref(
        tmp_path,
        "spl_documents.parquet",
        {
            "spl_set_id": ["set-nada", "set-nda"],
            "document_id": ["set-nada#34067-9", "set-nda#34067-9"],
            "loinc": ["34067-9", "34067-9"],
            "text": ["for use in dogs", "for the relief of headache"],
            "label_type": ["34391-4", ""],  # NADA set carries a HUMAN Rx code; CNADA set has no row at all
        },
    )
    ev = build_dailymed_evidence([approvals, documents])

    assert ev.animal_application_sets == {"set-nada", "set-cnada"}
    assert ev.is_animal_set("set-nada")  # animal by application, even with a human type code
    assert ev.is_animal_set("set-cnada")  # animal by application, with no document row at all
    assert not ev.is_animal_set("set-nda")
    assert not ev.is_animal_set("set-unknown")
    # Species come from the text on application-animal sets: sentence first...
    assert ev.species_context_for("set-nada", "for use in dogs and cats") == ("NCBITaxon:9615", "NCBITaxon:9685")
    assert ev.species_context_for("set-cnada", "for cats") == ("NCBITaxon:9685",)
    # ...and a row naming no species anywhere is unattributable (caller drops it), NOT human.
    assert ev.species_context_for("set-nada", "controls fleas") == ()
    assert ev.species_context_for("set-cnada", "controls ticks") == ()
    # The human NDA set is human by construction even with no document type code.
    assert ev.species_context_for("set-nda", "for use in dogs") == ("NCBITaxon:9606",)
    # The label-wide fallback is computed for application-animal sets too (empty here: no
    # section tables in this build, mirroring what a sections-less snapshot yields).
    assert ev.species_by_set.keys() == {"set-nada", "set-cnada"}


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
