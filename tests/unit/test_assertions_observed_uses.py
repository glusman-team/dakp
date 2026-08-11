"""Unit tests for FAERS observed-use (applied_to_treat) aggregation (Milestone 5).

Covers distinct-case case-count aggregation, the preserved FAERS label/status behavior, object
resolution via the lexical baseline, provenance columns, determinism, empty inputs, and the
end-to-end shaper TSV output.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.assertions.evidence import find_faers_cases
from dakp_pipeline.assertions.observed_uses import ObservedUsesShaper, build_observed_use_rows, is_non_disease_indication
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext


def test_case_count_aggregates_distinct_cases(disease_map: dict[str, dict[str, str]]) -> None:
    # (DrugX, condY) appears across 3 distinct cases; one case contributes two rows (e.g. two
    # drug_seq) which must NOT inflate the count. (DrugX, other) is a single case.
    cases = pl.DataFrame(
        {
            "primaryid": ["1", "2", "3", "3", "9"],
            "drugname": ["DrugX", "DrugX", "DrugX", "DrugX", "DrugX"],
            "indication": ["condY", "condY", "condY", "condY", "other"],
        }
    )
    rows = build_observed_use_rows(cases, disease_map)
    counts = {(r["subject_text"], r["object_text"]): r["case_count"] for r in rows}
    assert counts[("DrugX", "condY")] == "3"  # distinct primaryids, not 4 rows
    assert counts[("DrugX", "other")] == "1"


def test_case_count_falls_back_to_rows_without_primaryid(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame({"drugname": ["DrugX", "DrugX"], "indication": ["condY", "condY"]})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["case_count"] == "2"  # no primaryid column -> row count


def test_case_count_mixes_distinct_cases_and_anonymous_rows(disease_map: dict[str, dict[str, str]]) -> None:
    # Distinct non-empty primaryids dedup; null/empty primaryids each count as their own
    # observation (legacy _row{index} fallback) — the pair total is the sum of both.
    cases = pl.DataFrame({"primaryid": ["1", "", None, "1", "2"], "drugname": ["DrugX"] * 5, "indication": ["condY"] * 5})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["case_count"] == "4"  # distinct {1, 2} + 2 anonymous rows


def test_observed_uses_from_fixture_cases(faers_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    cases = find_faers_cases(faers_refs)
    rows = build_observed_use_rows(cases, disease_map)
    by_subject = {r["subject_text"]: r for r in rows}

    # No DELETE fixture -> Examplestatin, Advil, Placebo all present, one case each.
    assert set(by_subject) == {"Examplestatin", "Advil", "Placebo"}
    assert by_subject["Examplestatin"]["object_text"] == "hypercholesterolemia"
    assert by_subject["Examplestatin"]["object_curie"] == "MONDO:0005154"
    assert by_subject["Examplestatin"]["case_count"] == "1"
    # 'back pain' resolves through the dictionary substring match on 'pain'.
    assert by_subject["Placebo"]["object_text"] == "pain"
    assert by_subject["Advil"]["predicate"] == "biolink:applied_to_treat"


def test_faers_label_and_status_behavior_preserved(faers_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    rows = build_observed_use_rows(find_faers_cases(faers_refs), disease_map)
    assert rows
    for row in rows:
        # biolink-valid ClinicalApprovalStatusEnum member: FAERS makes no approval claim
        # (the legacy ``observed_use`` label is not an enum member and would fail validation
        # now that Tablassert >= 8.2 emits the field first-class).
        assert row["clinical_approval_status"] == "not_provided"
        assert row["knowledge_level"] == "statistical_association"
        assert row["agent_type"] == "manual_validation_of_automated_agent"
        assert row["primary_knowledge_source"] == "infores:multiomics-drugapprovals"
        assert row["upstream_resource_ids"] == "infores:faers|infores:dailymed"
        assert row["subject_curie"] == ""  # FAERS provides no drug id here (text-first)


def test_rows_are_deterministically_ordered(faers_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    cases = find_faers_cases(faers_refs)
    first = build_observed_use_rows(cases, disease_map)
    second = build_observed_use_rows(cases, disease_map)
    assert first == second
    keys = [(r["subject_text"], r["object_text"]) for r in first]
    assert keys == sorted(keys)


def test_no_faers_cases_yields_no_rows(disease_map: dict[str, dict[str, str]]) -> None:
    assert build_observed_use_rows(None, disease_map) == []


def test_is_non_disease_indication_classifier() -> None:
    # Placeholders / usage-context / generic procedures (no real condition named) -> filtered.
    for bad in [
        "Product used for unknown indication",
        "Prophylaxis",
        "Ill-defined disorder",
        "Off label use",
        "Chemotherapy",
        "Medication error",
        "Premedication",
        "Adverse drug reaction",
        "Supplementation therapy",
        "Product use in unapproved indication",
        # legacy DAKP FAERS stop-list union (ref/legacy FAERS/bin):
        "Intentional product misuse",
        "Exposure during pregnancy",
        "Foetal exposure during pregnancy",
        "Product origin unknown",
        "Accidental exposure to product",
        "Product use issue",
        "Product administration",
    ]:
        assert is_non_disease_indication(bad), bad
    # Case-insensitive and tolerant of surrounding whitespace.
    assert is_non_disease_indication("  product used for UNKNOWN indication  ")
    # Specific conditions — including procedure-worded ones that NAME a condition — are kept.
    for good in [
        "Migraine prophylaxis",
        "Prophylaxis against graft versus host disease",
        "Hormone receptor positive HER2 negative breast cancer",
        "Type 2 diabetes mellitus",
        "asthma",
        "pain",
        "Contraception",
    ]:
        assert not is_non_disease_indication(good), good


def test_non_disease_indications_are_filtered_from_rows(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame({"drugname": ["DrugX", "DrugX", "DrugX"], "indication": ["Product used for unknown indication", "Prophylaxis", "asthma"]})
    rows = build_observed_use_rows(cases, disease_map)
    # Only the real condition survives; the two placeholder indications are dropped.
    assert [r["object_text"] for r in rows] == ["asthma"]


def test_shaper_writes_uncompressed_tsv_with_contract_columns(faers_refs: list[ArtifactRef], ctx: TaskContext) -> None:
    refs = ObservedUsesShaper().transform(faers_refs, ctx)
    assert len(refs) == 1
    out = refs[0]
    assert out.uri.name == "faers_applied_to_treat_assertions.tsv"

    frame = schemas.read_table(out.uri)
    assert frame.columns == schemas.FAERS_APPLIED_TO_TREAT_COLUMNS
    assert frame.height == 3
    assert out.uri.read_bytes().startswith(b"subject_text\t")
