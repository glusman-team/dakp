"""Unit tests for FAERS observed-use (applied_to_treat) aggregation (Milestone 5).

Covers distinct-case case-count aggregation, the preserved FAERS label/status behavior, object
resolution via the lexical baseline, provenance columns, determinism, empty inputs, and the
end-to-end shaper TSV output.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.assertions.evidence import FDAApprovalIndex, find_faers_cases
from dakp_pipeline.assertions.observed_uses import ObservedUsesShaper, _approved_pair_index, build_observed_use_rows, is_non_disease_indication
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
    ids = {(r["subject_text"], r["object_text"]): r["case_ids"] for r in rows}
    assert ids[("DrugX", "condY")] == "1|2|3"  # the exact token set behind the count
    assert ids[("DrugX", "other")] == "9"


def test_observed_use_retains_faers_report_and_nda_provenance(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame(
        {
            "drugname": ["Advil", "Advil"],
            "indication": ["headache", "headache"],
            "primaryid": ["1001", "1002"],
            "nda": ["17977", "017977"],
            "nda_raw": ["017977", "017977"],
            "quarter": ["24Q3", "24Q2"],
            "drug_seq": ["1", "2"],
            "source_record_id": ["24Q3:1001:1:headache", "24Q2:1002:2:headache"],
        }
    )
    rows = build_observed_use_rows(
        cases,
        disease_map,
        approved_pairs=set(),
        faers_quarter_urls={"24Q3": "https://example.test/faers-24q3.zip", "24Q2": "https://example.test/faers-24q2.zip"},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["case_count"] == "2"
    # No approval index supplied: the number is emitted exactly as FAERS recorded it.
    assert row["FDA_regulatory_approvals"] == "017977"
    assert row["edge_evidence"] == ""  # faers: report ids no longer ride publications
    assert row["supporting_faers_records"] == "24Q2:1002:2:headache|24Q3:1001:1:headache"
    assert row["supporting_faers_urls"] == "https://example.test/faers-24q2.zip|https://example.test/faers-24q3.zip"


def test_observed_use_expands_faers_numbers_to_the_fda_display_form(disease_map: dict[str, dict[str, str]]) -> None:
    """FAERS strips both the type prefix and the leading zeros; the index puts them back.

    The reported bug: pembrolizumab's ``applied_to_treat`` edge carried ``125514``, which
    resolves to nothing — the FDA application is ``BLA125514``.
    """
    cases = pl.DataFrame(
        {
            "drugname": ["Keytruda", "Keytruda", "Keytruda"],
            "indication": ["headache", "headache", "headache"],
            "primaryid": ["1001", "1002", "1003"],
            "nda": ["125514", "0125514", "17977"],
            "nda_raw": ["125514", "0125514", "17977"],
            "quarter": ["24Q3", "24Q3", "24Q3"],
            "source_record_id": ["a", "b", "c"],
        }
    )
    index = FDAApprovalIndex({"125514": ("BLA125514",), "17977": ("NDA017977",)})
    rows = build_observed_use_rows(cases, disease_map, approved_pairs=set(), approvals=index)

    assert len(rows) == 1
    # Both FAERS spellings of 125514 collapse to the single FDA form, sorted with the other.
    assert rows[0]["FDA_regulatory_approvals"] == "BLA125514|NDA017977"


def test_case_count_falls_back_to_rows_without_primaryid(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame({"drugname": ["DrugX", "DrugX"], "indication": ["condY", "condY"]})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["case_count"] == "2"  # no primaryid column -> row count
    # Id-less rows still carry one token each (per-group synthetic pads), keeping
    # len(case_ids) == case_count so the Tablassert merge union stays exact.
    assert rows[0]["case_ids"] == "anon:row:DrugX:condY:0|anon:row:DrugX:condY:1"


def test_case_ids_tokenize_anonymous_rows_by_source_record(disease_map: dict[str, dict[str, str]]) -> None:
    # Primaryid-less rows with a source_record_id tokenize as ``anon:<source_record_id>``;
    # the token set size still equals the count.
    cases = pl.DataFrame(
        {
            "primaryid": ["1", "", ""],
            "drugname": ["DrugX"] * 3,
            "indication": ["condY"] * 3,
            "source_record_id": ["24Q3:1:1:condY", "24Q3:2:1:condY", "24Q2:3:1:condY"],
        }
    )
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["case_count"] == "3"
    assert rows[0]["case_ids"] == "1|anon:24Q2:3:1:condY|anon:24Q3:2:1:condY"


def test_case_count_mixes_distinct_cases_and_anonymous_rows(disease_map: dict[str, dict[str, str]]) -> None:
    # Distinct non-empty primaryids dedup; null/empty primaryids each count as their own
    # observation (legacy _row{index} fallback) — the pair total is the sum of both.
    cases = pl.DataFrame({"primaryid": ["1", "", None, "1", "2"], "drugname": ["DrugX"] * 5, "indication": ["condY"] * 5})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["case_count"] == "4"  # distinct {1, 2} + 2 anonymous rows
    assert rows[0]["case_ids"] == "1|2|anon:row:DrugX:condY:0|anon:row:DrugX:condY:1"


def test_wordings_resolving_to_one_object_merge_into_a_single_row() -> None:
    """Duplicate fold: one edge-identity key, exact merged case count, unioned evidence.

    Two raw indication wordings (case variants) hit the same dictionary key, so they share the
    edge-identity key ``(subject_text, object_text)`` — the production ``uuid-fields-not-a-key``
    collision shape (CHEBI:62088 applied_to_treat HP:0012531). Resolution-first aggregation
    merges them into ONE row: ``case_count`` is the exact distinct-case count across BOTH
    wordings (case 1, reported under each wording, counts once — summing per-wording counts
    would double it), and the provenance columns are the deduplicated, sorted, pipe-joined
    union of the merged wordings' evidence.
    """
    disease_map = {"Asthma": {"curie": "MONDO:0004979", "name": "asthma", "category": "Disease"}}
    cases = pl.DataFrame(
        {
            "drugname": ["DrugX", "DrugX", "DrugX"],
            "indication": ["Asthma", "ASTHMA", "ASTHMA"],
            "primaryid": ["1", "1", "2"],
            "nda": ["17977", "125514", "17977"],
            "nda_raw": ["017977", "125514", "017977"],
            "quarter": ["24Q3", "24Q2", "24Q3"],
            "source_record_id": ["24Q3:1:1:Asthma", "24Q2:1:1:ASTHMA", "24Q3:2:1:ASTHMA"],
        }
    )
    index = FDAApprovalIndex({"17977": ("NDA017977",), "125514": ("BLA125514",)})
    rows = build_observed_use_rows(
        cases,
        disease_map,
        approved_pairs=set(),
        approvals=index,
        faers_quarter_urls={"24Q3": "https://example.test/faers-24q3.zip", "24Q2": "https://example.test/faers-24q2.zip"},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["object_text"] == "Asthma"
    assert row["object_curie"] == "MONDO:0004979"
    assert row["case_count"] == "2"  # distinct {1, 2} — case 1 counted once across both wordings
    assert row["case_ids"] == "1|2"  # the merge-exact token set Tablassert unions on collision
    assert row["FDA_regulatory_approvals"] == "BLA125514|NDA017977"
    assert row["supporting_faers_records"] == "24Q2:1:1:ASTHMA|24Q3:1:1:Asthma|24Q3:2:1:ASTHMA"
    assert row["supporting_faers_urls"] == "https://example.test/faers-24q2.zip|https://example.test/faers-24q3.zip"


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
    # No approved-treats table passed (degraded mode) -> every row is ``not_provided``.
    rows = build_observed_use_rows(find_faers_cases(faers_refs), disease_map)
    assert rows
    for row in rows:
        # biolink-valid ClinicalApprovalStatusEnum member: with no approved-treats table to
        # cross-reference, the pair's approval status is unknown (never the legacy
        # ``observed_use`` label, which is not an enum member and would fail validation
        # now that Tablassert >= 8.2 emits the field first-class).
        assert row["clinical_approval_status"] == "not_provided"
        assert row["knowledge_level"] == "statistical_association"
        assert row["agent_type"] == "manual_validation_of_automated_agent"
        assert row["primary_knowledge_source"] == "infores:multiomics-drugapprovals"
        assert row["upstream_resource_ids"] == "infores:faers|infores:dailymed"
        assert row["subject_curie"] == ""  # FAERS provides no drug id here (text-first)


# --- clinical_approval_status cross-reference with the approved-treats table -----


def test_pair_with_treats_counterpart_is_approved_for_condition(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame({"drugname": ["Examplestatin", "Advil"], "indication": ["hypercholesterolemia", "headache"]})
    rows = build_observed_use_rows(cases, disease_map, {("examplestatin", "hypercholesterolemia")})
    by_subject = {r["subject_text"]: r for r in rows}
    assert by_subject["Examplestatin"]["clinical_approval_status"] == "approved_for_condition"
    # No treats counterpart for (Advil, headache) -> the legacy off-label signal.
    assert by_subject["Advil"]["clinical_approval_status"] == "off_label_use"


def test_pair_matching_is_case_and_punctuation_insensitive() -> None:
    # FAERS casing/punctuation differs from the approved-treats text; normalized matching still pairs them.
    cases = pl.DataFrame({"drugname": ["EXAMPLESTATIN"], "indication": ["Type-2 Diabetes"]})
    rows = build_observed_use_rows(cases, {}, {("examplestatin", "type 2 diabetes")})
    assert rows[0]["clinical_approval_status"] == "approved_for_condition"


def test_approved_pair_index_normalizes_and_skips_incomplete_rows() -> None:
    frame = pl.DataFrame({"subject_text": ["Examplestatin", "", "DrugY"], "object_text": ["Hypercholesterolemia", "pain", ""]})
    assert _approved_pair_index(frame) == {("examplestatin", "hypercholesterolemia")}


def test_shaper_reads_approved_treats_table_for_status(faers_refs: list[ArtifactRef], ctx: TaskContext, tmp_path: Path) -> None:
    # The produced approved_treats_assertions.tsv, wired in as an input, drives the status.
    columns = schemas.columns_for("approved_treats_assertions")
    approved_row = dict.fromkeys(columns, "")
    approved_row.update({"subject_text": "Examplestatin", "object_text": "hypercholesterolemia"})
    approved_path = tmp_path / "approved_treats_assertions.tsv"
    schemas.write_tsv(pl.DataFrame([approved_row], schema=columns), approved_path)
    approved_ref = ArtifactRef(uri=approved_path, blake3="b3:" + "4" * 64, media_type=schemas.TSV_MEDIA_TYPE)

    refs = ObservedUsesShaper().transform([*faers_refs, approved_ref], ctx)
    assert len(refs) == 1
    status = {rec["subject_text"]: rec["clinical_approval_status"] for rec in schemas.read_table(refs[0].uri).iter_rows(named=True)}
    # Examplestatin matches the approved pair; Advil (brand name vs the DailyMed ingredient text)
    # and Placebo (no treats row) read as off-label — the documented name-variant limitation.
    assert status == {"Examplestatin": "approved_for_condition", "Advil": "off_label_use", "Placebo": "off_label_use"}


def test_rows_are_deterministically_ordered(faers_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    cases = find_faers_cases(faers_refs)
    first = build_observed_use_rows(cases, disease_map)
    second = build_observed_use_rows(cases, disease_map)
    assert first == second
    keys = [(r["subject_text"], r["object_text"]) for r in first]
    assert keys == sorted(keys)
    # The carrier invariant: every row's token count equals its case_count exactly.
    for row in first:
        assert len(row["case_ids"].split("|")) == int(row["case_count"])


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
        # MedDRA product-issue PTs that slip past the bare literals (the ASCIMINIB edge):
        "Contraindicated product administered",
        "Contraindicated product prescribed",
        "Product administered to patient of inappropriate age",
        "Product prescribed at wrong time",
        "Product dispensed to wrong patient",
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
