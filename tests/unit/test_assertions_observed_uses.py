"""Unit tests for FAERS observed-use (applied_to_treat) aggregation (Milestone 5).

Covers distinct-case case-count aggregation, the preserved FAERS label/status behavior, object
resolution via the lexical baseline, provenance columns, determinism, empty inputs, and the
end-to-end shaper TSV output.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.assertions.evidence import FDAApprovalIndex, find_faers_cases
from dakp_pipeline.assertions.observed_uses import (
    ApprovedTreatsIndex,
    ObservedUsesShaper,
    _coerce_approved_index,
    _proper_spans,
    build_observed_use_rows,
    is_non_disease_indication,
)
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext


def test_number_of_cases_aggregates_distinct_cases(disease_map: dict[str, dict[str, str]]) -> None:
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
    counts = {(r["subject_text"], r["object_text"]): r["number_of_cases"] for r in rows}
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
        approvals=FDAApprovalIndex({"17977": ("NDA017977",)}),
        faers_quarter_urls={"24Q3": "https://example.test/faers-24q3.zip", "24Q2": "https://example.test/faers-24q2.zip"},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["number_of_cases"] == "2"
    # Both FAERS spellings of the number resolve to the one FDA display form.
    assert row["FDA_regulatory_approvals"] == "NDA017977"
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


def test_observed_use_drops_faers_application_numbers_no_register_knows(disease_map: dict[str, dict[str, str]]) -> None:
    """FAERS ``nda_num`` is reporter free text, so its junk must not ride ``regulatory_approvals``.

    The reported bug: the v1.16.0 release shipped ``regulatory_approvals: ["999999"]`` on 2,748
    edges (BENADRYL -> Somnolence among them) and ``["99"]`` on 6,091 more, alongside concatenated
    free text like ``501930535019305``. ``999999``/``99`` are placeholder spellings of "unknown",
    and a bare number with no application type resolves to nothing. Dropping the VALUE keeps the
    edge and its FAERS provenance intact, which is what the legacy pipeline threw away.
    """
    cases = pl.DataFrame(
        {
            "drugname": ["BENADRYL"] * 5,
            "indication": ["Somnolence"] * 5,
            "primaryid": ["1", "2", "3", "4", "5"],
            "nda": ["999999", "99", "501930535019305", "17977", "17977"],
            "nda_raw": ["999999", "0099", "501930535019305", "017977", "17977"],
            "quarter": ["24Q3"] * 5,
            "source_record_id": ["a", "b", "c", "d", "e"],
        }
    )
    index = FDAApprovalIndex({"17977": ("NDA017977",)})
    rows = build_observed_use_rows(
        cases, disease_map, approved_pairs=set(), approvals=index, faers_quarter_urls={"24Q3": "https://example.test/faers-24q3.zip"}
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["FDA_regulatory_approvals"] == "NDA017977"  # only the resolvable number survives
    assert row["number_of_cases"] == "5"  # dropping a value never drops a case or the edge
    assert row["supporting_faers_urls"] == "https://example.test/faers-24q3.zip"


def test_number_of_cases_falls_back_to_rows_without_primaryid(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame({"drugname": ["DrugX", "DrugX"], "indication": ["condY", "condY"]})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["number_of_cases"] == "2"  # no primaryid column -> row count
    # Id-less rows still carry one token each (per-group synthetic pads), keeping
    # len(case_ids) == number_of_cases so the Tablassert merge union stays exact.
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
    assert rows[0]["number_of_cases"] == "3"
    assert rows[0]["case_ids"] == "1|anon:24Q2:3:1:condY|anon:24Q3:2:1:condY"


def test_number_of_cases_mixes_distinct_cases_and_anonymous_rows(disease_map: dict[str, dict[str, str]]) -> None:
    # Distinct non-empty primaryids dedup; null/empty primaryids each count as their own
    # observation (legacy _row{index} fallback) — the pair total is the sum of both.
    cases = pl.DataFrame({"primaryid": ["1", "", None, "1", "2"], "drugname": ["DrugX"] * 5, "indication": ["condY"] * 5})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["number_of_cases"] == "4"  # distinct {1, 2} + 2 anonymous rows
    assert rows[0]["case_ids"] == "1|2|anon:row:DrugX:condY:0|anon:row:DrugX:condY:1"


def test_wordings_resolving_to_one_object_merge_into_a_single_row() -> None:
    """Duplicate fold: one edge-identity key, exact merged case count, unioned evidence.

    Two raw indication wordings (case variants) hit the same dictionary key, so they share the
    edge-identity key ``(subject_text, object_text)`` — the production ``uuid-fields-not-a-key``
    collision shape (CHEBI:62088 applied_to_treat HP:0012531). Resolution-first aggregation
    merges them into ONE row: ``number_of_cases`` is the exact distinct-case count across BOTH
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
    assert row["number_of_cases"] == "2"  # distinct {1, 2} — case 1 counted once across both wordings
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
    assert by_subject["Examplestatin"]["number_of_cases"] == "1"
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
        assert row["primary_knowledge_source"] == "infores:drugapprovals-kp"
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


def test_status_is_canonical_across_drugname_spelling_variants(disease_map: dict[str, dict[str, str]]) -> None:
    # v1.13.0 foldreport: 3,283 merged edges carried conflicting clinical_approval_status
    # scalars because cross-spelling drugname variants derived different statuses. The pair
    # key now runs the SAME textnorm chain on both sides, so the brand alias (Xefo ->
    # Lornoxicam), a dosage-junk variant, and the plain generic ALL derive the ONE status of
    # the approved pair -- Tablassert's first-wins merge then sees no conflict.
    cases = pl.DataFrame({"drugname": ["XEFO", "Xefo 90 MG TABLET", "LORNOXICAM"], "indication": ["arthritis", "arthritis", "arthritis"]})
    rows = build_observed_use_rows(cases, disease_map, {("lornoxicam", "arthritis")})
    assert {r["clinical_approval_status"] for r in rows} == {"approved_for_condition"}
    # A drug with no treats counterpart stays off_label in every variant.
    cases2 = pl.DataFrame({"drugname": ["XEFO", "LORNOXICAM"], "indication": ["dizziness", "dizziness"]})
    rows2 = build_observed_use_rows(cases2, disease_map, {("lornoxicam", "arthritis")})
    assert {r["clinical_approval_status"] for r in rows2} == {"off_label_use"}


def test_combo_drugname_stays_one_edge_with_exact_case_count(disease_map: dict[str, dict[str, str]]) -> None:
    # US-004 invariant: one FAERS case contributes to ONE product edge. A multi-ingredient
    # drugname canonicalizes to its mixture-level subject_text and the distinct-case count
    # passes through unchanged (components are never split into per-ingredient edges).
    cases = pl.DataFrame(
        {
            "drugname": [".ALPHA.-TOCOPHEROL ACETATE\\ASCORBIC ACID", ".ALPHA.-TOCOPHEROL ACETATE\\ASCORBIC ACID"],
            "indication": ["fatigue", "fatigue"],
            "primaryid": ["1001", "1002"],
        }
    )
    rows = build_observed_use_rows(cases, disease_map, approved_pairs=None)
    assert len(rows) == 1
    assert rows[0]["subject_text"] == "alpha-TOCOPHEROL ACETATE / ASCORBIC ACID"
    assert rows[0]["number_of_cases"] == "2"


def test_approved_treats_index_runs_the_full_textnorm_chain() -> None:
    # The index is built from approved-treats subject_text, which can itself carry FAERS
    # fallback junk (brand aliases, dosage tails); both sides must canonicalize identically
    # or the lookup answers asymmetrically.
    frame = pl.DataFrame({"subject_text": ["XEFO 8MG", "Examplestatin"], "object_text": ["Arthritis", "Pain"]})
    assert ApprovedTreatsIndex.from_frame(frame).pairs == {("lornoxicam", "arthritis"), ("examplestatin", "pain")}


def test_approved_treats_index_normalizes_and_skips_incomplete_rows() -> None:
    frame = pl.DataFrame({"subject_text": ["Examplestatin", "", "DrugY"], "object_text": ["Hypercholesterolemia", "pain", ""]})
    index = ApprovedTreatsIndex.from_frame(frame)
    assert index.pairs == {("examplestatin", "hypercholesterolemia")}
    # An empty object means the row approves no condition, so it indexes no application either:
    # an application key with nothing behind it could only produce false approvals.
    assert index.objects_by_approval == {}


def test_approved_treats_index_maps_applications_to_their_objects() -> None:
    """The application number is the product identity both tables share.

    ``FDA_regulatory_approvals`` is a pipe-joined multivalued cell, so one row can index several
    applications for the same object; the normalized object key is what the observed-uses side
    compares against, which is what makes the brand/ingredient spelling difference irrelevant.
    """
    frame = pl.DataFrame(
        {
            "subject_text": ["LEUPROLIDE", "LEUPROLIDE", "DENOSUMAB"],
            "object_text": ["Prostate cancer", "Endometriosis", ""],
            "FDA_regulatory_approvals": ["NDA021343|NDA021379", "NDA020708", "BLA125320"],
        }
    )
    objects = ApprovedTreatsIndex.from_frame(frame).objects_by_approval
    assert objects["NDA021343"] == frozenset({"prostate cancer"})
    assert objects["NDA021379"] == frozenset({"prostate cancer"})
    assert objects["NDA020708"] == frozenset({"endometriosis"})
    assert "BLA125320" not in objects  # empty object_text indexes nothing


def test_status_uses_fda_application_identity_when_subject_text_differs() -> None:
    """The reported ELIGARD bug: a brand subject never text-matches the ingredient subject.

    Production v1.16.0 shipped ``ELIGARD | Prostate cancer | off_label_use`` with 35,977 cases
    even though approved_treats holds ``LEUPROLIDE | Prostate cancer`` for the SAME applications
    (NDA021343 and friends), and the FDA label for NDA021343 reads "ELIGARD is indicated for the
    treatment of advanced prostate cancer". The application number both tables already carry is
    the identity that answers; the drug spelling must not have to.
    """
    cases = pl.DataFrame({"drugname": ["ELIGARD"], "indication": ["Prostate cancer"], "primaryid": ["1001"], "nda": ["21343"], "nda_raw": ["021343"]})
    approved = ApprovedTreatsIndex.from_frame(
        pl.DataFrame({"subject_text": ["LEUPROLIDE"], "object_text": ["Prostate cancer"], "FDA_regulatory_approvals": ["NDA021343|NDA021379"]})
    )
    index = FDAApprovalIndex({"21343": ("NDA021343",)})
    rows = build_observed_use_rows(cases, {}, approved, approvals=index)
    assert len(rows) == 1
    assert rows[0]["subject_text"] == "ELIGARD"  # the subject text is untouched, only the status
    assert rows[0]["FDA_regulatory_approvals"] == "NDA021343"
    assert rows[0]["clinical_approval_status"] == "approved_for_condition"


def test_status_stays_off_label_when_no_application_matches() -> None:
    """Negative tests: the application rule must not approve on a partial match.

    A shared application with a different condition, a shared condition under a different
    application, and a report carrying no application number at all all stay ``off_label_use``
    (the last one has only the text rule left, and the brand text does not match the ingredient).
    """
    approved = ApprovedTreatsIndex.from_frame(
        pl.DataFrame(
            {
                "subject_text": ["LEUPROLIDE", "LEUPROLIDE"],
                "object_text": ["Prostate cancer", "Endometriosis"],
                "FDA_regulatory_approvals": ["NDA021343", "NDA020708"],
            }
        )
    )
    cases = pl.DataFrame(
        {
            "drugname": ["ELIGARD", "PROCREN", "ELIGARD"],
            "indication": ["Endometriosis", "Prostate cancer", "Prostate cancer"],
            "primaryid": ["1", "2", "3"],
            "nda": ["21343", "20708", ""],
            "nda_raw": ["021343", "020708", ""],
        }
    )
    index = FDAApprovalIndex({"21343": ("NDA021343",), "20708": ("NDA020708",)})
    rows = build_observed_use_rows(cases, {}, approved, approvals=index)
    statuses = {(r["subject_text"], r["object_text"]): r["clinical_approval_status"] for r in rows}
    assert statuses == {
        ("ELIGARD", "Endometriosis"): "off_label_use",  # right application, wrong condition
        ("PROCREN", "Prostate cancer"): "off_label_use",  # right condition, wrong application
        ("ELIGARD", "Prostate cancer"): "off_label_use",  # no application number to match on
    }


def _eligard_index() -> ApprovedTreatsIndex:
    """The real approved_treats row behind the reported bug (LEUPROLIDE / Prostate cancer).

    Subject is the DailyMed ingredient text and the approval cell is the real one from the
    v1.16.0 production table, so the fixture reproduces the brand-vs-ingredient spelling gap
    rather than a convenient one.
    """
    return ApprovedTreatsIndex.from_frame(
        pl.DataFrame(
            {
                "subject_text": ["LEUPROLIDE"],
                "object_text": ["Prostate cancer"],
                "FDA_regulatory_approvals": ["NDA019732|NDA020517|NDA021343|NDA021379|NDA021488|NDA021731|NDA205054|NDA211488"],
            }
        )
    )


def test_reported_eligard_prostate_cancer_stage_iv_edge_is_approved() -> None:
    """The exact production edge that was wrong: KGX id 32c1b660-8fcf-3330-8576-21ec0ac0fab3.

    ``DRUG_APPROVALS_KP_1.16.0.edges.ndjson`` shipped ELIGARD applied_to_treat
    "Prostate cancer stage IV" (55 cases) as ``off_label_use`` while carrying
    NDA021343|NDA021379|NDA021488|NDA021731 on the same edge. The FDA label for those four
    applications reads "ELIGARD is indicated for the treatment of advanced prostate cancer",
    and stage IV IS advanced prostate cancer, so the observation is on-label.
    """
    approvals = {"21343": "NDA021343", "21379": "NDA021379", "21488": "NDA021488", "21731": "NDA021731"}
    cases = pl.DataFrame(
        {
            "drugname": ["ELIGARD"] * 4,
            "indication": ["Prostate cancer stage IV"] * 4,
            "primaryid": ["1", "2", "3", "4"],
            "nda": sorted(approvals),
            "nda_raw": ["021343", "021379", "021488", "021731"],
        }
    )
    index = FDAApprovalIndex({norm: (display,) for norm, display in approvals.items()})
    rows = build_observed_use_rows(cases, {}, _eligard_index(), approvals=index)
    assert len(rows) == 1
    assert rows[0]["FDA_regulatory_approvals"] == "NDA021343|NDA021379|NDA021488|NDA021731"
    assert rows[0]["clinical_approval_status"] == "approved_for_condition"
    # The status is the only thing that changes: the object text stays the specific FAERS wording.
    assert rows[0]["object_text"] == "Prostate cancer stage IV"


def test_status_bridges_object_granularity_within_one_application() -> None:
    """The general approved object covers the more specific FAERS wording, under one application.

    FAERS reports stage and laterality wording the label never spells out; the label's general
    term is the approval, so the specific report of the SAME product is on-label.
    """
    index = FDAApprovalIndex({"21343": ("NDA021343",)})
    cases = pl.DataFrame(
        {
            "drugname": ["ELIGARD", "ELIGARD", "ELIGARD"],
            "indication": ["Prostate cancer metastatic", "Hormone refractory prostate cancer", "Prostate cancer"],
            "primaryid": ["1", "2", "3"],
            "nda": ["21343"] * 3,
            "nda_raw": ["021343"] * 3,
        }
    )
    rows = build_observed_use_rows(cases, {}, _eligard_index(), approvals=index)
    statuses = {r["object_text"]: r["clinical_approval_status"] for r in rows}
    assert statuses == {
        "Prostate cancer metastatic": "approved_for_condition",  # trailing qualifier
        "Hormone refractory prostate cancer": "approved_for_condition",  # leading qualifiers
        "Prostate cancer": "approved_for_condition",  # exact, the R1 path
    }


def test_containment_never_approves_the_reverse_direction() -> None:
    """An approved object MORE specific than the observation does not approve it.

    A label for stage IV disease says nothing about earlier stages, so the general report stays
    off-label. Structurally guaranteed: a longer key cannot be a token span of a shorter one.
    """
    approved = ApprovedTreatsIndex.from_frame(
        pl.DataFrame({"subject_text": ["Xprostatin"], "object_text": ["Prostate cancer stage IV"], "FDA_regulatory_approvals": ["NDA021343"]})
    )
    cases = pl.DataFrame({"drugname": ["XPROSTATIN"], "indication": ["Prostate cancer"], "primaryid": ["1"], "nda": ["21343"], "nda_raw": ["021343"]})
    index = FDAApprovalIndex({"21343": ("NDA021343",)})
    rows = build_observed_use_rows(cases, {}, approved, approvals=index)
    assert rows[0]["clinical_approval_status"] == "off_label_use"


def test_containment_is_whole_word_only() -> None:
    """A partial-word overlap is not a match: ``pain`` must not approve ``painful swelling``.

    Span enumeration over the normalized token list gives this for free, and it is the property
    that keeps a short approved object from approving everything that merely contains its
    letters.
    """
    approved = ApprovedTreatsIndex.from_frame(
        pl.DataFrame({"subject_text": ["Analgetol"], "object_text": ["Pain"], "FDA_regulatory_approvals": ["NDA020708"]})
    )
    cases = pl.DataFrame(
        {
            "drugname": ["ANALGETOL", "ANALGETOL"],
            "indication": ["Painful swelling", "Chronic pain"],
            "primaryid": ["1", "2"],
            "nda": ["20708"] * 2,
            "nda_raw": ["020708"] * 2,
        }
    )
    index = FDAApprovalIndex({"20708": ("NDA020708",)})
    rows = build_observed_use_rows(cases, {}, approved, approvals=index)
    statuses = {r["object_text"]: r["clinical_approval_status"] for r in rows}
    assert statuses == {"Painful swelling": "off_label_use", "Chronic pain": "approved_for_condition"}


def test_containment_needs_an_application_number() -> None:
    """Without an application there is no product identity, so only the text rule remains.

    The granularity bridge is deliberately scoped to one application: extending it across drugs
    would let any drug's general approval cover any other drug's specific report.
    """
    cases = pl.DataFrame({"drugname": ["LEUPROLIDE"], "indication": ["Prostate cancer stage IV"], "primaryid": ["1"]})
    rows = build_observed_use_rows(cases, {}, _eligard_index())
    assert rows[0]["FDA_regulatory_approvals"] == ""
    assert rows[0]["clinical_approval_status"] == "off_label_use"


def test_proper_spans_cover_every_contiguous_subphrase_but_not_the_whole_key() -> None:
    """The span set IS the whole-word containment relation, minus the already-checked equality."""
    assert _proper_spans("prostate cancer stage iv") == (
        "prostate",
        "prostate cancer",
        "prostate cancer stage",
        "cancer",
        "cancer stage",
        "cancer stage iv",
        "stage",
        "stage iv",
        "iv",
    )
    assert _proper_spans("cancer") == ()  # one token: only the exact match can approve
    assert _proper_spans("") == ()
    # The property the enumeration exists for: every span is a whole-word occurrence of the key,
    # and the key itself is excluded because equality is checked before the bridge runs.
    key = "hormone refractory prostate cancer"
    spans = _proper_spans(key)
    assert key not in spans
    assert all(f" {span} " in f" {key} " for span in spans)
    assert len(spans) == len(set(spans)) == len(key.split()) * (len(key.split()) + 1) // 2 - 1


def test_all_digit_application_keys_are_not_identities() -> None:
    """A reporter typo must not become a product identity.

    ``FDAApprovalIndex.expand`` falls back to ``<prefix><digits>`` for a number no register
    knows, and FAERS numbers carry no prefix, so tokens like ``99`` reach both tables. Indexing
    them would let two unrelated products that happened to report the same typo cross-approve.
    """
    approved = ApprovedTreatsIndex.from_frame(
        pl.DataFrame({"subject_text": ["Leuprolide"], "object_text": ["Prostate cancer"], "FDA_regulatory_approvals": ["99|02248240|NDA021343"]})
    )
    assert "NDA021343" in approved.objects_by_approval
    assert "99" not in approved.objects_by_approval
    assert "02248240" not in approved.objects_by_approval

    # The typo-citing report stays off-label; the report citing the real application is approved.
    # Two drugnames, because one (drugname, object) group merges into a single row and would
    # union the two application numbers.
    cases = pl.DataFrame(
        {
            "drugname": ["Mysterydrug", "Realdrug"],
            "indication": ["Prostate cancer", "Prostate cancer"],
            "primaryid": ["1", "2"],
            "nda": ["99", "21343"],
            "nda_raw": ["99", "021343"],
        }
    )
    index = FDAApprovalIndex({"21343": ("NDA021343",)})
    rows = build_observed_use_rows(cases, {}, approved, approvals=index)
    by_approvals = {r["FDA_regulatory_approvals"]: r["clinical_approval_status"] for r in rows}
    assert by_approvals == {"99": "off_label_use", "NDA021343": "approved_for_condition"}


def test_the_application_index_is_read_only() -> None:
    """One index is shared by every row of a 1.6M-row table, so a caller cannot corrupt it."""
    index = ApprovedTreatsIndex.from_frame(
        pl.DataFrame({"subject_text": ["Leuprolide"], "object_text": ["Prostate cancer"], "FDA_regulatory_approvals": ["NDA021343"]})
    )
    with pytest.raises(TypeError):
        index.objects_by_approval["NDA999999"] = frozenset({"headache"})  # type: ignore[index]
    assert ApprovedTreatsIndex().objects_by_approval == {}


def test_a_bare_pair_iterable_drops_empty_keys() -> None:
    """An empty key would approve every row whose subject or object normalizes to nothing.

    ``from_frame`` already skips incomplete rows; the compatibility path for a bare iterable of
    pairs has to hold the same line or the two ways of building an index disagree.
    """
    empty = _coerce_approved_index({("", "pain"), ("drug", ""), ("", "")})
    assert empty is not None
    assert empty.pairs == frozenset()
    text_only = _coerce_approved_index({("drug", "pain")})
    assert text_only is not None
    assert text_only.pairs == {("drug", "pain")}
    assert _coerce_approved_index(None) is None
    existing = ApprovedTreatsIndex(pairs=frozenset({("drug", "pain")}))
    assert _coerce_approved_index(existing) is existing  # an index passes through untouched


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
    # Examplestatin matches the approved pair by normalized text. Advil and Placebo read as
    # off-label because this approved fixture row carries NO FDA_regulatory_approvals, so only
    # the text key exists and the brand subject does not match the ingredient text. The same
    # fixture WITH the application number is the integration case
    # (test_semantic_equivalence.py::test_applied_to_treat_carries_the_off_label_signal), where
    # Advil is approved through NDA017977.
    assert status == {"Examplestatin": "approved_for_condition", "Advil": "off_label_use", "Placebo": "off_label_use"}


def test_rows_are_deterministically_ordered(faers_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    cases = find_faers_cases(faers_refs)
    first = build_observed_use_rows(cases, disease_map)
    second = build_observed_use_rows(cases, disease_map)
    assert first == second
    keys = [(r["subject_text"], r["object_text"]) for r in first]
    assert keys == sorted(keys)
    # The carrier invariant: every row's token count equals its number_of_cases exactly.
    for row in first:
        assert len(row["case_ids"].split("|")) == int(row["number_of_cases"])


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
