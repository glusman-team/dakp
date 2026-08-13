"""Unit tests for legacy-informed regression guardrails (Milestone 8).

The guardrails assert family/provenance/label invariants — not edge-for-edge equality. Positive
tests confirm valid rows (and the *real* assertion tables produced by the shapers) pass; negative
tests confirm each broken invariant is reported specifically, aggregated per (family, invariant).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.translator import EXPECTED_FAMILIES, RegressionReport, check_assertion_tables, check_rows

_DAKP = "infores:multiomics-drugapprovals"


def _row(
    predicate: str,
    *,
    status: str = "",
    knowledge_level: str = "knowledge_assertion",
    primary: str = _DAKP,
    upstream: str = "infores:dailymed|infores:faers",
) -> dict[str, str]:
    return {
        "predicate": predicate,
        "clinical_approval_status": status,
        "knowledge_level": knowledge_level,
        "primary_knowledge_source": primary,
        "upstream_resource_ids": upstream,
    }


def _valid_rows() -> list[dict[str, str]]:
    return [
        _row("biolink:treats", status="approved_for_condition", upstream="infores:dailymed|infores:faers"),
        _row("biolink:applied_to_treat", status="not_provided", knowledge_level="statistical_association", upstream="infores:faers|infores:dailymed"),
        _row("biolink:contraindicated_in", upstream="infores:dailymed"),
    ]


def _violations(report: RegressionReport) -> set[tuple[str, str]]:
    return {(violation.family, violation.invariant) for violation in report.violations}


# --- positive --------------------------------------------------------------------


def test_valid_rows_pass_all_invariants() -> None:
    report = check_rows(_valid_rows())
    assert report.ok is True
    assert report.violations == []
    assert report.families_seen == sorted(EXPECTED_FAMILIES)
    assert report.row_count == 3


def test_real_assertion_tables_preserve_invariants(
    dailymed_refs: list[ArtifactRef], drugsfda_refs: list[ArtifactRef], faers_refs: list[ArtifactRef], ema_refs: list[ArtifactRef], ctx: TaskContext
) -> None:
    approved = approved_treats.transform([*dailymed_refs, *drugsfda_refs, *ema_refs], ctx)
    uses = observed_uses.transform([*faers_refs, *dailymed_refs], ctx)
    contra = contraindications.transform([*dailymed_refs], ctx)

    report = check_assertion_tables([*approved, *uses, *contra])
    assert report.violations == []
    assert report.ok is True
    assert report.families_seen == sorted(EXPECTED_FAMILIES)
    assert report.row_count > 0


def test_absent_family_is_not_a_violation() -> None:
    # A build with only treats rows is still regression-clean; absence is coverage, not regression.
    report = check_rows([_row("biolink:treats", status="approved_for_condition")])
    assert report.ok is True
    assert report.families_seen == ["biolink:treats"]


def test_ema_upstream_satisfies_the_treats_invariant() -> None:
    # EMA-derived treats rows carry infores:ema instead of the FDA dailymed|faers chain.
    report = check_rows([_row("biolink:treats", status="approved_for_condition", upstream="infores:ema")])
    assert report.ok is True
    assert report.violations == []


def test_epar_upstream_satisfies_the_treats_invariant() -> None:
    # EPAR indication-mined treats rows carry infores:epar instead of the FDA dailymed|faers chain.
    report = check_rows([_row("biolink:treats", status="approved_for_condition", upstream="infores:epar")])
    assert report.ok is True
    assert report.violations == []


def test_non_family_rows_are_ignored_but_counted() -> None:
    report = check_rows([{"predicate": "biolink:causes", "primary_knowledge_source": "infores:other"}])
    assert report.ok is True
    assert report.families_seen == []
    assert report.row_count == 1


# --- negative: each invariant is reported specifically ---------------------------


def test_wrong_approval_status_reported() -> None:
    report = check_rows([_row("biolink:treats", status="off_label")])
    assert _violations(report) == {("biolink:treats", "clinical_approval_status")}


def test_missing_faers_upstream_reported() -> None:
    report = check_rows([_row("biolink:treats", status="approved_for_condition", upstream="infores:dailymed")])
    assert _violations(report) == {("biolink:treats", "upstream_provenance")}
    assert "infores:faers" in report.violations[0].message


def test_wrong_primary_source_reported() -> None:
    report = check_rows([_row("biolink:applied_to_treat", status="not_provided", knowledge_level="statistical_association", primary="infores:faers")])
    assert _violations(report) == {("biolink:applied_to_treat", "primary_knowledge_source")}


def test_contra_missing_dailymed_reported() -> None:
    report = check_rows([_row("biolink:contraindicated_in", upstream="")])
    assert _violations(report) == {("biolink:contraindicated_in", "upstream_provenance")}
    assert "infores:dailymed" in report.violations[0].message


def test_wrong_knowledge_level_reported() -> None:
    report = check_rows([_row("biolink:applied_to_treat", status="not_provided", knowledge_level="knowledge_assertion")])
    assert _violations(report) == {("biolink:applied_to_treat", "knowledge_level")}


def test_violations_aggregated_with_offending_count() -> None:
    rows = [_row("biolink:treats", status="off_label") for _ in range(3)]
    report = check_rows(rows)
    assert len(report.violations) == 1
    assert report.violations[0].message.startswith("3 row(s):")


def test_check_assertion_tables_reports_violations_read_from_disk(tmp_path: Path) -> None:
    # A treats row with a broken clinical_approval_status invariant, written as a real TSV.
    columns = schemas.columns_for("approved_treats_assertions")
    row = dict.fromkeys(columns, "")
    row.update(
        {
            "predicate": "biolink:treats",
            "clinical_approval_status": "wrong_status",
            "knowledge_level": "knowledge_assertion",
            "primary_knowledge_source": _DAKP,
            "upstream_resource_ids": "infores:dailymed|infores:faers",
        }
    )
    bad_path = tmp_path / "approved_treats_assertions.tsv"
    schemas.write_tsv(pl.DataFrame([row], schema=columns), bad_path)
    bad_ref = ArtifactRef(uri=bad_path, blake3="b3:" + "2" * 64, media_type=schemas.TSV_MEDIA_TYPE)

    report = check_assertion_tables([bad_ref])
    assert report.ok is False
    assert ("biolink:treats", "clinical_approval_status") in _violations(report)


def test_check_assertion_tables_without_family_rows_reports_empty_families(tmp_path: Path) -> None:
    columns = schemas.columns_for("faers_applied_to_treat_assertions")
    empty_path = tmp_path / "faers_applied_to_treat_assertions.tsv"
    schemas.write_tsv(pl.DataFrame(schema=columns), empty_path)
    ref = ArtifactRef(uri=empty_path, blake3="b3:" + "3" * 64, media_type=schemas.TSV_MEDIA_TYPE)

    report = check_assertion_tables([ref])
    assert report.ok is True
    assert report.families_seen == []
    assert report.row_count == 0
