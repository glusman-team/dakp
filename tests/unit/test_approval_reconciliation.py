"""Resolved approvals protect legacy pair semantics without weakening edge identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dakp_pipeline import approval_reconciliation as reconciliation
from dakp_pipeline.translator import INVALID_APPROVAL_STATUS, INVALID_OBSERVED_APPROVALS, validate_kgx


def _edge(predicate: str, subject: str = "CHEBI:1", obj: str = "MONDO:1", status: str = "off_label_use", **extras: object) -> dict[str, object]:
    return {"id": predicate + subject + obj, "subject": subject, "predicate": predicate, "object": obj, "clinical_approval_status": status, **extras}


def _write(path: Path, edges: list[dict[str, object]]) -> bytes:
    path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")
    return path.read_bytes()


@pytest.mark.parametrize("status", ["off_label_use", "not_provided", "", "approved_for_condition"])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_resolved_counterpart_promotes_without_changing_evidence(tmp_path: Path, status: str, reverse: bool) -> None:
    """Source order and synonyms must not label a resolved label-approved pair off-label."""
    observed = _edge(reconciliation.OBSERVED, status=status, number_of_cases=42, sources=[{"resource_id": "infores:faers"}])
    treated = _edge(reconciliation.TREATS, status=reconciliation.APPROVED, disease_context_qualifier="MONDO:2")
    edges = [observed, treated] if reverse else [treated, observed]
    path = tmp_path / "edges.ndjson"
    _write(path, edges)
    assert reconciliation.reconcile_approval_status(path) == int(status != reconciliation.APPROVED)
    result = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {**observed, "clinical_approval_status": reconciliation.APPROVED}
    assert result == ([expected, treated] if reverse else [treated, expected])
    assert reconciliation.reconcile_approval_status(path) == 0


def test_unrelated_pairs_and_preapproved_observations_remain_byte_identical(tmp_path: Path) -> None:
    """No ancestor, spelling or reverse inference may create approvals or demote FDA approval."""
    path = tmp_path / "edges.ndjson"
    original = _write(
        path,
        [
            _edge(reconciliation.TREATS),
            _edge(reconciliation.OBSERVED, obj="MONDO:2"),
            _edge(reconciliation.OBSERVED, subject="CHEBI:2"),
            _edge(reconciliation.OBSERVED, subject="CHEBI:3", status=reconciliation.APPROVED),
            _edge("biolink:contraindicated_in"),
            {"predicate": reconciliation.TREATS, "subject": "", "object": ""},
            {"predicate": reconciliation.OBSERVED},
        ],
    )
    assert reconciliation.reconcile_approval_status(path) == 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("endpoint", [[], {}, ["CHEBI:1"], 1, None, ""])
@pytest.mark.parametrize("field", ["subject", "object"])
def test_malformed_endpoints_reach_contract_diagnostics(tmp_path: Path, endpoint: object, field: str) -> None:
    """Bad scalar endpoints must not crash reconciliation ahead of the publish report."""
    path = tmp_path / "edges.ndjson"
    edges = [_edge(reconciliation.TREATS), _edge(reconciliation.OBSERVED)]
    edges[0][field] = endpoint
    edges[1][field] = endpoint
    original = _write(path, edges)
    assert reconciliation.reconcile_approval_status(path) == 0
    assert path.read_bytes() == original
    report = validate_kgx([], edges)
    assert any(problem.field == field and problem.code == "missing_edge_field" for problem in report.kgx_problems)


def test_empty_and_blank_streams_are_safe(tmp_path: Path) -> None:
    """Empty output and blank separator lines are not treated as edge records."""
    path = tmp_path / "edges.ndjson"
    for original in (b"", b"\n \n"):
        path.write_bytes(original)
        assert reconciliation.reconcile_approval_status(path) == 0
        assert path.read_bytes() == original


def test_parse_failure_preserves_the_original(tmp_path: Path) -> None:
    """A corrupt build cannot be partially repaired and published as a valid graph."""
    path = tmp_path / "edges.ndjson"
    path.write_text("{bad\n")
    with pytest.raises(json.JSONDecodeError):
        reconciliation.reconcile_approval_status(path)
    assert path.read_text() == "{bad\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["edges.ndjson"]


def test_replace_failure_preserves_the_original_and_cleans_temporary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed publication of the reconciled stream must leave a recoverable original."""
    path = tmp_path / "edges.ndjson"
    original = _write(path, [_edge(reconciliation.TREATS), _edge(reconciliation.OBSERVED)])

    def fail(*args: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(reconciliation.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        reconciliation.reconcile_approval_status(path)
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["edges.ndjson"]


def test_contract_rejects_overlap_until_reconciled() -> None:
    """The publish gate must catch off-label overlap even if repair integration regresses."""
    edges = [_edge(reconciliation.TREATS), _edge(reconciliation.OBSERVED)]
    problems = validate_kgx([], edges).kgx_problems
    assert [p.code for p in problems].count(INVALID_APPROVAL_STATUS) == 1
    edges[1]["clinical_approval_status"] = reconciliation.APPROVED
    assert INVALID_APPROVAL_STATUS not in {p.code for p in validate_kgx([], edges).kgx_problems}


@pytest.mark.parametrize("field", ["regulatory_approvals", "FDA_regulatory_approvals", "approval_ids", "approvals"])
@pytest.mark.parametrize("value", [["NDA022334"], [], None, ""])
@pytest.mark.parametrize("status", ["off_label_use", "approved_for_condition", "not_provided"])
@pytest.mark.parametrize("predicate", [reconciliation.OBSERVED, f" {reconciliation.OBSERVED} "])
def test_contract_rejects_approval_fields_on_observed_edges(field: str, value: object, status: str, predicate: str) -> None:
    """Neither status nor an empty value permits an approval slot on an observation."""
    edge = _edge(predicate, status=status)
    edge[field] = value
    problems = [p for p in validate_kgx([], [edge]).kgx_problems if p.code == INVALID_OBSERVED_APPROVALS]
    assert len(problems) == 1
    assert problems[0].field == field
    edge["predicate"] = reconciliation.TREATS
    assert INVALID_OBSERVED_APPROVALS not in {p.code for p in validate_kgx([], [edge]).kgx_problems}
