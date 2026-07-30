"""Unit tests for KGX/Translator contract validation (Milestone 8).

Validates small fixture KGX (nodes/edges JSONL) mirroring what Tablassert would emit:
a valid fixture covering the three DAKP edge families passes cleanly, and a broken fixture
reports each defect as a specific structured :class:`ContractProblem` code. Also guards the
legacy assertion-table :func:`validate` contract used by the runner/DAG.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from dakp_pipeline.translator import contract
from dakp_pipeline.translator.contract import (
    DUPLICATE_NODE_ID,
    INCOMPATIBLE_OBJECT_CATEGORY,
    INCOMPATIBLE_SUBJECT_CATEGORY,
    INVALID_NODE_CATEGORY,
    INVALID_PREDICATE,
    MISSING_EDGE_FIELD,
    MISSING_NODE_FIELD,
    MISSING_NODE_REFERENCE,
    MISSING_PROVENANCE,
    ContractReport,
    read_kgx_jsonl,
    validate,
    validate_kgx,
)

KGX_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "kgx"


def _load(name: str) -> list[dict[str, object]]:
    return read_kgx_jsonl(KGX_ROOT / name)


def _codes(report: ContractReport) -> set[str]:
    return {problem.code for problem in report.kgx_problems}


def _by_entity(report: ContractReport, entity_id: str) -> list[contract.ContractProblem]:
    return [problem for problem in report.kgx_problems if problem.entity_id == entity_id]


# --- positive: the valid fixture passes cleanly ----------------------------------


def test_valid_fixture_passes_with_no_problems() -> None:
    report = validate_kgx(_load("nodes.jsonl"), _load("edges.jsonl"))
    assert report.ok is True
    assert report.kgx_problems == []
    assert report.problems == []


def test_valid_fixture_exercises_all_three_edge_families() -> None:
    edges = _load("edges.jsonl")
    predicates = {edge["predicate"] for edge in edges}
    assert predicates == {"biolink:treats", "biolink:applied_to_treat", "biolink:contraindicated_in"}
    assert validate_kgx(_load("nodes.jsonl"), edges).ok is True


def test_single_inline_valid_edge_passes() -> None:
    nodes = [{"id": "CHEBI:1", "name": "drug", "category": ["biolink:Drug"]}, {"id": "MONDO:1", "name": "disease", "category": ["biolink:Disease"]}]
    edges = [
        {
            "id": "e1",
            "subject": "CHEBI:1",
            "predicate": "biolink:treats",
            "object": "MONDO:1",
            "category": ["biolink:EntityToDiseaseAssociation"],
            "knowledge_level": "knowledge_assertion",
            "agent_type": "manual_validation_of_automated_agent",
            "primary_knowledge_source": "infores:multiomics-drugapprovals",
            "sources": [{"resource_id": "infores:multiomics-drugapprovals", "upstream_resource_ids": ["infores:dailymed", "infores:faers"]}],
        }
    ]
    assert validate_kgx(nodes, edges).ok is True


def test_validation_is_deterministic() -> None:
    nodes, edges = _load("broken_nodes.jsonl"), _load("broken_edges.jsonl")
    assert validate_kgx(nodes, edges).kgx_problems == validate_kgx(nodes, edges).kgx_problems


def test_read_kgx_jsonl_reads_plain_and_gzip(tmp_path: Path) -> None:
    plain = KGX_ROOT / "nodes.jsonl"
    gz_path = tmp_path / "nodes.jsonl.gz"
    with gzip.open(gz_path, "wb") as handle:
        handle.write(plain.read_bytes())
    assert read_kgx_jsonl(gz_path) == read_kgx_jsonl(plain)


# --- negative: the broken fixture reports specific problems ----------------------


def test_broken_fixture_reports_every_defect_code() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    assert report.ok is False
    assert _codes(report) == {
        MISSING_NODE_REFERENCE,
        INVALID_PREDICATE,
        INCOMPATIBLE_SUBJECT_CATEGORY,
        INCOMPATIBLE_OBJECT_CATEGORY,
        MISSING_PROVENANCE,
        MISSING_EDGE_FIELD,
        MISSING_NODE_FIELD,
        INVALID_NODE_CATEGORY,
        DUPLICATE_NODE_ID,
    }


def test_missing_node_reference_isolated_to_offending_edge() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    problems = _by_entity(report, "uuid-broken-e1")
    assert [(p.code, p.field) for p in problems] == [(MISSING_NODE_REFERENCE, "object")]
    assert "MONDO:0000001" in problems[0].message


def test_bad_predicate_isolated() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    problems = _by_entity(report, "uuid-broken-e2")
    assert [p.code for p in problems] == [INVALID_PREDICATE]
    assert "biolink:causes" in problems[0].message


def test_incompatible_subject_and_object_categories_isolated() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    assert [p.code for p in _by_entity(report, "uuid-broken-e3")] == [INCOMPATIBLE_SUBJECT_CATEGORY]
    assert [p.code for p in _by_entity(report, "uuid-broken-e4")] == [INCOMPATIBLE_OBJECT_CATEGORY]


def test_missing_provenance_presence_isolated() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    problems = _by_entity(report, "uuid-broken-e5")
    assert [(p.code, p.field) for p in problems] == [(MISSING_PROVENANCE, "primary_knowledge_source/sources")]


def test_missing_family_upstream_names_dailymed() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    problems = _by_entity(report, "uuid-broken-e7")
    assert [p.code for p in problems] == [MISSING_PROVENANCE]
    assert "infores:dailymed" in problems[0].message


def test_missing_required_edge_fields_isolated() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    problems = _by_entity(report, "uuid-broken-e6")
    assert {p.code for p in problems} == {MISSING_EDGE_FIELD}
    assert {p.field for p in problems} == {"knowledge_level", "agent_type"}


def test_node_level_problems_are_specific() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    node_problems = {p.entity_id: p.code for p in report.kgx_problems if p.entity_kind == "node"}
    assert node_problems["CHEBI:11111"] == MISSING_NODE_FIELD  # missing name
    assert node_problems["MONDO:99999"] == INVALID_NODE_CATEGORY  # category lacks biolink: prefix
    assert node_problems["CHEBI:99999"] == DUPLICATE_NODE_ID  # second occurrence of the id


def test_rendered_problems_carry_their_code() -> None:
    report = validate_kgx(_load("broken_nodes.jsonl"), _load("broken_edges.jsonl"))
    assert report.problems
    for line, problem in zip(report.problems, report.kgx_problems, strict=True):
        assert line == f"[{problem.code}] {problem.message}"


# --- backward compatibility: the legacy assertion-table contract -----------------


def test_legacy_validate_reports_missing_assertion_tables() -> None:
    report = validate([])
    assert report.ok is False
    assert any("approved_treats_assertions" in problem for problem in report.problems)
    assert report.kgx_problems == []  # legacy path never populates KGX problems
