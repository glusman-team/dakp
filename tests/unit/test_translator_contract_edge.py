"""Edge-case tests for ``dakp_pipeline.translator.contract``.

Targets the negative branches the fixture-based tests do not isolate: the legacy ``validate``
unreadable-table and missing-column paths, and the KGX validator's per-field guards (malformed
category containers, provenance extraction edge cases, missing/duplicate ids, missing subject
references, and a non-family predicate that still lacks the DAKP infores).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.translator.contract import (
    DUPLICATE_EDGE_ID,
    INVALID_NODE_CATEGORY,
    INVALID_PREDICATE,
    MISSING_EDGE_FIELD,
    MISSING_NODE_FIELD,
    MISSING_NODE_REFERENCE,
    MISSING_PROVENANCE,
    validate,
    validate_kgx,
)

_DAKP = "infores:multiomics-drugapprovals"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3="b3:" + "0" * 64, media_type="text/tab-separated-values")


def _chem(node_id: str = "CHEBI:1") -> dict[str, Any]:
    return {"id": node_id, "name": "drug", "category": ["biolink:Drug"]}


def _dis(node_id: str = "MONDO:1") -> dict[str, Any]:
    return {"id": node_id, "name": "disease", "category": ["biolink:Disease"]}


def _edge(**overrides: Any) -> dict[str, Any]:
    edge: dict[str, Any] = {
        "id": "e1",
        "subject": "CHEBI:1",
        "predicate": "biolink:treats",
        "object": "MONDO:1",
        "category": ["biolink:EntityToDiseaseAssociation"],
        "knowledge_level": "knowledge_assertion",
        "agent_type": "manual_validation_of_automated_agent",
        "primary_knowledge_source": _DAKP,
        "sources": [{"resource_id": _DAKP, "upstream_resource_ids": ["infores:dailymed", "infores:faers"]}],
    }
    edge.update(overrides)
    return edge


def _codes(report: Any) -> set[tuple[str, str]]:
    return {(problem.code, problem.field) for problem in report.kgx_problems}


# --- legacy assertion-table validate() -------------------------------------------


def test_validate_reports_unreadable_table(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "approved_treats_assertions.tsv")  # does not exist -> read raises
    report = validate([ref])
    assert report.ok is False
    assert any("unreadable table" in problem for problem in report.problems)


def test_validate_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "approved_treats_assertions.tsv"
    path.write_text("subject_text\tobject_text\na\tb\n", encoding="utf-8")
    report = validate([_ref(path)])
    assert report.ok is False
    assert any("missing columns" in problem for problem in report.problems)
    assert report.tables["approved_treats_assertions"]["rows"] == 1
    assert report.tables["approved_treats_assertions"]["missing_columns"]


# --- KGX node guards --------------------------------------------------------------


def test_node_missing_id_name_and_category() -> None:
    report = validate_kgx([{}], [])
    assert (MISSING_NODE_FIELD, "id") in _codes(report)
    assert (MISSING_NODE_FIELD, "name") in _codes(report)
    assert (INVALID_NODE_CATEGORY, "category") in _codes(report)


def test_node_empty_category_list_is_invalid() -> None:
    report = validate_kgx([{"id": "X:1", "name": "thing", "category": []}], [])
    assert (INVALID_NODE_CATEGORY, "category") in _codes(report)


# --- KGX edge guards --------------------------------------------------------------


def test_edge_category_as_string_is_malformed() -> None:
    report = validate_kgx([_chem(), _dis()], [_edge(category="biolink:notalist")])
    assert (MISSING_EDGE_FIELD, "category") in _codes(report)


def test_edge_empty_category_list_is_malformed() -> None:
    report = validate_kgx([_chem(), _dis()], [_edge(category=[])])
    assert (MISSING_EDGE_FIELD, "category") in _codes(report)


def test_edge_missing_id() -> None:
    edge = _edge()
    del edge["id"]
    report = validate_kgx([_chem(), _dis()], [edge])
    assert (MISSING_EDGE_FIELD, "id") in _codes(report)


def test_edge_duplicate_id() -> None:
    report = validate_kgx([_chem(), _dis()], [_edge(id="dup"), _edge(id="dup")])
    assert (DUPLICATE_EDGE_ID, "id") in _codes(report)


def test_edge_subject_not_in_nodes() -> None:
    report = validate_kgx([_chem("CHEBI:1"), _dis()], [_edge(subject="CHEBI:999")])
    assert (MISSING_NODE_REFERENCE, "subject") in _codes(report)


# --- provenance extraction edge cases --------------------------------------------


def test_provenance_from_sources_without_primary() -> None:
    edge = _edge()
    del edge["primary_knowledge_source"]  # sources still carry DAKP + dailymed + faers
    report = validate_kgx([_chem(), _dis()], [edge])
    assert report.ok is True  # _edge_infores skips the empty primary, reads the sources list


def test_provenance_sources_not_a_list() -> None:
    edge = _edge()
    del edge["sources"]  # primary present; sources absent -> not a list -> skipped
    report = validate_kgx([_chem(), _dis()], [edge])
    assert (MISSING_PROVENANCE, "sources") in _codes(report)  # dailymed/faers upstream now missing


def test_provenance_source_entry_not_a_mapping() -> None:
    report = validate_kgx([_chem(), _dis()], [_edge(sources=["not-a-mapping"])])
    assert (MISSING_PROVENANCE, "sources") in _codes(report)


def test_non_family_predicate_missing_dakp_infores() -> None:
    edge = _edge(predicate="biolink:causes", primary_knowledge_source="infores:other", sources=[{"resource_id": "infores:other"}])
    report = validate_kgx([_chem(), _dis()], [edge])
    codes = {problem.code for problem in report.kgx_problems}
    assert INVALID_PREDICATE in codes
    assert MISSING_PROVENANCE in codes  # family is None -> the elif DAKP-infores guard fires
