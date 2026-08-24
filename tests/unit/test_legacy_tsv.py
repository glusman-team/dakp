"""Unit tests for the legacy DAKP TSV export (:mod:`dakp_pipeline.legacy_tsv`).

Covers the converter semantics recovered from the original ``jsonlines2tsv.py`` (first-element
node categories, ``NA`` fills, comma-joined multi-values, canonical-name-first endpoint names,
int/str case counts) and the stage entry point (deferred -> ``[]``, real -> the TSV pair written
beside the ndjson sources + registered, loud ``RuntimeError`` guards on a missing report or
missing current graph/version pair).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline import __version__
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.legacy_tsv import EDGES_HEADER, NA, NODES_HEADER, convert_edges, convert_nodes, export
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import GRAPH_NAME, REPORT_NAME

# Reference records mirroring the two sample rows the internal service consumes (v0.5.3):
# CHEBI:4875/Etanercept treats+applied MONDO:0008383/rheumatoid arthritis.
_NODES: list[dict[str, Any]] = [
    {"id": "CHEBI:4875", "name": "Etanercept", "category": ["biolink:ChemicalEntity", "biolink:SmallMolecule"]},
    {"id": "MONDO:0008383", "name": "rheumatoid arthritis", "category": ["biolink:Disease"]},
    {"id": "CHEBI:5855", "name": "ibuprofen", "category": "biolink:Drug"},
]

_EDGES: list[dict[str, Any]] = [
    {
        "id": "408826a1-4c4d-392f-8fd0-c38dee954490",
        "subject": "CHEBI:4875",
        "predicate": "biolink:applied_to_treat",
        "object": "MONDO:0008383",
        "knowledge_level": "observation",
        "agent_type": "manual_validation_of_automated_agent",
        "evidence_count": 269572,
    },
    {
        "id": "8ccfd259",
        "subject": "CHEBI:5855",
        "predicate": "biolink:treats",
        "object": "HP:0002315",
        "original_subject": "Advil",
        "original_object": "headache",
        "knowledge_level": "knowledge_assertion",
        "agent_type": "manual_validation_of_automated_agent",
        "approval_ids": ["NDA012345", "ANDA065432"],
        "has_evidence": ["dailymed:set-a", "dailymed:set-b"],
    },
    {
        "id": "contra-1",
        "subject": "X:1",
        "predicate": "biolink:contraindicated_in",
        "object": "Y:2",
        "knowledge_level": "knowledge_assertion",
        "agent_type": "manual_validation_of_automated_agent",
        "evidence_count": "7",
        "has_evidence": [],
    },
]


# --- converters --------------------------------------------------------------------


def test_convert_nodes_takes_first_category_and_keeps_scalar() -> None:
    frame = convert_nodes(_NODES)
    assert frame.columns == NODES_HEADER
    rows = frame.to_dicts()
    assert rows[0] == {"id": "CHEBI:4875", "name": "Etanercept", "category": "biolink:ChemicalEntity"}
    assert rows[2] == {"id": "CHEBI:5855", "name": "ibuprofen", "category": "biolink:Drug"}  # scalar category passes through


def test_convert_nodes_fills_na_for_missing_fields_and_empty_category() -> None:
    frame = convert_nodes([{"id": "CHEBI:1"}, {"name": "nameless", "category": []}])
    assert frame.to_dicts() == [{"id": "CHEBI:1", "name": NA, "category": NA}, {"id": NA, "name": "nameless", "category": NA}]


def test_convert_nodes_empty_input_is_header_only() -> None:
    assert convert_nodes([]).columns == NODES_HEADER
    assert convert_nodes([]).height == 0


def test_convert_edges_matches_the_legacy_sample_row() -> None:
    # Row 1 is the exact legacy reference: applied_to_treat with N_cases, everything else NA.
    row = convert_edges([_EDGES[0]], _NODES).to_dicts()[0]
    assert row == {
        "id": "408826a1-4c4d-392f-8fd0-c38dee954490",
        "subject": "CHEBI:4875",
        "predicate": "biolink:applied_to_treat",
        "object": "MONDO:0008383",
        "subject_name": "Etanercept",
        "object_name": "rheumatoid arthritis",
        "object_modifier": NA,
        "knowledge_level": "observation",
        "agent_type": "manual_validation_of_automated_agent",
        "approval": NA,
        "N_cases": "269572",
        "supporting_spls": NA,
    }


def test_convert_edges_comma_joins_multivalued_fields() -> None:
    row = convert_edges([_EDGES[1]], _NODES).to_dicts()[0]
    assert row["approval"] == "NDA012345,ANDA065432"
    assert row["supporting_spls"] == "dailymed:set-a,dailymed:set-b"


def test_convert_edges_endpoint_name_falls_back_to_mention_then_na() -> None:
    rows = convert_edges(_EDGES, _NODES).to_dicts()
    # Subject resolved in nodes -> canonical name wins over the raw FAERS brand mention.
    assert rows[1]["subject_name"] == "ibuprofen"
    # Object missing from nodes -> original_object mention.
    assert rows[1]["object_name"] == "headache"
    # Both endpoints unknown and no original_* fields -> NA.
    assert rows[2]["subject_name"] == NA
    assert rows[2]["object_name"] == NA


def test_convert_edges_totals_on_odd_shapes() -> None:
    # A scalar approval blob passes through; a numeric-string count stringifies; an empty
    # evidence list is NA; missing knowledge fields are NA; a node with an NA id never wins a
    # name lookup.
    edge = {
        "id": "odd",
        "subject": "CHEBI:5855",
        "predicate": "biolink:treats",
        "object": "MONDO:0008383",
        "approval_ids": "NDA1|NDA2",
        "evidence_count": 3,
        "has_evidence": "dailymed:one",
    }
    nodes = [*_NODES, {"name": "ghost", "category": ["biolink:Disease"]}]
    row = convert_edges([edge], nodes).to_dicts()[0]
    assert row["approval"] == "NDA1|NDA2"
    assert row["N_cases"] == "3"
    assert row["supporting_spls"] == "dailymed:one"
    assert row["knowledge_level"] == NA
    assert row["agent_type"] == NA
    assert row["subject_name"] == "ibuprofen"


def test_convert_edges_empty_list_evidence_is_na_and_str_count_stringifies() -> None:
    row = convert_edges([_EDGES[2]], _NODES).to_dicts()[0]
    assert row["supporting_spls"] == NA
    assert row["N_cases"] == "7"


# --- export stage ------------------------------------------------------------------


def _report_ref(workdir: Workdir, mode: str) -> ArtifactRef:
    """Write a handoff report with the given mode under reports/ and return its ref."""
    path = workdir.reports / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": mode, "status": mode}), encoding="utf-8")
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/json")


def _write_kgx(workdir: Workdir) -> tuple[Path, Path]:
    """Drop exactly one KGX ndjson pair under data/, like a successful build-kg."""
    data = workdir.root / "data"
    nodes_path = data / f"{GRAPH_NAME}_{__version__}.nodes.ndjson"
    edges_path = data / f"{GRAPH_NAME}_{__version__}.edges.ndjson"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in _NODES), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in _EDGES), encoding="utf-8")
    return nodes_path, edges_path


def _ctx(workdir: Workdir) -> TaskContext:
    return TaskContext(workdir=workdir.root, fixture_root=None, params={})


def test_export_deferred_handoff_returns_empty(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    workdir.create()
    refs = export([_report_ref(workdir, "deferred")], _ctx(workdir))
    assert refs == []
    assert list((workdir.root / "data").glob("*.nodes.tsv")) == []


def test_export_real_handoff_writes_the_legacy_pair(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    workdir.create()
    nodes_ndjson, edges_ndjson = _write_kgx(workdir)
    report_ref = _report_ref(workdir, "real")
    refs = export([report_ref], _ctx(workdir))

    assert [ref.uri for ref in refs] == [nodes_ndjson.with_suffix(".tsv"), edges_ndjson.with_suffix(".tsv")]
    assert [ref.rows for ref in refs] == [len(_NODES), len(_EDGES)]
    assert all(ref.media_type == "text/tab-separated-values" for ref in refs)
    for ref, ndjson in zip(refs, (nodes_ndjson, edges_ndjson), strict=True):
        assert hash_file(ndjson) in (json.loads(ref.manifest.read_text(encoding="utf-8"))["inputs"] if ref.manifest else [])

    nodes_lines = (workdir.root / "data" / f"{GRAPH_NAME}_{__version__}.nodes.tsv").read_text(encoding="utf-8").splitlines()
    assert nodes_lines[0].split("\t") == NODES_HEADER
    assert nodes_lines[1].split("\t") == ["CHEBI:4875", "Etanercept", "biolink:ChemicalEntity"]

    edges_lines = (workdir.root / "data" / f"{GRAPH_NAME}_{__version__}.edges.tsv").read_text(encoding="utf-8").splitlines()
    assert edges_lines[0].split("\t") == EDGES_HEADER
    assert edges_lines[1].split("\t")[4] == "Etanercept"  # subject_name
    assert edges_lines[1].split("\t")[10] == "269572"  # N_cases
    assert edges_lines[2].split("\t")[9] == "NDA012345,ANDA065432"  # approval


def test_export_without_a_handoff_report_raises(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    workdir.create()
    with pytest.raises(RuntimeError, match=r"tablassert_handoff\.json"):
        export([], _ctx(workdir))


def test_export_with_two_handoff_reports_raises(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    workdir.create()
    with pytest.raises(RuntimeError, match="exactly one"):
        export([_report_ref(workdir, "real"), _report_ref(workdir, "real")], _ctx(workdir))


def test_export_with_missing_ndjson_raises(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    workdir.create()
    _, edges_ndjson = _write_kgx(workdir)
    edges_ndjson.unlink()
    with pytest.raises(RuntimeError, match=rf"'{GRAPH_NAME}_{__version__}\.edges\.ndjson'.*found 0"):
        export([_report_ref(workdir, "real")], _ctx(workdir))


def test_export_ignores_stale_kgx_pairs(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path)
    workdir.create()
    nodes_ndjson, edges_ndjson = _write_kgx(workdir)
    data = workdir.root / "data"
    for stem in ("dakp_0.1.0", "DRUG_APPROVALS_KP_1.0.0"):
        (data / f"{stem}.nodes.ndjson").write_text(json.dumps(_NODES[0]) + "\n", encoding="utf-8")
        (data / f"{stem}.edges.ndjson").write_text(json.dumps(_EDGES[0]) + "\n", encoding="utf-8")

    refs = export([_report_ref(workdir, "real")], _ctx(workdir))

    assert [ref.uri for ref in refs] == [nodes_ndjson.with_suffix(".tsv"), edges_ndjson.with_suffix(".tsv")]
