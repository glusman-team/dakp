"""Legacy DAKP TSV export: retrofit the Tablassert KGX pair into the old TSV schema.

An internal service still consumes the pre-rewrite ``druginfo`` TSV pair
(``drug_approvals_kg_nodes_v0.5.3.tsv.gz`` / ``..._edges_...``) — three node columns
(``id``/``name``/``category``) and twelve edge columns. DAKP's own compiler was retired
(US-004); the graph now comes out of the Tablassert handoff as KGX ndjson
(``data/DRUG_APPROVALS_KP_<version>.{nodes,edges}.ndjson``), so the old pair no longer exists. This stage
converts that ndjson pair back into the legacy schema — same stem and directory, extension
swapped: ``<workdir>/data/DRUG_APPROVALS_KP_<version>.nodes.tsv`` / ``.edges.tsv`` (plain TSV, like every
other DAKP tabular output).

The conversion semantics are the ORIGINAL ones, recovered from the deleted legacy producer
``ref/legacy/bin/jsonlines2tsv.py`` (readable via
``git show c9444e3^:ref/legacy/bin/jsonlines2tsv.py``):

* nodes — ``category`` is the FIRST element when the record carries a list;
* edges — one column per header, ``NA`` for absent fields, comma-joined multi-values:
  ``approval <- ",".join(approvals)`` and ``supporting_spls <- ",".join(has_evidence)``.

Field map (current Tablassert 12 KGX edge -> legacy column):

===============  ============================================================  ==============================
legacy column     source                                                        notes
===============  ============================================================  ==============================
id                ``id``                                                        deterministic UUID
subject/predicate ``subject`` / ``predicate`` / ``object``                      resolved CURIEs
/object
subject_name /    node ``name`` from nodes.ndjson; fallback ``original_subject`` canonical name preferred (legacy
object_name       / ``original_object``; else ``NA``                            parity), raw mention kept when
                                                                                the node lookup misses
object_modifier   always ``NA``                                                 the legacy KG never populated it
knowledge_level   ``knowledge_level``
agent_type        ``agent_type``
approval          ``",".join(approval_ids)``                                    treats edges only
N_cases           ``str(evidence_count)``                                       applied_to_treat edges only
supporting_spls   ``",".join(has_evidence)``                                    ``dailymed:<spl_set_id>`` CURIEs
===============  ============================================================  ==============================

The converters are total: a list where a scalar is expected joins, a scalar where a list is
expected passes through, an absent/empty cell becomes ``NA`` — a weird record degrades, never
crashes the export.

The stage entry point :func:`export` is Airflow-free. A **deferred** handoff (no fullmap ->
:func:`dakp_pipeline.tablassert.run` never invoked ``build-kg`` -> no ndjson to convert) returns
an EMPTY ref list — never an error — mirroring the deferred-handoff convention; the DAG task
turns that into an ``AirflowSkipException`` so the run shows skipped, not failed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline import __version__
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.io.schemas import TSV_MEDIA_TYPE, write_tsv
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import GRAPH_NAME, REPORT_NAME
from dakp_pipeline.translator import read_kgx_jsonl

#: Legacy node contract: ``id  name  category``.
NODES_HEADER: list[str] = ["id", "name", "category"]

#: Legacy edge contract — the exact 12 columns of ``drug_approvals_kg_edges_v0.5.3.tsv.gz``
#: (tab-separated in the original header string; order is load-bearing for the old service).
EDGES_HEADER: list[str] = [
    "id",
    "subject",
    "predicate",
    "object",
    "subject_name",
    "object_name",
    "object_modifier",
    "knowledge_level",
    "agent_type",
    "approval",
    "N_cases",
    "supporting_spls",
]

#: The never-populated modifier column; the legacy KG left it ``NA`` on every edge.
NA = "NA"

_EXPORT_OPERATION = "export_legacy_tsv"


# --- cell shaping (legacy jsonlines2tsv semantics) ---------------------------------


def _text(value: Any) -> str:
    """One scalar cell as a string (``NA`` when absent/empty)."""
    return value.strip() if isinstance(value, str) and value.strip() else NA


def _first_or_text(value: Any) -> str:
    """Node ``category``: the FIRST element of a list, else the scalar (legacy rule)."""
    if isinstance(value, list):
        return _first_or_text(value[0]) if value else NA
    return _text(value)


def _joined(value: Any) -> str:
    """One multi-valued cell: comma-joined list, passthrough scalar, ``NA`` when empty."""
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else NA
    return _text(value)


def _endpoint_name(edge: Mapping[str, Any], side: str, names: Mapping[str, str]) -> str:
    """``subject_name`` / ``object_name``: canonical node name, mention fallback, ``NA``.

    Legacy edges carried the canonical resolved-entity name ("Etanercept", never the raw label
    text), so the node's ``name`` from nodes.ndjson wins; ``original_subject`` /
    ``original_object`` (the raw assertion mention, e.g. the FAERS brand "Advil") is the fallback
    for an endpoint that somehow missed the node set.
    """
    curie = _text(edge.get(side))
    canonical = names.get(curie, "")
    if canonical:
        return canonical
    return _text(edge.get(f"original_{side}"))


def _edge_row(edge: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, str]:
    """One KGX edge record -> one legacy edge row (all 12 columns, every cell a string)."""
    return {
        "id": _text(edge.get("id")),
        "subject": _text(edge.get("subject")),
        "predicate": _text(edge.get("predicate")),
        "object": _text(edge.get("object")),
        "subject_name": _endpoint_name(edge, "subject", names),
        "object_name": _endpoint_name(edge, "object", names),
        "object_modifier": NA,
        "knowledge_level": _text(edge.get("knowledge_level")),
        "agent_type": _text(edge.get("agent_type")),
        "approval": _joined(edge.get("approval_ids")),
        "N_cases": _text(str(edge.get("evidence_count"))) if edge.get("evidence_count") is not None else NA,
        "supporting_spls": _joined(edge.get("has_evidence")),
    }


# --- converters (pure: records -> frames) -------------------------------------------


def convert_nodes(nodes: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Convert KGX node records into the legacy 3-column node frame."""
    rows = [{"id": _text(node.get("id")), "name": _text(node.get("name")), "category": _first_or_text(node.get("category"))} for node in nodes]
    return pl.DataFrame(rows, schema=NODES_HEADER, orient="row")


def convert_edges(edges: Sequence[Mapping[str, Any]], nodes: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Convert KGX edge records into the legacy 12-column edge frame.

    ``nodes`` supplies the canonical endpoint names (id -> ``name`` index); edges referencing an
    unknown node fall back to their own ``original_subject`` / ``original_object`` mention text.
    """
    names = {_text(node.get("id")): _text(node.get("name")) for node in nodes}
    names.pop(NA, None)
    rows = [_edge_row(edge, names) for edge in edges]
    return pl.DataFrame(rows, schema=EDGES_HEADER, orient="row")


# --- stage entry point ---------------------------------------------------------------


def _single_glob(directory: Path, pattern: str) -> Path:
    """Exactly one match for ``pattern`` in ``directory`` (RuntimeError on zero or many)."""
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        msg = f"expected exactly one {pattern!r} in {directory}, found {len(matches)}: {[str(m) for m in matches]}"
        raise RuntimeError(msg)
    return matches[0]


def export(kgx_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Export the legacy TSV pair from the Tablassert KGX ndjson; ``[]`` when handoff deferred.

    Reads the Tablassert handoff report among ``kgx_refs`` (the single ref ``run_tablassert``
    returns): a deferred report means no ``build-kg`` ran and there is no ndjson to convert, so
    the stage returns an empty ref list — never an error, mirroring the deferred-handoff
    convention. A real (successful) handoff must have left the current graph/version pair
    ``DRUG_APPROVALS_KP_<version>.nodes.ndjson`` and ``.edges.ndjson`` under ``<workdir>/data``.
    Stale KGX files from an earlier graph/version are ignored; a missing current pair is a loud
    ``RuntimeError``.

    Writes ``<ndjson stem>.nodes.tsv`` / ``.edges.tsv`` beside their ndjson sources, registers
    both with the artifact store (provenance inputs = the ndjson content hashes), and returns
    ``[nodes_ref, edges_ref]``.
    """
    event = _EXPORT_OPERATION
    report_refs = [ref for ref in kgx_refs if ref.uri.name == REPORT_NAME]
    if len(report_refs) != 1:
        msg = f"expected exactly one {REPORT_NAME} among the tablassert handoff refs, found {len(report_refs)}"
        raise RuntimeError(msg)
    report = json.loads(report_refs[0].uri.read_text(encoding="utf-8"))
    if report.get("mode") == "deferred":
        stats(logger, event, mode="deferred", reason="no Tablassert handoff ran; nothing to retrofit")
        return []

    workdir = Workdir(ctx.workdir)
    data_dir = workdir.root / "data"
    with step(logger, event):
        kgx_stem = f"{GRAPH_NAME}_{__version__}"
        nodes_ndjson = _single_glob(data_dir, f"{kgx_stem}.nodes.ndjson")
        edges_ndjson = _single_glob(data_dir, f"{kgx_stem}.edges.ndjson")
        nodes = read_kgx_jsonl(nodes_ndjson)
        edges = read_kgx_jsonl(edges_ndjson)
        nodes_path = nodes_ndjson.with_suffix(".tsv")
        edges_path = edges_ndjson.with_suffix(".tsv")
        write_tsv(convert_nodes(nodes), nodes_path)
        write_tsv(convert_edges(edges, nodes), edges_path)

        store = ArtifactStore(workdir)
        operation = OperationBlock(name=_EXPORT_OPERATION)
        nodes_ref = store.register(nodes_path, media_type=TSV_MEDIA_TYPE, rows=len(nodes), inputs=[hash_file(nodes_ndjson)], operation=operation)
        edges_ref = store.register(edges_path, media_type=TSV_MEDIA_TYPE, rows=len(edges), inputs=[hash_file(edges_ndjson)], operation=operation)
        stats(logger, event, mode="real", nodes_tsv=str(nodes_path), edges_tsv=str(edges_path), nodes=len(nodes), edges=len(edges))
    return [nodes_ref, edges_ref]


__all__ = ["EDGES_HEADER", "NA", "NODES_HEADER", "convert_edges", "convert_nodes", "export"]
