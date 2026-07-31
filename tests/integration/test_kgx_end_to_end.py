"""End-to-end DAKP -> Tablassert -> KGX integration test (requires ``tablassert``).

Proves the FULL path works on a TINY, hermetic fullmap (no network):

1. run the mock pipeline over the tiny pipeline fixtures -> the three DAKP assertion TSVs
   (``data/tabular/*_assertions.tsv``) + the generated ``tables/*.yaml`` Tablassert configs;
2. build a tiny ``fullmap.redb`` (:mod:`tiny_fullmap`) mapping the assertion-table mention text
   (Ibuprofen/Advil, Examplestatin, hypercholesterolemia, headache, pain, asthma, ...) to CURIEs;
3. invoke a REAL ``tablassert build-kg`` through the DAKP :class:`TablassertRunner` (the installed
   ``tablassert`` CLI, real subprocess);
4. load the produced KGX ``dakp_0.1.0.{nodes,edges}.ndjson`` and assert: nodes carry
   ``id``/``name``/``category``; edges carry ``subject``/``predicate``/``object`` + DAKP provenance
   (``infores:multiomics-drugapprovals`` primary + the per-family upstream infores); all three edge
   families (``treats`` / ``applied_to_treat`` / ``contraindicated_in``) are present; and
   :func:`dakp_pipeline.translator.contract.validate_kgx` passes.

The whole module SKIPS when ``tablassert`` is not importable (dependencies not yet installed) — it
runs once ``uv sync`` has materialized the runtime. ``tests/`` is outside the coverage ``source``,
so the skip does not affect the 100% ``src/`` coverage gate.

Known gap (NOTE, not fixed here — ``src/`` is owned elsewhere): Tablassert 8.0.0 emits edge provenance
as a LIST ``primary_knowledge_source`` (``["infores:multiomics-drugapprovals"]``) plus a top-level
``upstream_resource_ids`` field and NO ``sources`` list, whereas ``validate_kgx`` expects a SCALAR
``primary_knowledge_source`` plus a ``sources`` list (the shape in ``tests/fixtures/kgx/edges.jsonl``).
So ``validate_kgx`` reports ``missing_provenance`` on RAW Tablassert edges. :func:`_canonicalize_edge`
is the test-side adapter from Tablassert's raw shape to the canonical Translator contract; the
recommended ``src/`` reconciliation is to accept Tablassert's shape (or canonicalize in the pipeline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import tiny_fullmap
from harness import install_fixture_fetchers, run_stages

from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.tablassert.run import TablassertRunner
from dakp_pipeline.translator.contract import EDGE_FAMILIES, INFORES_DAKP, read_kgx_jsonl, validate_kgx

# Skip the WHOLE module when tablassert is not importable (deps not installed). tiny_fullmap imports
# tablassert only inside its child build process, so importing it above is safe even when absent.
pytest.importorskip("tablassert", reason="tablassert not installed; run `uv sync`")

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"

# The exact (subject, predicate, object) triples the tiny fixtures produce, post fullmap resolution.
# Advil (FAERS brand) canonicalizes to CHEBI:5855 (ibuprofen); Examplestatin -> the fictional
# CHEBI:1000001 stand-in. See tiny_fullmap.TERMS.
_TREATS = "biolink:treats"
_APPLIED = "biolink:applied_to_treat"
_CONTRA = "biolink:contraindicated_in"
EXPECTED_EDGES: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("CHEBI:1000001", _TREATS, "MONDO:0005154"),  # Examplestatin -> hypercholesterolemia
        ("CHEBI:5855", _TREATS, "HP:0002315"),  # Ibuprofen -> headache
        ("CHEBI:5855", _TREATS, "HP:0012531"),  # Ibuprofen -> pain
        ("CHEBI:5855", _APPLIED, "HP:0002315"),  # Advil -> headache
        ("CHEBI:1000001", _APPLIED, "MONDO:0005154"),  # Examplestatin -> hypercholesterolemia
        ("CHEBI:5855", _CONTRA, "MONDO:0004979"),  # Ibuprofen -> asthma
    }
)
EXPECTED_NODE_IDS: frozenset[str] = frozenset({"CHEBI:5855", "CHEBI:1000001", "MONDO:0005154", "HP:0002315", "HP:0012531", "MONDO:0004979"})


@dataclass(frozen=True)
class KgxBuild:
    """Result of one real DAKP -> Tablassert build-kg run over the tiny fullmap."""

    workdir: Path
    report: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def _ref(path: Path, media_type: str) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type=media_type)


def _canonicalize_edge(edge: dict[str, Any]) -> dict[str, Any]:
    """Adapt a RAW Tablassert edge to the canonical Translator KGX shape ``validate_kgx`` checks.

    Tablassert 8.0.0: ``primary_knowledge_source`` is a LIST and the upstream infores live in a
    top-level ``upstream_resource_ids`` field (no ``sources``). Canonical contract (see
    ``tests/fixtures/kgx/edges.jsonl``): SCALAR ``primary_knowledge_source`` + a ``sources`` list whose
    primary entry carries ``upstream_resource_ids``. See the module docstring "Known gap" note.
    """
    raw_primary = edge.get("primary_knowledge_source")
    if isinstance(raw_primary, list):
        primary = raw_primary[0] if raw_primary else ""
    elif isinstance(raw_primary, str):
        primary = raw_primary
    else:
        primary = ""
    upstream = [entry for entry in (edge.get("upstream_resource_ids") or []) if isinstance(entry, str)]
    sources: list[dict[str, Any]] = [{"resource_id": primary, "resource_role": "primary_knowledge_source", "upstream_resource_ids": upstream}]
    sources.extend({"resource_id": infores, "resource_role": "supporting_data_source"} for infores in upstream)
    canonical = dict(edge)
    canonical["primary_knowledge_source"] = primary
    canonical["sources"] = sources
    return canonical


@pytest.fixture(scope="module")
def kgx_build(tmp_path_factory: pytest.TempPathFactory) -> KgxBuild:
    """Run the offline fixture pipeline + a REAL ``tablassert build-kg`` once; return the KGX."""
    work = tmp_path_factory.mktemp("dakp-kgx-e2e") / "work"

    # (1) Hermetic offline pipeline -> assertion TSVs (data/tabular/) + generated tables/*.yaml.
    # Fetchers always run their real branches; route them to the fixtures for this hermetic build.
    monkeypatch = pytest.MonkeyPatch()
    install_fixture_fetchers(monkeypatch)
    try:
        run_stages(fixture_root=_FIXTURE_ROOT, workdir=work)
    finally:
        monkeypatch.undo()

    # (2) Tiny fullmap at <work>/.fullmap/fullmap.redb (graph.yaml's `fullmap: ".fullmap"` resolves
    #     relative to the build cwd = <work>). Built in a child process so the build-kg subprocess
    #     below can acquire redb's exclusive flock (see tiny_fullmap docstring).
    tiny_fullmap.build_tiny_fullmap(work / ".fullmap" / "fullmap.redb")

    # (3) REAL build-kg via the DAKP TablassertRunner (installed tablassert CLI, real subprocess).
    tabular = work / "data" / "tabular"
    assertion_refs = [_ref(path, "text/tab-separated-values") for path in sorted(tabular.glob("*_assertions.tsv"))]
    tables_dir = work / "tables"
    config_refs = [_ref(path, "application/x-yaml") for path in sorted(tables_dir.glob("*.yaml"))]
    # We call TablassertRunner.run() directly (bypassing the module-level run() dispatcher); only
    # params["fullmap"] is read by the runner.
    ctx = TaskContext(workdir=work, fixture_root=_FIXTURE_ROOT, params={"fullmap": ".fullmap"})
    report_refs = TablassertRunner().run(assertion_refs, config_refs, ctx)
    report: dict[str, Any] = json.loads(report_refs[0].uri.read_text(encoding="utf-8"))

    # (4) Load the produced KGX NDJSON (compile_graph writes <name>_<version>.{nodes,edges}.ndjson to cwd).
    node_files = sorted(work.glob("*.nodes.ndjson"))
    edge_files = sorted(work.glob("*.edges.ndjson"))
    assert len(node_files) == 1, f"expected exactly one KGX nodes file, found {node_files}"
    assert len(edge_files) == 1, f"expected exactly one KGX edges file, found {edge_files}"
    return KgxBuild(workdir=work, report=report, nodes=read_kgx_jsonl(node_files[0]), edges=read_kgx_jsonl(edge_files[0]))


def test_build_kg_handoff_succeeds(kgx_build: KgxBuild) -> None:
    """The DAKP TablassertRunner ran the REAL build-kg CLI and it exited cleanly."""
    assert kgx_build.report["mode"] == "real"
    assert kgx_build.report["status"] == "ok"
    assert kgx_build.report["exit_code"] == 0
    command = kgx_build.report["command"]
    assert "build-kg" in command
    assert any(arg.endswith("graph.yaml") for arg in command)


def test_nodes_have_required_fields(kgx_build: KgxBuild) -> None:
    """Every KGX node carries id/name/category with biolink-prefixed categories."""
    assert kgx_build.nodes, "build-kg produced no nodes"
    for node in kgx_build.nodes:
        node_id = node.get("id")
        assert isinstance(node_id, str)
        assert node_id
        node_name = node.get("name")
        assert isinstance(node_name, str)
        assert node_name
        category = node.get("category")
        assert isinstance(category, list)
        assert category
        assert all(isinstance(entry, str) and entry.startswith("biolink:") for entry in category)
    assert {node["id"] for node in kgx_build.nodes} == EXPECTED_NODE_IDS


def test_three_edge_families_present(kgx_build: KgxBuild) -> None:
    """All three DAKP edge families are present, with the exact resolved triples."""
    triples = {(edge["subject"], edge["predicate"], edge["object"]) for edge in kgx_build.edges}
    assert triples == EXPECTED_EDGES
    predicates = {edge["predicate"] for edge in kgx_build.edges}
    assert predicates == {_TREATS, _APPLIED, _CONTRA}


def test_edges_carry_dakp_provenance(kgx_build: KgxBuild) -> None:
    """RAW Tablassert edges carry the DAKP primary infores + the per-family upstream infores.

    Pins Tablassert 8.0.0's actual provenance shape: ``primary_knowledge_source`` is a LIST and the
    upstream infores are a top-level ``upstream_resource_ids`` field (NOT a ``sources`` list) — the
    shape ``validate_kgx`` does not accept raw (see module docstring + ``_canonicalize_edge``).
    """
    assert kgx_build.edges, "build-kg produced no edges"
    for edge in kgx_build.edges:
        # subject/predicate/object are resolved CURIEs.
        for field_name in ("subject", "predicate", "object"):
            value = edge.get(field_name)
            assert isinstance(value, str)
            assert value
        # Primary provenance: the DAKP infores (Tablassert emits it as a list).
        primary = edge.get("primary_knowledge_source")
        assert isinstance(primary, list)
        assert INFORES_DAKP in primary
        # Upstream provenance matches the edge family's required upstream infores.
        family = EDGE_FAMILIES[edge["predicate"]]
        upstream = frozenset(edge.get("upstream_resource_ids") or [])
        assert upstream == family.required_upstream


def test_validate_kgx_passes_on_canonical_kgx(kgx_build: KgxBuild) -> None:
    """validate_kgx passes once RAW Tablassert edges are canonicalized to the Translator contract."""
    canonical_edges = [_canonicalize_edge(edge) for edge in kgx_build.edges]
    report = validate_kgx(kgx_build.nodes, canonical_edges)
    assert report.ok, f"validate_kgx reported problems: {report.problems}"
    assert report.problems == []
    assert report.kgx_problems == []
