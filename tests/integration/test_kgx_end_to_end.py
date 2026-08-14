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
   :func:`dakp_pipeline.translator.validate_kgx` passes.

The whole module SKIPS when ``tablassert`` is not importable (dependencies not yet installed) — it
runs once ``uv sync`` has materialized the runtime. ``tests/`` is outside the coverage ``source``,
so the skip does not affect the 100% ``src/`` coverage gate.

Provenance shape (Tablassert >= 11): edges carry NO flat ``primary_knowledge_source`` scalar —
retrieval provenance lives only in the ``sources`` list of RetrievalSource entries; the primary
entry carries ``upstream_resource_ids`` and the table's real ``source_record_urls``, each upstream
infores appears as a ``supporting_data_source`` entry — exactly the canonical Translator contract
``validate_kgx`` checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import tiny_fullmap
from harness import install_fixture_fetchers, run_stages

from dakp_pipeline.assertions.evidence import DAILYMED_SET_URL_BASE
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.tablassert import TablassertRunner
from dakp_pipeline.translator import EDGE_FAMILIES, INFORES_DAKP, read_kgx_jsonl, validate_kgx

# Skip the WHOLE module when tablassert is not importable (deps not installed). tiny_fullmap imports
# tablassert only inside its child build process, so importing it above is safe even when absent.
pytest.importorskip("tablassert", reason="tablassert not installed; run `uv sync`")

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"

# The exact (subject, predicate, object) triples the tiny fixtures produce, post fullmap resolution.
# Advil (FAERS brand) canonicalizes to CHEBI:5855 (ibuprofen); Examplestatin -> the fictional
# CHEBI:1000001 stand-in. See tiny_fullmap.TERMS. Treats edges reflect the FAERS-primary candidate
# path: only the NDA-bearing FAERS pairs (Examplestatin/hypercholesterolemia, Advil/headache).
_TREATS = "biolink:treats"
_APPLIED = "biolink:applied_to_treat"
_CONTRA = "biolink:contraindicated_in"
EXPECTED_EDGES: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("CHEBI:1000001", _TREATS, "MONDO:0005154"),  # Examplestatin -> hypercholesterolemia
        ("CHEBI:5855", _TREATS, "HP:0002315"),  # Ibuprofen -> headache
        ("CHEBI:5855", _APPLIED, "HP:0002315"),  # Advil -> headache
        ("CHEBI:1000001", _APPLIED, "MONDO:0005154"),  # Examplestatin -> hypercholesterolemia
        ("CHEBI:5855", _CONTRA, "MONDO:0004979"),  # Ibuprofen -> asthma
    }
)
EXPECTED_NODE_IDS: frozenset[str] = frozenset({"CHEBI:5855", "CHEBI:1000001", "MONDO:0005154", "HP:0002315", "MONDO:0004979"})

# The DAKP category allow-lists (the generated table configs emit ``avoid`` as the complement of
# these per side): only drug-side and disease-side categories may survive fullmap resolution into
# the graph — no taxa, genes, publications, devices, or other wacky fullmap categories.
ALLOWED_NODE_CATEGORIES: frozenset[str] = frozenset(
    {"biolink:Drug", "biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:Disease", "biolink:PhenotypicFeature"}
)


@dataclass(frozen=True)
class KgxBuild:
    """Result of one real DAKP -> Tablassert build-kg run over the tiny fullmap."""

    workdir: Path
    report: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def _ref(path: Path, media_type: str) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type=media_type)


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

    # (4) Load the produced KGX NDJSON (compile_graph writes <name>_<version>.{nodes,edges}.ndjson
    #     into rig.artifact_base_path = "data", relative to the build cwd = <work>).
    node_files = sorted((work / "data").glob("*.nodes.ndjson"))
    edge_files = sorted((work / "data").glob("*.edges.ndjson"))
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
    """Every KGX node carries id/name/category with biolink-prefixed, allow-listed categories."""
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
        # Hard allow-list guard: the generated configs ``avoid`` every off-list category, so
        # nothing outside the drug/disease side allow-lists may reach the graph.
        assert set(category) <= ALLOWED_NODE_CATEGORIES, f"node {node_id} carries off-allow-list categories: {category}"
    assert {node["id"] for node in kgx_build.nodes} == EXPECTED_NODE_IDS


def test_three_edge_families_present(kgx_build: KgxBuild) -> None:
    """All three DAKP edge families are present, with the exact resolved triples."""
    triples = {(edge["subject"], edge["predicate"], edge["object"]) for edge in kgx_build.edges}
    assert triples == EXPECTED_EDGES
    predicates = {edge["predicate"] for edge in kgx_build.edges}
    assert predicates == {_TREATS, _APPLIED, _CONTRA}


def test_edges_carry_dakp_provenance(kgx_build: KgxBuild) -> None:
    """RAW Tablassert >= 11 edges carry the canonical Translator provenance shape directly.

    NO flat ``primary_knowledge_source`` scalar (removed in Tablassert 11.0): retrieval
    provenance lives only in the ``sources`` RetrievalSource list — the primary entry carries
    the edge family's upstream infores and the table's REAL ``source_record_urls`` (never the
    retired ``example.invalid`` placeholder); each upstream infores also appears as its own
    ``supporting_data_source`` entry.
    """
    assert kgx_build.edges, "build-kg produced no edges"
    for edge in kgx_build.edges:
        # subject/predicate/object are resolved CURIEs.
        for field_name in ("subject", "predicate", "object"):
            value = edge.get(field_name)
            assert isinstance(value, str)
            assert value
        # The flat scalar is gone (Tablassert >= 11); the primary source is the sources entry.
        assert "primary_knowledge_source" not in edge
        # Retrieval provenance: primary entry carries the family's upstream infores + real URLs.
        sources = edge.get("sources")
        assert isinstance(sources, list)
        assert sources
        primary_entry = next(entry for entry in sources if entry.get("resource_role") == "primary_knowledge_source")
        assert primary_entry.get("resource_id") == INFORES_DAKP
        family = EDGE_FAMILIES[edge["predicate"]]
        assert frozenset(primary_entry.get("upstream_resource_ids") or []) == family.required_upstream
        record_urls = primary_entry.get("source_record_urls")
        assert isinstance(record_urls, list)
        assert record_urls
        for url in record_urls:
            assert url.startswith("https://")
            assert "example.invalid" not in url
        # Each upstream infores is its own supporting entry.
        supporting = {entry.get("resource_id") for entry in sources if entry.get("resource_role") == "supporting_data_source"}
        assert supporting == set(family.required_upstream)


def test_dailymed_evidence_lands_where_tablassert_10_puts_it(kgx_build: KgxBuild) -> None:
    """The two DailyMed evidence columns land in their VERIFIED Tablassert >= 10 destinations.

    ``has_evidence`` is a real multivalued slot of the association class every DAKP edge resolves to
    (``ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation``), and the generated configs encode it
    with ``split_by: "|"`` — so DAKP's aggregated, pipe-joined SPL-set cell arrives as a real JSON
    array of DailyMed label URLs, never a one-element list holding a joined ``"a|b"`` blob (the
    silent corruption ``split_by`` exists to prevent). The tiny fixtures carry ONE SPL set per
    assertion row, so the arrays here are single-element; the pipe guard is what fails loudly if
    ``split_by`` is ever dropped from the emitted configs.

    ``supporting_documents`` is on Tablassert's edge-field allow-list (so it is never folded into
    ``supporting_text``) but is NOT a slot of that association class, so ``prune_to_class`` routes it
    onto the inlined supporting study instead of the edge — as per-section DailyMed URLs.
    """
    dailymed_backed = [edge for edge in kgx_build.edges if edge["predicate"] in {_TREATS, _CONTRA}]
    assert dailymed_backed, "expected DailyMed-backed edges in the build"
    for edge in dailymed_backed:
        evidence = edge.get("has_evidence")
        assert isinstance(evidence, list), f"has_evidence must be a JSON array, got {evidence!r}"
        assert evidence
        for value in evidence:
            assert isinstance(value, str)
            assert value.startswith(DAILYMED_SET_URL_BASE), f"has_evidence value is not a DailyMed link: {value!r}"
            assert "|" not in value, f"has_evidence kept a joined cell instead of splitting it: {value!r}"
        # Routed off the edge onto the inlined study, values intact.
        assert "supporting_documents" not in edge
        results = next(iter(edge["has_supporting_studies"].values()))["has_study_results"]
        descriptions = [result.get("description", "") for result in results]
        assert any("supporting_documents=" in description for description in descriptions), descriptions
        assert any(DAILYMED_SET_URL_BASE in description for description in descriptions), descriptions

    # FAERS edges have no SPL evidence at all (the assertion table carries no such column).
    for edge in kgx_build.edges:
        if edge["predicate"] == _APPLIED:
            assert "has_evidence" not in edge


def test_validate_kgx_passes_on_raw_kgx(kgx_build: KgxBuild) -> None:
    """validate_kgx passes on the RAW Tablassert >= 8.2 edges (no canonicalization needed)."""
    report = validate_kgx(kgx_build.nodes, kgx_build.edges)
    assert report.ok, f"validate_kgx reported problems: {report.problems}"
    assert report.problems == []
    assert report.kgx_problems == []
