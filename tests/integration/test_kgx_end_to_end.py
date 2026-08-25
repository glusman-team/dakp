"""End-to-end DAKP -> Tablassert -> KGX integration test (requires ``tablassert``).

Proves the FULL path works on a TINY, hermetic fullmap (no network):

1. run the mock pipeline over the tiny pipeline fixtures -> the three DAKP assertion TSVs
   (``data/tabular/*_assertions.tsv``) + the generated ``tables/*.yaml`` Tablassert configs;
2. build a tiny ``fullmap.redb`` (:mod:`tiny_fullmap`) mapping the assertion-table mention text
   (Ibuprofen/Advil, Examplestatin, hypercholesterolemia, headache, pain, asthma, ...) to CURIEs;
3. invoke a REAL ``tablassert build-kg`` through the DAKP :class:`TablassertRunner` (the installed
   ``tablassert`` CLI, real subprocess);
4. load the produced KGX ``DRUG_APPROVALS_KP_1.0.0.{nodes,edges}.ndjson`` and assert: nodes carry
   ``id``/``name``/``category``; edges carry ``subject``/``predicate``/``object`` + DAKP provenance
   (``infores:multiomics-drugapprovals`` primary + the per-family upstream infores); all three edge
   families (``treats`` / ``applied_to_treat`` / ``contraindicated_in``) are present; and
   :func:`dakp_pipeline.translator.validate_kgx` passes.

The whole module SKIPS when ``tablassert`` is not importable (dependencies not yet installed) — it
runs once ``uv sync`` has materialized the runtime. ``tests/`` is outside the coverage ``source``,
so the skip does not affect the 100% ``src/`` coverage gate.

Provenance shape (Tablassert >= 14.0, explicit ``override.sources``, SkyeAv/Tablassert#116): edges
carry NO flat ``primary_knowledge_source`` scalar — retrieval provenance lives only in the
``sources`` list, which replicates the shipped legacy DAKP edge provenance VERBATIM: the
``infores:multiomics-drugapprovals`` wrapper entry (primary for ``treats``, aggregator for
``applied_to_treat``/``contraindicated_in``) carries the gestalt per-edge record URL with the
edge's OWN id resolved in place of ``{edge_id}``; FAERS is the primary entry for
``applied_to_treat`` and ``infores:medi`` for ``contraindicated_in`` (legacy shape parity — this
rebuild has no MEDI module); no entry carries a dataset-level URL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import tiny_fullmap
from harness import install_fixture_fetchers, run_stages

from dakp_pipeline import legacy_tsv
from dakp_pipeline.assertions.evidence import DAILYMED_SET_CURIE_PREFIX
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.tablassert import GESTALT_RECORD_URL_TEMPLATE, TablassertRunner
from dakp_pipeline.translator import INFORES_DAKP, read_kgx_jsonl, validate_kgx

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
    legacy_refs: list[Any]


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

    # (5) Legacy TSV export, exactly as the DAG wires it (run_tablassert -> export_legacy_tsv).
    legacy_refs = legacy_tsv.export(report_refs, ctx)
    return KgxBuild(workdir=work, report=report, nodes=read_kgx_jsonl(node_files[0]), edges=read_kgx_jsonl(edge_files[0]), legacy_refs=legacy_refs)


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


# predicate -> the legacy DAKP ``sources`` shape the explicit ``override.sources`` template
# (Tablassert >= 14.0, SkyeAv/Tablassert#116) stamps on every edge of that family:
# (resource_id, resource_role, upstream_resource_ids-or-None) in legacy entry order. The DAKP
# wrapper entry additionally carries the gestalt record URL with the edge's own id resolved;
# NO entry carries a dataset-level URL.
EXPECTED_SOURCES_BY_PREDICATE: dict[str, list[tuple[str, str, list[str] | None]]] = {
    _TREATS: [
        (INFORES_DAKP, "primary_knowledge_source", ["infores:dailymed", "infores:faers"]),
        ("infores:faers", "supporting_data_source", None),
        ("infores:dailymed", "supporting_data_source", None),
    ],
    _APPLIED: [
        (INFORES_DAKP, "aggregator_knowledge_source", ["infores:dailymed", "infores:faers"]),
        ("infores:faers", "primary_knowledge_source", None),
        ("infores:dailymed", "supporting_data_source", None),
    ],
    _CONTRA: [
        (INFORES_DAKP, "aggregator_knowledge_source", ["infores:dailymed", "infores:medi"]),
        ("infores:medi", "primary_knowledge_source", ["infores:dailymed"]),
        ("infores:dailymed", "supporting_data_source", None),
    ],
}


def test_edges_carry_dakp_provenance(kgx_build: KgxBuild) -> None:
    """RAW Tablassert edges replicate the shipped legacy DAKP provenance shape exactly.

    NO flat ``primary_knowledge_source`` scalar (removed in Tablassert 11.0): retrieval
    provenance lives only in the ``sources`` RetrievalSource list — the explicit
    ``override.sources`` template (SkyeAv/Tablassert#116), which is the legacy shape: the DAKP
    wrapper entry (primary for treats, aggregator for the mined families) carries the gestalt
    per-edge record URL with ``{edge_id}`` resolved to the edge's own id; FAERS is the primary
    entry for ``applied_to_treat`` and ``infores:medi`` for ``contraindicated_in``; no entry
    carries a dataset-level URL (they are irrelevant per-edge).
    """
    assert kgx_build.edges, "build-kg produced no edges"
    for edge in kgx_build.edges:
        # subject/predicate/object are resolved CURIEs.
        for field_name in ("subject", "predicate", "object"):
            value = edge.get(field_name)
            assert isinstance(value, str)
            assert value
        # The flat scalar is gone (Tablassert >= 11); the primary source is a sources entry.
        assert "primary_knowledge_source" not in edge
        sources = edge.get("sources")
        assert isinstance(sources, list)
        assert sources
        # Entry order and roles match the legacy shape exactly; no empty keys anywhere.
        assert [(entry.get("resource_id"), entry.get("resource_role"), entry.get("upstream_resource_ids")) for entry in sources] == [
            (resource_id, role, upstream) for resource_id, role, upstream in EXPECTED_SOURCES_BY_PREDICATE[edge["predicate"]]
        ]
        for entry in sources:
            assert all(value != [] for value in entry.values()), f"empty list leaked onto the edge: {entry}"
            assert "example.invalid" not in str(entry)
        dakp_entry = next(entry for entry in sources if entry.get("resource_id") == INFORES_DAKP)
        # The gestalt viewer deep-link resolved the template to THIS edge's id.
        assert dakp_entry.get("source_record_urls") == [GESTALT_RECORD_URL_TEMPLATE.replace("{edge_id}", edge["id"])]
        # No dataset-level record URLs on any other entry.
        for entry in sources:
            if entry is not dakp_entry:
                assert "source_record_urls" not in entry


def test_edge_evidence_lands_on_the_edge_not_in_a_study(kgx_build: KgxBuild) -> None:
    """DailyMed evidence rides the edge in ONE ``publications`` array.

    ``publications`` is a real multivalued slot of the root ``Association`` class every DAKP edge
    resolves under (whatever ``OBJECT_CATEGORY_OVERRIDE`` pins), and the generated configs encode it
    with ``split_by: "|"`` — so DAKP's aggregated, pipe-joined cell arrives as a real JSON array of
    legacy-form ``dailymed:<spl_set_id>`` identifiers (sorted,
    deduped), never a
    one-element list holding a joined ``"a|b"`` blob (the silent corruption ``split_by`` exists to
    prevent). FAERS report provenance stays in the unannotated ``supporting_faers_*`` debug
    columns and never enters ``publications``.

    The negative half is the point of the change: nothing is relocated into the inlined supporting
    study any more. ``supporting_documents`` (deprecated in Biolink and declared by no association
    class) used to be nulled by ``prune_to_class`` and stringified into a StudyResult
    ``description`` as ``supporting_documents=a, b`` — an unqueryable junk drawer no
    translator-ingests source models.
    """
    dailymed_backed = [edge for edge in kgx_build.edges if edge["predicate"] in {_TREATS, _CONTRA}]
    assert dailymed_backed, "expected DailyMed-backed edges in the build"
    for edge in kgx_build.edges:
        for field in ("FDA_regulatory_approvals", "publications"):
            if field in edge:
                assert isinstance(edge[field], list), f"{field} must be a list: {edge}"
                assert edge[field], f"{field} must be omitted or non-empty: {edge}"
    for edge in dailymed_backed:
        evidence = edge.get("publications")
        assert isinstance(evidence, list), f"publications must be a JSON array, got {evidence!r}"
        assert evidence
        for value in evidence:
            assert isinstance(value, str)
            assert value.startswith(DAILYMED_SET_CURIE_PREFIX), f"unknown publications identifier: {value!r}"
            assert "|" not in value, f"publications kept a joined cell instead of splitting it: {value!r}"
            assert "#" not in value, f"publications kept section granularity instead of set CURIEs: {value!r}"
        assert "supporting_documents" not in edge

    # No DAKP value is stringified into a supporting study any more. Tablassert >= 12 keeps a
    # study result's ``description`` only for REAL rescued/routed values; DAKP's contract is
    # that nothing is ever relocated there, so any study struct present (a bare row-reference
    # study still appears when prune_to_class's rescue machinery engaged on a column no row
    # actually lost) must carry NO description at all.
    for edge in kgx_build.edges:
        for study in (edge.get("has_supporting_studies") or {}).values():
            for result in study.get("has_study_results") or []:
                assert not result.get("description"), f"DAKP value relocated into a study: {result}"

    applied = [edge for edge in kgx_build.edges if edge["predicate"] == _APPLIED]
    assert applied, "expected FAERS applied_to_treat edges"
    for edge in applied:
        # FAERS-only edges carry no publications identifiers at all now.
        evidence = edge.get("publications") or []
        assert all(not value.startswith("faers:") for value in evidence)


def test_faers_case_count_rides_the_edge_as_number_of_cases(kgx_build: KgxBuild) -> None:
    """The FAERS case count rides the edge as ``number_of_cases``, the literal Biolink slot.

    ``statement.category_override`` pins classes that declare it, and Tablassert 15.1's
    ``STUDY_SIZE_EXEMPT_PATTERN`` (SkyeAv/Tablassert#119) stops the study-size classifier from
    renaming the column onto ``Study.study_size`` — before 15.1 the count had to ride as
    ``evidence_count`` because ``coerce.study_size_target("number_of_cases")`` returned
    ``"study_size"``. The last assertion is the guard: if Tablassert ever coerces the name again,
    this keeps failing until DAKP falls back to the alias.

    On the JSON type: Tablassert reads TSV cells as text and numerically coerces only the
    p-value / effect-size / study-size columns (``lib.numeric_columns``), so the count arrives as
    the string ``"1"`` even though ``format_numeric`` already knows ``number_of_cases`` is an int
    slot (``biolink.numeric_slot_kind``). Biolink's pydantic models are lax, so the string still
    validates against the integer-ranged slot. Accept either shape: widening Tablassert's
    ``numeric_columns`` to the count slots turns this into a real int without editing this test.
    """
    from tablassert.coerce import study_size_target

    applied = [edge for edge in kgx_build.edges if edge["predicate"] == _APPLIED]
    assert applied, "expected FAERS applied_to_treat edges in the build"
    for edge in applied:
        count = edge.get("number_of_cases")
        assert isinstance(count, int | str), f"number_of_cases missing or oddly typed: {count!r}"
        assert int(count) > 0
        assert "evidence_count" not in edge
    assert study_size_target("number_of_cases") is None, "Tablassert coerces the name again; fall back to the evidence_count alias"


def test_fda_regulatory_approvals_ride_the_edge_as_a_top_level_json_array(kgx_build: KgxBuild) -> None:
    """``FDA_regulatory_approvals`` reaches the edge as its own top-level JSON array.

    The Biolink slot for FDA application numbers ("numbers that identify specific drug
    applications", multivalued) and the name ``NCATSTranslator/translator-ingests`` maps DAKP's
    approvals onto. DAKP annotates it with ``split_by: "|"`` so the pipe-joined cell is emitted
    as a real JSON array — the legacy ``approvals`` list shape — instead of a joined scalar, and
    never as a folded ``"<name>: <value>"`` ``supporting_text`` string.
    """
    edges = [edge for edge in kgx_build.edges if edge["predicate"] in {_TREATS, _APPLIED, _CONTRA}]
    assert edges, "expected DAKP edges in the build"
    # The fixture build always has approval provenance, so "no edge carries the field" means the
    # value was relocated, not that the source lacked it.
    assert any("FDA_regulatory_approvals" in edge for edge in edges), "no edge carries FDA_regulatory_approvals as its own field"
    for edge in edges:
        approvals = edge.get("FDA_regulatory_approvals")
        if approvals is None:
            # Missing source approval provenance is intentionally omitted, not represented by [].
            continue
        assert isinstance(approvals, list), f"FDA_regulatory_approvals missing or not a JSON array: {approvals!r}"
        assert approvals, "populated FDA_regulatory_approvals must not be an empty array"
        for value in approvals:
            assert isinstance(value, str)
            assert value.strip(), f"empty approval id in {approvals!r}"
            assert "|" not in value, f"FDA_regulatory_approvals kept a joined cell instead of splitting it: {value!r}"
        assert "FDA_regulatory_approvals" not in (edge.get("supporting_text") or "")


def test_validate_kgx_passes_on_raw_kgx(kgx_build: KgxBuild) -> None:
    """validate_kgx passes on the RAW Tablassert >= 8.2 edges (no canonicalization needed)."""
    report = validate_kgx(kgx_build.nodes, kgx_build.edges)
    assert report.ok, f"validate_kgx reported problems: {report.problems}"
    assert report.problems == []
    assert report.kgx_problems == []


# --- legacy TSV retrofit -----------------------------------------------------------


def test_legacy_tsv_pair_matches_the_old_schema(kgx_build: KgxBuild) -> None:
    """The export stage writes the old-service TSV pair with the exact legacy semantics.

    Same stem/directory as the ndjson sources, extension swapped (``.nodes.tsv`` / ``.edges.tsv``);
    node categories are first-element; edge multi-values are comma-joined (`FDA_regulatory_approvals` ->
    `approval`, `publications` -> `supporting_spls`); every absent field is ``NA`` — including
    `object_modifier`, always; endpoint names are the CANONICAL node names (legacy parity).
    """
    data = kgx_build.workdir / "data"
    assert [ref.uri.name for ref in kgx_build.legacy_refs] == [
        next(data.glob("*.nodes.ndjson")).with_suffix(".tsv").name,
        next(data.glob("*.edges.ndjson")).with_suffix(".tsv").name,
    ]
    nodes_ref, edges_ref = kgx_build.legacy_refs
    assert nodes_ref.rows == len(kgx_build.nodes)
    assert edges_ref.rows == len(kgx_build.edges)

    nodes_lines = nodes_ref.uri.read_text(encoding="utf-8").splitlines()
    assert nodes_lines[0].split("\t") == legacy_tsv.NODES_HEADER
    node_names = {line.split("\t")[0]: line.split("\t")[1] for line in nodes_lines[1:]}
    assert set(node_names) == {node["id"] for node in kgx_build.nodes}
    for node in kgx_build.nodes:
        assert f"{node['id']}\t{node['name']}\t{node['category'][0]}" in nodes_lines  # first-element category

    edges_lines = edges_ref.uri.read_text(encoding="utf-8").splitlines()
    assert edges_lines[0].split("\t") == legacy_tsv.EDGES_HEADER
    by_id = {edge["id"]: edge for edge in kgx_build.edges}
    for line in edges_lines[1:]:
        row = dict(zip(legacy_tsv.EDGES_HEADER, line.split("\t"), strict=True))
        edge = by_id[row["id"]]
        assert row["subject"] == edge["subject"]
        assert row["predicate"] == edge["predicate"]
        assert row["object"] == edge["object"]
        assert row["subject_name"] == node_names[edge["subject"]]  # canonical name parity
        assert row["object_name"] == node_names[edge["object"]]
        assert row["object_modifier"] == "NA"  # always NA — legacy parity
        assert row["knowledge_level"] == edge["knowledge_level"]
        assert row["agent_type"] == edge["agent_type"]
        assert row["approval"] == (",".join(edge["FDA_regulatory_approvals"]) if "FDA_regulatory_approvals" in edge else "NA")
        assert row["N_cases"] == (str(edge["number_of_cases"]) if "number_of_cases" in edge else "NA")
        assert row["supporting_spls"] == (",".join(edge["publications"]) if "publications" in edge else "NA")
