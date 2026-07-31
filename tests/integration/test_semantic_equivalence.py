"""Semantic-equivalence verification: the NEW DAKP preserves the OLD DAKP's knowledge semantics.

The legacy DAKP build (``ref/legacy/bin/drug2indi2kg.py``, ``ref/legacy/bin/uselist2kg.py``,
``ref/legacy/bin/dakp-postprocess2jsonlBL.py``, ``ref/legacy/matrix/bin/contraindications2kg.py``)
established a precise Translator contract: three edge families, fixed subject/object categories,
a per-family provenance chain under ``infores:multiomics-drugapprovals``, a
``clinical_approval_status`` rule, and FDA-approval / FAERS-case / SPL evidence fields. This
module asserts the rebuild **preserves** those semantics.

This is a semantic-*preservation* guardrail, not edge-for-edge equality with the legacy build.
The rebuild deliberately improves coverage and mappings (PLAN.md "allowing improved coverage and
improved mappings"; :mod:`dakp_pipeline.translator.regression`); the invariants locked here are the
family / predicate / category / provenance / label / evidence contracts the old KP established and
the DINGO reference ingest (``../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml``) publishes.

The deliberate, documented *differences* (contraindications NER-mined from DailyMed instead of the
MEDI/Matrix xlsx; the preserved FAERS ``observed_use``/``statistical_association`` label instead of
the legacy heuristic ``off_label_use``; ontology mapping delegated to Tablassert/fullmap) are
asserted here as the NEW behavior and explained in ``docs/semantic-equivalence.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from harness import install_fixture_fetchers, run_stages

from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.translator import contract, regression

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"

# --- the three DAKP edge families (legacy + DINGO) --------------------------------

TREATS = "biolink:treats"
APPLIED_TO_TREAT = "biolink:applied_to_treat"
CONTRAINDICATED_IN = "biolink:contraindicated_in"

# --- expected DINGO ``dakp_rig.yaml`` category contract --------------------------
# Mirrored from ../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml so the test stays
# green without the sibling repo present.

DINGO_SUBJECT_CATEGORIES: tuple[str, ...] = (
    "biolink:ChemicalEntity",
    "biolink:SmallMolecule",
    "biolink:MolecularMixture",
    "biolink:ComplexMolecularMixture",
    "biolink:Drug",
)
DINGO_OBJECT_CATEGORIES: tuple[str, ...] = ("biolink:Disease", "biolink:PhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")

# Legacy subject/object category tuples (ref/legacy/bin/*.py ``interventionCategories`` /
# ``conditionCategories``). The rebuild's chemical/drug contract is the legacy intervention set
# minus the protein/mixture edge cases the DINGO RIG does not publish; the condition set is identical.
LEGACY_CONDITION_CATEGORIES: tuple[str, ...] = ("Disease", "PhenotypicFeature")


# --- pipeline fixture (module-scoped: run the offline fixture pipeline once) ------


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the offline fixture pipeline (fetchers monkeypatched, no Tablassert) once."""
    workdir = tmp_path_factory.mktemp("semantic-equiv")
    # Fetchers always run their real branches; route them to the fixtures for this offline build.
    # The patch is undone once the tables are built (the byte-determinism re-run re-installs it).
    monkeypatch = pytest.MonkeyPatch()
    install_fixture_fetchers(monkeypatch)
    try:
        result = run_stages(fixture_root=_FIXTURE_ROOT, workdir=workdir)
    finally:
        monkeypatch.undo()
    tables: dict[str, pl.DataFrame] = {}
    refs: list[ArtifactRef] = []
    for name in ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions"):
        table = result.table(name)
        # infer_schema_length=0 -> every column read as the exact stored string (no int coercion).
        tables[name] = pl.read_csv(table.path, separator="\t", infer_schema_length=0)
    # Recover the registered ArtifactRefs from the build summary's table list.
    import json

    assert result.build_summary is not None, "offline pipeline produced no build summary"
    summary = json.loads(result.build_summary.read_text(encoding="utf-8"))
    for entry in summary["tables"]:
        refs.append(ArtifactRef(uri=Path(entry["path"]), blake3=entry["artifact_id"], media_type="text/tab-separated-values", rows=entry["rows"]))
    return {"result": result, "tables": tables, "refs": refs, "workdir": workdir}


def _family_rows(tables: dict[str, pl.DataFrame], predicate: str) -> list[dict[str, str]]:
    """All assertion rows carrying ``predicate`` across the three tables."""
    rows: list[dict[str, str]] = []
    for frame in tables.values():
        rows.extend(rec for rec in frame.iter_rows(named=True) if rec.get("predicate") == predicate)
    return rows


# --- 1. the three edge families are all produced ----------------------------------


def test_all_three_edge_families_present(built: dict[str, Any]) -> None:
    """The rebuild emits all three legacy edge families (treats / applied_to_treat / contraindicated_in)."""
    tables = built["tables"]
    assert _family_rows(tables, TREATS), "no biolink:treats rows produced"
    assert _family_rows(tables, APPLIED_TO_TREAT), "no biolink:applied_to_treat rows produced"
    assert _family_rows(tables, CONTRAINDICATED_IN), "no biolink:contraindicated_in rows produced"


def test_regression_invariants_pass_and_see_every_family(built: dict[str, Any]) -> None:
    """The legacy-informed regression guardrail passes and observes all three families."""
    report = regression.check_assertion_tables(built["refs"])
    assert report.ok, f"regression violations: {[v.message for v in report.violations]}"
    assert set(report.families_seen) == {TREATS, APPLIED_TO_TREAT, CONTRAINDICATED_IN}
    assert report.row_count > 0


def test_assertion_tables_satisfy_column_contract(built: dict[str, Any]) -> None:
    """Every public assertion table exists with its declared Translator column contract."""
    report = contract.validate(built["refs"])
    assert report.ok, f"contract problems: {report.problems}"


# --- 2. subject + object categories match the legacy/DINGO contract ---------------


def test_subject_categories_are_chemical_drug(built: dict[str, Any]) -> None:
    """Every subject carries a chemical/drug category (legacy ``interventionCategories``)."""
    chemical = {cat.removeprefix("biolink:") for cat in DINGO_SUBJECT_CATEGORIES}
    for frame in built["tables"].values():
        categories = {str(rec.get("subject_category")) for rec in frame.iter_rows(named=True)}
        assert categories, "empty subject category set"
        assert categories <= chemical, f"non-chemical subject categories: {categories - chemical}"


def test_object_categories_are_disease_or_phenotype(built: dict[str, Any]) -> None:
    """Resolved objects carry Disease/PhenotypicFeature (legacy ``conditionCategories``).

    Contraindication objects are mined mention TEXT with the category left empty for
    Tablassert/fullmap (an intentional delegation, not a regression); the resolved families
    (treats / applied_to_treat) must carry a legacy condition category on every object.
    """
    for predicate in (TREATS, APPLIED_TO_TREAT):
        for rec in _family_rows(built["tables"], predicate):
            assert str(rec.get("object_category")) in LEGACY_CONDITION_CATEGORIES, (
                f"{predicate} object category {rec.get('object_category')!r} not in {LEGACY_CONDITION_CATEGORIES}"
            )


# --- 3. provenance chains match the legacy postprocess + DINGO ingest -------------


def test_provenance_primary_is_always_dakp(built: dict[str, Any]) -> None:
    """Every family aggregates under infores:multiomics-drugapprovals as primary_knowledge_source."""
    for frame in built["tables"].values():
        primaries = {str(rec.get("primary_knowledge_source")) for rec in frame.iter_rows(named=True)}
        assert primaries == {contract.INFORES_DAKP}, f"unexpected primary sources: {primaries}"


@pytest.mark.parametrize(
    ("predicate", "expected_upstream"),
    [
        # legacy dakp-postprocess2jsonlBL.py ``sources`` blocks + DINGO dakp_rig.yaml:
        pytest.param(TREATS, {"infores:dailymed", "infores:faers"}, id="treats-dailymed+faers"),
        pytest.param(APPLIED_TO_TREAT, {"infores:faers", "infores:dailymed"}, id="applied_to_treat-faers+dailymed"),
        # IMPROVEMENT: contraindications are DailyMed-NER-mined, so upstream is dailymed (not medi).
        pytest.param(CONTRAINDICATED_IN, {"infores:dailymed"}, id="contraindicated_in-dailymed"),
    ],
)
def test_provenance_upstream_per_family(built: dict[str, Any], predicate: str, expected_upstream: set[str]) -> None:
    """Each family carries exactly the legacy/DINGO upstream infores chain (order-insensitive)."""
    for rec in _family_rows(built["tables"], predicate):
        upstream = {token for token in str(rec.get("upstream_resource_ids")).split("|") if token}
        assert upstream == expected_upstream, f"{predicate} upstream {upstream} != {expected_upstream}"


def test_medi_is_not_a_provenance_source_anywhere(built: dict[str, Any]) -> None:
    """IMPROVEMENT: MEDI/Matrix is gone — no edge cites infores:medi (contraindications are DailyMed-NER)."""
    for frame in built["tables"].values():
        for rec in frame.iter_rows(named=True):
            upstream = str(rec.get("upstream_resource_ids"))
            assert "infores:medi" not in upstream, f"unexpected medi provenance: {upstream}"
            assert str(rec.get("primary_knowledge_source")) != "infores:medi"


# --- 4. clinical_approval_status + knowledge_level + agent_type logic -------------


def test_treats_clinical_approval_status_is_approved_for_condition(built: dict[str, Any]) -> None:
    """Legacy postprocess set treats -> approved_for_condition; the rebuild preserves it exactly."""
    for rec in _family_rows(built["tables"], TREATS):
        assert str(rec.get("clinical_approval_status")) == "approved_for_condition"
        assert str(rec.get("knowledge_level")) == "knowledge_assertion"
        assert str(rec.get("agent_type")) == "manual_validation_of_automated_agent"


def test_applied_to_treat_preserves_the_faers_label(built: dict[str, Any]) -> None:
    """applied_to_treat keeps the FAERS label/status (observed_use / statistical_association).

    Documented refinement of the legacy heuristic ``off_label_use`` (see docs/semantic-equivalence.md):
    the rebuild preserves the FAERS-derived label rather than inferring off-label use.
    """
    for rec in _family_rows(built["tables"], APPLIED_TO_TREAT):
        assert str(rec.get("clinical_approval_status")) == "observed_use"
        assert str(rec.get("knowledge_level")) == "statistical_association"
        assert str(rec.get("agent_type")) == "manual_validation_of_automated_agent"


def test_contraindications_are_knowledge_assertions_text_mined(built: dict[str, Any]) -> None:
    """contraindicated_in is a knowledge_assertion mined from DailyMed (agent_type text_mining_agent)."""
    for rec in _family_rows(built["tables"], CONTRAINDICATED_IN):
        assert str(rec.get("knowledge_level")) == "knowledge_assertion"
        assert str(rec.get("agent_type")) == "text_mining_agent"


# --- 5. evidence fields: FDA approval/NDA, FAERS case counts, SPL support ---------


def test_treats_carries_fda_approval_and_spl_evidence(built: dict[str, Any]) -> None:
    """Legacy ``approval`` (NDA) + ``supporting_spls`` survive as approval_ids + supporting_spl_*."""
    for rec in _family_rows(built["tables"], TREATS):
        assert str(rec.get("approval_ids")).strip(), "treats row missing FDA approval/NDA id"
        assert str(rec.get("supporting_spl_sets")).strip(), "treats row missing supporting SPL set"
        assert str(rec.get("supporting_spl_documents")).strip(), "treats row missing supporting SPL document"


def test_applied_to_treat_carries_faers_case_counts(built: dict[str, Any]) -> None:
    """Legacy ``N_cases`` survives as a positive case_count on every applied_to_treat row."""
    for rec in _family_rows(built["tables"], APPLIED_TO_TREAT):
        assert int(str(rec.get("case_count"))) > 0, "applied_to_treat row missing FAERS case count"


def test_contraindications_carry_spl_support(built: dict[str, Any]) -> None:
    """Contraindication rows carry the DailyMed SPL set/document they were mined from."""
    for rec in _family_rows(built["tables"], CONTRAINDICATED_IN):
        assert str(rec.get("supporting_spl_sets")).strip(), "contraindication row missing supporting SPL set"
        assert str(rec.get("supporting_spl_documents")).strip(), "contraindication row missing supporting SPL document"


# --- 6. deterministic output (precondition for Tablassert's deterministic edge ids)


def test_rows_uniquely_keyed_by_subject_predicate_object(built: dict[str, Any]) -> None:
    """No duplicate (subject, predicate, object) triples — the legacy saveEdge dedup invariant.

    The legacy build deduped edges on (subj, pred, obj) and derived a deterministic uuid3 edge id
    from that triple (``namespace_uuid('drug_approvals_kp', subj, pred, obj)``). The rebuild keeps
    the triple unique + deterministically ordered so Tablassert's deterministic id machinery is stable.
    """
    for frame in built["tables"].values():
        triples = [(str(r.get("subject_text")), str(r.get("predicate")), str(r.get("object_text"))) for r in frame.iter_rows(named=True)]
        assert len(triples) == len(set(triples)), f"duplicate (subject,predicate,object) triple in {frame.columns}"


def test_output_is_byte_deterministic_across_runs(built: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running the identical pipeline reproduces byte-identical assertion TSVs.

    Deterministic bytes in -> deterministic edge ids out (Tablassert derives UUIDs from the
    resolved triples). This is the rebuild's equivalent of the legacy uuid3 edge-id stability.
    """
    install_fixture_fetchers(monkeypatch)
    rerun = run_stages(fixture_root=_FIXTURE_ROOT, workdir=tmp_path / "rerun")
    for name in ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions"):
        first = built["result"].table(name).path.read_bytes()
        second = rerun.table(name).path.read_bytes()
        assert first == second, f"{name} output is not byte-deterministic across runs"


# --- 7. cross-check the Translator contract against the DINGO RIG -----------------


def test_contract_categories_match_dingo_rig() -> None:
    """contract.py's subject/object category tuples equal the DINGO dakp_rig.yaml categories."""
    assert contract.CHEMICAL_DRUG_CATEGORIES == DINGO_SUBJECT_CATEGORIES
    assert contract.DISEASE_PHENOTYPE_CATEGORIES == DINGO_OBJECT_CATEGORIES


def test_contract_edge_families_match_dingo_predicates() -> None:
    """contract.EDGE_FAMILIES publishes exactly the three DINGO predicates + upstream chains."""
    assert tuple(contract.EDGE_FAMILIES) == (TREATS, APPLIED_TO_TREAT, CONTRAINDICATED_IN)
    assert contract.EDGE_FAMILIES[TREATS].required_upstream == frozenset({"infores:dailymed", "infores:faers"})
    assert contract.EDGE_FAMILIES[APPLIED_TO_TREAT].required_upstream == frozenset({"infores:faers", "infores:dailymed"})
    assert contract.EDGE_FAMILIES[CONTRAINDICATED_IN].required_upstream == frozenset({"infores:dailymed"})


# --- 8. the produced rows satisfy the full KGX/Translator contract ----------------


def _edge_category(object_category: str) -> str:
    """Legacy ``dakp-postprocess2jsonlBL.py`` edge-category derivation from the object category."""
    if object_category == "Disease":
        return "biolink:EntityToDiseaseAssociation"
    if object_category == "PhenotypicFeature":
        return "biolink:EntityToPhenotypicFeatureAssociation"
    return f"biolink:EntityTo{object_category}Association"


def _node_id(curie: str, text: str) -> str:
    """A stable node id: the source CURIE when present, else a deterministic text placeholder.

    Mirrors the resolution Tablassert/fullmap performs for mention text (contraindication objects
    and FAERS subjects carry no source CURIE in the assertion table).
    """
    return curie if curie.strip() else f"TEXT:{text.strip()}"


def _synthesize_kgx(tables: dict[str, pl.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build KGX node/edge records from the assertion tables, simulating Tablassert resolution.

    Object CURIEs/categories left empty by the shapers (contraindication mentions) are resolved to
    a deterministic id + the shaper's own default category (Disease), exactly as fullmap would before
    KGX emission — so this validates the contract the *resolved* graph must satisfy.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, name: str, category: str) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "name": name, "category": [f"biolink:{category}"]}

    for frame in tables.values():
        for rec in frame.iter_rows(named=True):
            subject_category = str(rec.get("subject_category") or "ChemicalEntity")
            object_category = str(rec.get("object_category")) or "Disease"  # fullmap default for mined mentions
            subject_id = _node_id(str(rec.get("subject_curie")), str(rec.get("subject_text")))
            object_id = _node_id(str(rec.get("object_curie")), str(rec.get("object_text")))
            add_node(subject_id, str(rec.get("subject_name")) or str(rec.get("subject_text")), subject_category)
            add_node(object_id, str(rec.get("object_name")) or str(rec.get("object_text")), object_category)

            upstream = [token for token in str(rec.get("upstream_resource_ids")).split("|") if token]
            edges.append(
                {
                    "id": f"edge:{subject_id}:{rec.get('predicate')}:{object_id}",
                    "subject": subject_id,
                    "predicate": str(rec.get("predicate")),
                    "object": object_id,
                    "category": [_edge_category(object_category)],
                    "knowledge_level": str(rec.get("knowledge_level")),
                    "agent_type": str(rec.get("agent_type")),
                    "primary_knowledge_source": str(rec.get("primary_knowledge_source")),
                    "sources": [
                        {
                            "resource_id": str(rec.get("primary_knowledge_source")),
                            "resource_role": "primary_knowledge_source",
                            "upstream_resource_ids": upstream,
                        }
                    ],
                }
            )
    return list(nodes.values()), edges


def test_produced_rows_pass_the_kgx_translator_contract(built: dict[str, Any]) -> None:
    """After (simulated) resolution, every node/edge passes contract.validate_kgx.

    This is the deepest semantic check: node coverage, biolink-prefixed categories, the three
    edge families with chemical/drug subjects + disease/phenotype objects, and the per-family
    infores provenance chain — the same gate the DINGO ingest applies to the published KGX.
    """
    nodes, edges = _synthesize_kgx(built["tables"])
    assert edges, "no edges synthesized from the assertion tables"
    report = contract.validate_kgx(nodes, edges)
    assert report.ok, f"KGX contract problems: {report.problems}"
