"""Unit tests for DAKP RIG generation (Milestone 8).

Asserts the generated RIG matches the structure/conventions of
``../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml``: required top-level sections,
the DAKP source infores, DailyMed/FAERS/MEDI supporting sources, the three edge families with
chemical/drug -> disease/phenotype category compatibility (kept in sync with the KGX contract),
and node identifier types. YAML serialization is dependency-free; validity is checked via an
optional ``yaml`` round-trip when the library happens to be present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline.translator import rig
from dakp_pipeline.translator.contract import CHEMICAL_DRUG_CATEGORIES, DISEASE_PHENOTYPE_CATEGORIES
from dakp_pipeline.translator.rig import generate_rig, rig_text, rig_yaml, write_rig

_REQUIRED_SECTIONS = {"name", "supporting_data_source_info", "source_info", "ingest_info", "target_info", "provenance_info"}


def test_generate_rig_has_required_sections() -> None:
    content = generate_rig()
    assert set(content) == _REQUIRED_SECTIONS
    assert content["name"] == rig.RIG_NAME


def test_source_info_is_the_dakp_infores() -> None:
    source = generate_rig()["source_info"]
    assert source["infores_id"] == "infores:multiomics-drugapprovals"
    assert "kgx" in source["data_formats"]
    assert source["terms_of_use_info"]["terms_of_use_description"]


def test_supporting_data_sources_are_dailymed_faers_medi() -> None:
    sources = generate_rig()["supporting_data_source_info"]
    assert [source["infores_id"] for source in sources] == ["infores:dailymed", "infores:faers", "infores:medi"]
    for source in sources:
        assert source["name"]
        assert source["description"]
        assert source["terms_of_use_info"]["terms_of_use_description"]
        assert source["relevant_files"]


def test_edge_type_info_covers_the_three_families_in_order() -> None:
    edge_type_info = generate_rig()["target_info"]["edge_type_info"]
    assert [entry["predicates"] for entry in edge_type_info] == [["biolink:treats"], ["biolink:applied_to_treat"], ["biolink:contraindicated_in"]]
    for entry in edge_type_info:
        # Category compatibility matches the KGX contract exactly (no drift).
        assert entry["subject_categories"] == list(CHEMICAL_DRUG_CATEGORIES)
        assert entry["object_categories"] == list(DISEASE_PHENOTYPE_CATEGORIES)
        assert entry["knowledge_level"] == ["knowledge_assertion"]
        assert entry["agent_type"] == ["text_mining_agent"]
        assert entry["ui_explanation"]


def test_node_type_info_covers_chemical_and_disease_nodes() -> None:
    node_type_info = generate_rig()["target_info"]["node_type_info"]
    by_category = {entry["node_category"]: entry["source_identifier_types"] for entry in node_type_info}
    assert set(by_category) >= {
        "biolink:ChemicalEntity",
        "biolink:SmallMolecule",
        "biolink:MolecularMixture",
        "biolink:ComplexMolecularMixture",
        "biolink:Disease",
        "biolink:PhenotypicFeature",
    }
    assert by_category["biolink:Disease"] == ["MONDO"]
    assert by_category["biolink:PhenotypicFeature"] == ["HP"]
    assert "CHEBI" in by_category["biolink:ChemicalEntity"]


def test_rig_yaml_is_deterministic_and_carries_required_markers() -> None:
    text = rig_yaml()
    assert text == rig_yaml()  # deterministic
    for marker in (
        "supporting_data_source_info:",
        "source_info:",
        "ingest_info:",
        "target_info:",
        "edge_type_info:",
        "node_type_info:",
        "provenance_info:",
        "infores:multiomics-drugapprovals",
        "infores:dailymed",
        "infores:faers",
        "infores:medi",
        "biolink:treats",
        "biolink:applied_to_treat",
        "biolink:contraindicated_in",
    ):
        assert marker in text


def test_rig_yaml_round_trips_when_yaml_is_available() -> None:
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(rig_yaml()) == generate_rig()


def test_rig_text_matches_generated_yaml() -> None:
    assert rig_text() == rig_yaml(generate_rig())


def test_write_rig_writes_yaml_and_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "translator" / "dakp_rig.yaml"
    written = write_rig(target)
    assert written == target
    assert target.exists()
    assert target.read_text(encoding="utf-8") == rig_yaml()
    assert target.read_text(encoding="utf-8").startswith('name: "')
