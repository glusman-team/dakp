"""Real Tablassert/fullmap smoke for ``uuid_on_collision: merge`` (Tablassert >= 16.2).

Two FAERS observed-use rows name the SAME drug with different raw spellings ("Advil" vs
"Ibuprofen"). The shapers aggregate on raw text, so both rows reach Tablassert; fullmap
resolution then maps both subjects to CHEBI:5855, and the two rows derive ONE edge id under
DAKP's narrowed ``uuid_fields`` (subject/predicate/object + the — here absent — context
qualifier). With ``uuid_on_collision: merge`` the build merges them into a single edge
(list-valued evidence unions sorted; scalars first-wins) instead of aborting with
``uuid-fields-not-a-key``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dakp_pipeline import tablassert as dakp_tablassert
from dakp_pipeline.io import schemas

pytest.importorskip("tablassert")
from tablassert import rs
from tablassert.cli import build_pipeline
from tablassert.progress import PipelineProgress


def _synonym(curie: str, name: str, names: list[str], category: str) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": name, "names": names, "types": [category], "taxa": ["NCBITaxon:9606"]}


def _class(curie: str) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": []}


def test_synonym_spellings_resolving_to_one_curie_merge_into_one_edge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Advil" + "Ibuprofen" rows (both -> CHEBI:5855) for one indication merge into ONE edge."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    (tmp_path / "data" / "tabular").mkdir(parents=True)
    fullmap_root = tmp_path / "fullmap"
    fullmap_root.mkdir()
    classes = fullmap_root / "classes.ndjson"
    synonyms = fullmap_root / "synonyms.ndjson"
    classes.write_text("\n".join(json.dumps(_class(curie)) for curie in ("CHEBI:5855", "HP:0002315")) + "\n")
    synonyms.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                # ONE node, TWO mention spellings — the collision this merge mode exists for.
                _synonym("CHEBI:5855", "ibuprofen", ["Advil", "Ibuprofen"], "SmallMolecule"),
                _synonym("HP:0002315", "headache", ["headache"], "PhenotypicFeature"),
            )
        )
        + "\n"
    )
    fullmap = fullmap_root / "data" / "fullmap.redb"
    rs.build_fullmap_db(fullmap, [classes], [synonyms], threads=2)

    rows: list[dict[str, str]] = []
    for spelling, cases, approvals in (("Advil", "3", "017977"), ("Ibuprofen", "7", "021010")):
        row = dict.fromkeys(schemas.FAERS_APPLIED_TO_TREAT_COLUMNS, "")
        row.update(
            subject_text=spelling,
            object_text="headache",
            predicate="biolink:applied_to_treat",
            subject_category="ChemicalEntity",
            case_count=cases,
            FDA_regulatory_approvals=approvals,
            knowledge_level="statistical_association",
            agent_type="manual_validation_of_automated_agent",
            primary_knowledge_source="infores:faers",
            upstream_resource_ids="infores:faers",
        )
        rows.append(row)
    pl.DataFrame(rows, schema=schemas.FAERS_APPLIED_TO_TREAT_COLUMNS).write_csv(
        tmp_path / "data" / "tabular" / "faers_applied_to_treat_assertions.tsv", separator="\t"
    )

    table = tmp_path / "faers_applied_to_treat.yaml"
    table.write_text(dakp_tablassert.table_yaml("faers_applied_to_treat_assertions"), encoding="utf-8")
    graph = tmp_path / "graph.yaml"
    graph.write_text(dakp_tablassert.graph_yaml(["faers_applied_to_treat.yaml"], fullmap=str(fullmap)), encoding="utf-8")
    # The generated graph config is what turns the collision abort into a merge.
    assert dakp_tablassert.graph_config()["uuid_on_collision"] == "merge"

    build_pipeline(graph, PipelineProgress(total_stages=6))
    version = dakp_tablassert.graph_config()["version"]
    edges_path = tmp_path / "data" / f"{dakp_tablassert.GRAPH_NAME}_{version}.edges.ndjson"  # rig.artifact_base_path = "data" (Tablassert >= 11)
    edges = [json.loads(line) for line in edges_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ONE edge: the two spellings derived the same id and merged instead of aborting.
    assert len(edges) == 1
    edge = edges[0]
    assert (edge["subject"], edge["predicate"], edge["object"]) == ("CHEBI:5855", "biolink:applied_to_treat", "HP:0002315")
    # List-valued evidence unions (sorted); the conflicting scalar case count is first-wins in
    # the dedup stream, whose record order is Tablassert-internal — deterministic per build but
    # not the TSV row order, so either source row may win.
    assert edge["FDA_regulatory_approvals"] == ["017977", "021010"]
    assert int(edge["number_of_cases"]) in (3, 7)
