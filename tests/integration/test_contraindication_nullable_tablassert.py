"""Real Tablassert/fullmap smoke for sparse contraindication context."""

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


def _synonym(curie: str, name: str, category: str) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": name, "names": [name], "types": [category], "taxa": ["NCBITaxon:9606"]}


def _class(curie: str) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": []}


def test_generated_contraindication_config_separates_context_and_blank_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Context-bearing and unconditional rows stay DISTINCT edges; only the first is qualified.

    Tablassert 15.1's ``CLASS_FIELD_OVERRIDES`` grant (SkyeAv/Tablassert#120) lets the pinned
    ``EntityToDiseaseAssociation`` keep ``disease_context_qualifier``, and the emitted qualifier is
    ``nullable`` — so "contraindicated in asthma when treating hypertension" and "contraindicated in
    asthma" no longer deduplicate: the qualifier distinguishes them. The qualified edge carries the
    resolved context CURIE; the blank-context row keeps its edge minus only the qualifier.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    (tmp_path / "data" / "tabular").mkdir(parents=True)
    fullmap_root = tmp_path / "fullmap"
    fullmap_root.mkdir()
    classes = fullmap_root / "classes.ndjson"
    synonyms = fullmap_root / "synonyms.ndjson"
    classes.write_text("\n".join(json.dumps(_class(curie)) for curie in ("UNII:GOLD", "MONDO:0004979", "MONDO:0005148")) + "\n")
    synonyms.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _synonym("UNII:GOLD", "DrugGold", "Drug"),
                _synonym("MONDO:0004979", "asthma", "Disease"),
                _synonym("MONDO:0005148", "hypertension", "Disease"),
            )
        )
        + "\n"
    )
    fullmap = fullmap_root / "data" / "fullmap.redb"
    rs.build_fullmap_db(fullmap, [classes], [synonyms], threads=2)

    rows: list[dict[str, str]] = []
    for context, evidence in (
        ("hypertension", "Contraindicated for treatment of hypertension in patients with asthma."),
        ("", "Contraindicated in patients with asthma."),
    ):
        row = dict.fromkeys(schemas.CONTRAINDICATION_COLUMNS, "")
        row.update(
            subject_text="DrugGold",
            object_text="asthma",
            disease_context_text=context,
            evidence_text=evidence,
            predicate="biolink:contraindicated_in",
            subject_category="ChemicalEntity",
            knowledge_level="knowledge_assertion",
            agent_type="manual_validation_of_automated_agent",
            primary_knowledge_source="infores:multiomics-drugapprovals",
            upstream_resource_ids="infores:dailymed",
        )
        rows.append(row)
    pl.DataFrame(rows, schema=schemas.CONTRAINDICATION_COLUMNS).write_csv(
        tmp_path / "data" / "tabular" / "contraindication_assertions.tsv", separator="\t"
    )

    table = tmp_path / "contraindications.yaml"
    table.write_text(dakp_tablassert.table_yaml("contraindication_assertions"), encoding="utf-8")
    graph = tmp_path / "graph.yaml"
    graph.write_text(dakp_tablassert.graph_yaml(["contraindications.yaml"], fullmap=str(fullmap)), encoding="utf-8")

    build_pipeline(graph, PipelineProgress(total_stages=6))
    version = dakp_tablassert.graph_config()["version"]
    edges_path = tmp_path / "data" / f"{dakp_tablassert.GRAPH_NAME}_{version}.edges.ndjson"  # rig.artifact_base_path = "data" (Tablassert >= 11)
    edges = [json.loads(line) for line in edges_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(edges) == 2
    for edge in edges:
        assert (edge["subject"], edge["object"]) == ("UNII:GOLD", "MONDO:0004979")
    qualified = [edge for edge in edges if "disease_context_qualifier" in edge]
    assert len(qualified) == 1
    assert qualified[0]["disease_context_qualifier"] == "MONDO:0005148"
    # The context CURIE appears ONLY as the qualifier on the qualified edge — the unqualified edge
    # must not pick it up through a pruned-column rescue or a ``supporting_text`` fold.
    unqualified = [edge for edge in edges if "disease_context_qualifier" not in edge]
    assert len(unqualified) == 1
    assert not any("MONDO:0005148" in str(value) for value in unqualified[0].values())
    # The SPL evidence prose is deliberately NOT on the edges (full sentences under
    # ``supporting_text`` made them unreadable); ``evidence_text`` stays in the assertion TSV.
    for edge in edges:
        supporting = edge.get("supporting_text") or []
        assert not any("Contraindicated" in entry for entry in supporting)
