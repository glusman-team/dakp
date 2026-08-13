"""Real Tablassert/fullmap smoke for nullable contraindication context."""

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


def test_generated_contraindication_config_keeps_nullable_and_qualified_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real redb resolves one context and nullable keeps the blank-context row."""
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
            agent_type="text_mining_agent",
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
    edges_path = tmp_path / f"dakp_{version}.edges.ndjson"
    edges = [json.loads(line) for line in edges_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(edges) == 2
    assert {edge.get("disease_context_qualifier") for edge in edges} == {"MONDO:0005148", None}
    assert all("supporting_text" in edge for edge in edges)
