"""Real Tablassert/fullmap smoke for animal-label rows: species partitions edges, taxa do not delete them.

Species reaches the KGX output through the ``species_context_qualifier`` slot and the graph
``uuid_fields``, and DAKP deliberately declares NO ``taxon_from`` anywhere. The fixture is shaped
like the PRODUCTION fullmap measured on 2026-10-21 (``tablassert quick_map`` against
``/local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap``), because that shape is what makes the choice
load-bearing:

* ``OrganismTaxon`` rows carry ``TAXON_ID 0``, so their resolved taxon metadata is nulled
  (``fullmap._join_matches_eager``) and a species column cannot scope resolution: a build with
  ``taxon_from: species_context_qualifier`` dies with ``taxon-from-not-ncbitaxon``.
* Disease concepts carry the HUMAN taxon (``asthma`` -> ``MONDO:0004979`` at ``TAXON_ID 9606``)
  while the filter keeps only ``TAXON_ID == <scope> or 0``, so scoping the non-nullable object
  under a dog taxon returns zero rows and deletes the animal edge instead of partitioning it.

So this test asserts the contract that actually holds: a dog row and a human row over the same
drug and condition both build, resolve to the same drug and condition CURIEs, and remain TWO
edges with distinct ids whose species qualifiers differ. Anyone re-adding object scoping breaks
the dog row and sees ``len(edges) == 1`` here.
"""

from __future__ import annotations

import json
import multiprocessing
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


def _synonym(curie: str, name: str, names: list[str], category: str, taxa: list[str]) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": name, "names": names, "types": [category], "taxa": taxa}


def _class(curie: str) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": []}


def test_animal_and_human_rows_partition_by_species_without_taxon_scoping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dog contraindication and a human one build as two edges, over production-shaped taxa."""
    monkeypatch.setattr("tablassert.cli.Pool", lambda: multiprocessing.get_context("spawn").Pool(processes=1))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    (tmp_path / "tabular").mkdir(parents=True)
    fullmap_root = tmp_path / "fullmap"
    fullmap_root.mkdir()
    classes = fullmap_root / "classes.ndjson"
    synonyms = fullmap_root / "synonyms.ndjson"
    curies = ("CHEBI:5855", "MONDO:0004979", "NCBITaxon:9606", "NCBITaxon:9615")
    classes.write_text("\n".join(json.dumps(_class(curie)) for curie in curies) + "\n", encoding="utf-8")
    synonyms.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                # Production chemicals carry no taxon.
                _synonym("CHEBI:5855", "ibuprofen", ["ibuprofen"], "SmallMolecule", ["NCBITaxon:0"]),
                # Production diseases carry the HUMAN taxon: scoping the object by a dog row's
                # species would resolve nothing and drop the edge.
                _synonym("MONDO:0004979", "asthma", ["asthma"], "Disease", ["NCBITaxon:9606"]),
                # Production OrganismTaxon rows carry TAXON_ID 0, which is why a species column
                # cannot be a taxon_from source.
                _synonym("NCBITaxon:9606", "Homo sapiens (human)", ["NCBITaxon:9606"], "OrganismTaxon", ["NCBITaxon:0"]),
                _synonym("NCBITaxon:9615", "Canis lupus familiaris (dog)", ["NCBITaxon:9615"], "OrganismTaxon", ["NCBITaxon:0"]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    fullmap = fullmap_root / "kgx" / "fullmap.redb"
    rs.build_fullmap_db(fullmap, [classes], [synonyms])

    rows: list[dict[str, str]] = []
    for species in ("NCBITaxon:9615", "NCBITaxon:9606"):
        row = dict.fromkeys(schemas.CONTRAINDICATION_COLUMNS, "")
        row.update(
            subject_text="ibuprofen",
            object_text="asthma",
            predicate="biolink:contraindicated_in",
            subject_category="ChemicalEntity",
            species_context_text=species,
            knowledge_level="knowledge_assertion",
            agent_type="manual_validation_of_automated_agent",
            primary_knowledge_source="infores:drugapprovals-kp",
            upstream_resource_ids="infores:dailymed",
        )
        rows.append(row)
    pl.DataFrame(rows, schema=schemas.CONTRAINDICATION_COLUMNS).write_csv(tmp_path / "tabular" / "contraindication_assertions.tsv", separator="\t")

    table = tmp_path / "contraindications.yaml"
    table.write_text(dakp_tablassert.table_yaml("contraindication_assertions"), encoding="utf-8")
    graph = tmp_path / "graph.yaml"
    graph.write_text(dakp_tablassert.graph_yaml(["contraindications.yaml"], fullmap=str(fullmap)), encoding="utf-8")
    statement = dakp_tablassert.table_config("contraindication_assertions")["statement"]
    assert "taxon_from" not in statement["subject"]
    assert "taxon_from" not in statement["object"]
    species_slots = [entry for entry in statement["qualifiers"] if entry["qualifier"] == dakp_tablassert.SPECIES_QUALIFIER]
    assert len(species_slots) == 1
    assert species_slots[0]["taxon"] is None  # the human default is lifted

    build_pipeline(graph, PipelineProgress(total_stages=6))
    version = dakp_tablassert.graph_config()["version"]
    edges_path = tmp_path / "kgx" / f"{dakp_tablassert.GRAPH_NAME}_{version}.edges.ndjson"
    edges = [json.loads(line) for line in edges_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Both species build: the dog row survives its own species qualifier and the human-taxonomed object.
    assert len(edges) == 2, f"expected one edge per species, got {len(edges)} (a dropped animal row?)"
    by_species = {edge["species_context_qualifier"]: edge for edge in edges}
    assert set(by_species) == {"NCBITaxon:9615", "NCBITaxon:9606"}
    for edge in by_species.values():
        assert (edge["subject"], edge["predicate"], edge["object"]) == ("CHEBI:5855", "biolink:contraindicated_in", "MONDO:0004979")
    # Same drug, same condition, different species: two edges, never one folded edge.
    assert by_species["NCBITaxon:9615"]["id"] != by_species["NCBITaxon:9606"]["id"]
