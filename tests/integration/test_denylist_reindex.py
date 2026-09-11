"""Real Tablassert ``build-kg`` smoke for the DAKP row-exclusion denylists.

The shape tests in ``tests/unit/test_tablassert_configs.py`` only prove the generated configs
CARRY the ``source.reindex`` ``ne`` filters; this test runs a real build so a Tablassert-side
change to ``reindex``/``idxname`` semantics fails loudly here instead of silently shipping
denied rows to the KG.

Three FAERS observed-use rows go in, exactly one edge comes out:

1. ``Advil`` / ``headache`` SURVIVES — both terms resolve (CHEBI:5855 / HP:0002315).
2. ``Ibuprofen`` / ``hypersensitivity`` is DROPPED by the object denylist — the fullmap resolves
   "hypersensitivity" (MONDO:0000605, Disease) fine, so the drop is attributable to the reindex
   filter, not to an unresolved object.
3. ``*PLACEBO`` / ``headache`` is DROPPED by the subject denylist — likewise the fullmap carries
   the exact ``*PLACEBO`` spelling, so only the filter removes the row.

The fullmap intentionally makes both denied rows resolvable; without the denylist filters the
build would emit three edges, and each surviving-row assertion below fails.
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


def test_denylisted_rows_are_dropped_at_load_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Denied object_text/subject_text rows never reach the built KGX edges."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    (tmp_path / "tabular").mkdir(parents=True)
    fullmap_root = tmp_path / "fullmap"
    fullmap_root.mkdir()
    classes = fullmap_root / "classes.ndjson"
    synonyms = fullmap_root / "synonyms.ndjson"
    curies = ("CHEBI:5855", "HP:0002315", "MONDO:0000605", "UMLS:C1696465")
    classes.write_text("\n".join(json.dumps(_class(curie)) for curie in curies) + "\n")
    synonyms.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _synonym("CHEBI:5855", "ibuprofen", ["Advil", "Ibuprofen"], "SmallMolecule"),
                _synonym("HP:0002315", "headache", ["headache"], "PhenotypicFeature"),
                # Resolvable on purpose: the object denylist — not an empty resolution — must be
                # what removes rows 2-3.
                _synonym("MONDO:0000605", "hypersensitivity reaction disease", ["hypersensitivity"], "Disease"),
                _synonym("UMLS:C1696465", "placebo", ["*PLACEBO", "PLACEBO"], "SmallMolecule"),
            )
        )
        + "\n"
    )
    fullmap = fullmap_root / "kgx" / "fullmap.redb"
    rs.build_fullmap_db(fullmap, [classes], [synonyms])

    rows: list[dict[str, str]] = []
    for subject, obj, cases, case_ids in (
        ("Advil", "headache", "3", "1|2|3"),
        ("Ibuprofen", "hypersensitivity", "5", "11|12|13|14|15"),
        ("*PLACEBO", "headache", "2", "21|22"),
    ):
        row = dict.fromkeys(schemas.FAERS_APPLIED_TO_TREAT_COLUMNS, "")
        row.update(
            subject_text=subject,
            object_text=obj,
            predicate="biolink:applied_to_treat",
            subject_category="ChemicalEntity",
            number_of_cases=cases,
            case_ids=case_ids,
            knowledge_level="statistical_association",
            agent_type="manual_validation_of_automated_agent",
            primary_knowledge_source="infores:multiomics-drugapprovals",
            upstream_resource_ids="infores:dailymed|infores:faers",
        )
        rows.append(row)
    pl.DataFrame(rows, schema=schemas.FAERS_APPLIED_TO_TREAT_COLUMNS).write_csv(
        tmp_path / "tabular" / "faers_applied_to_treat_assertions.tsv", separator="\t"
    )

    table = tmp_path / "faers_applied_to_treat.yaml"
    table.write_text(dakp_tablassert.table_yaml("faers_applied_to_treat_assertions"), encoding="utf-8")
    graph = tmp_path / "graph.yaml"
    graph.write_text(dakp_tablassert.graph_yaml(["faers_applied_to_treat.yaml"], fullmap=str(fullmap)), encoding="utf-8")

    build_pipeline(graph, PipelineProgress(total_stages=6))
    version = dakp_tablassert.graph_config()["version"]
    edges_path = tmp_path / "kgx" / f"{dakp_tablassert.GRAPH_NAME}_{version}.edges.ndjson"
    edges = [json.loads(line) for line in edges_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ONLY the undenied row survives: both denied rows resolved fine against the fullmap, so a
    # Tablassert reindex regression shows up as extra edges here.
    assert len(edges) == 1
    edge = edges[0]
    assert (edge["subject"], edge["predicate"], edge["object"]) == ("CHEBI:5855", "biolink:applied_to_treat", "HP:0002315")
    assert edge["original_subject"] == "Advil"
    assert edge["original_object"] == "headache"
    assert int(edge["number_of_cases"]) == 3
