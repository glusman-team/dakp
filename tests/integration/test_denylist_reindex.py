"""Real Tablassert ``build-kg`` smoke for the DAKP row-exclusion denylists.

The shape tests in ``tests/unit/test_tablassert_configs.py`` only prove the generated configs
CARRY the ``source.reindex`` ``ne`` filters; this test runs a real build so a Tablassert-side
change to ``reindex``/``idxname`` semantics fails loudly here instead of silently shipping
denied rows to the KG.

Seven FAERS observed-use rows go in, exactly three edges come out:

1. ``Advil`` / ``headache`` SURVIVES — both terms resolve (CHEBI:5855 / HP:0002315).
2. ``Ibuprofen`` / ``hypersensitivity`` is DROPPED by the object denylist's allergy/hypersensitivity
   group — the fullmap resolves "hypersensitivity" (MONDO:0000605, Disease) fine, so the drop is
   attributable to the reindex filter, not to an unresolved object.
3. ``*PLACEBO`` / ``headache`` is DROPPED by the subject denylist — likewise the fullmap carries
   the exact ``*PLACEBO`` spelling, so only the filter removes the row.
4. ``Ibuprofen`` / ``Blood pressure`` is DROPPED by the object denylist's non-disease meta-term
   group (resolves to NCIT:C54707 Blood Pressure Finding) — and exercises the SENTENCE-case
   rendering of an entry, the second-most common casing in the deployed KG after lowercase
   (title case never occurs).
5. ``Advil`` / ``disease progression`` is DROPPED by the same group (UMLS:C0242656, a trial
   outcome) via the lowercase rendering.
6. ``Advil`` / ``high blood pressure`` SURVIVES — a QUALIFIED relative of a denied entry that
   names a real disease (MONDO:0005044 hypertensive disorder): the ``ne`` compare is exact
   whole-cell, so denying the bare meta-term must never take a disease-naming form with it.
7. ``Advil`` / ``depression`` SURVIVES as MONDO:0002050 ONLY — the fullmap ties "depression" to
   both the real depressive disorder and the junk phrase concept UMLS:C0812393 ("Cancer patients
   and suicide and depression"), and the object ``exclude_regex`` must drop the junk CURIE while
   the real disease's edge survives (without it, tied-CURIE retention emits the junk edge too).

The fullmap intentionally makes every denied row resolvable; without the denylist filters the
build would emit seven edges, and each assertion below fails.
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
    """Denied object_text/subject_text rows never reach the built KGX edges; qualified relatives do."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    (tmp_path / "tabular").mkdir(parents=True)
    fullmap_root = tmp_path / "fullmap"
    fullmap_root.mkdir()
    classes = fullmap_root / "classes.ndjson"
    synonyms = fullmap_root / "synonyms.ndjson"
    curies = (
        "CHEBI:5855",
        "HP:0002315",
        "MONDO:0000605",
        "UMLS:C1696465",
        "NCIT:C54707",
        "UMLS:C0242656",
        "MONDO:0005044",
        "MONDO:0002050",
        "UMLS:C0812393",
    )
    classes.write_text("\n".join(json.dumps(_class(curie)) for curie in curies) + "\n")
    synonyms.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _synonym("CHEBI:5855", "ibuprofen", ["Advil", "Ibuprofen"], "SmallMolecule"),
                _synonym("HP:0002315", "headache", ["headache"], "PhenotypicFeature"),
                # Resolvable on purpose: the object denylist — not an empty resolution — must be
                # what removes the denied rows below.
                _synonym("MONDO:0000605", "hypersensitivity reaction disease", ["hypersensitivity"], "Disease"),
                _synonym("UMLS:C1696465", "placebo", ["*PLACEBO", "PLACEBO"], "SmallMolecule"),
                _synonym("NCIT:C54707", "Blood Pressure Finding", ["Blood Pressure", "blood pressure"], "PhenotypicFeature"),
                _synonym("UMLS:C0242656", "Disease Progression", ["disease progression"], "Disease"),
                _synonym("MONDO:0005044", "hypertensive disorder", ["high blood pressure"], "Disease"),
                # TIED term: "depression" names BOTH the real disease and the junk phrase concept
                # the object exclude_regex must eliminate.
                _synonym("MONDO:0002050", "depressive disorder", ["depression"], "Disease"),
                _synonym("UMLS:C0812393", "Cancer patients and suicide and depression", ["depression"], "Disease"),
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
        ("Ibuprofen", "Blood pressure", "4", "31|32|33|34"),
        ("Advil", "disease progression", "6", "41|42|43|44|45|46"),
        ("Advil", "high blood pressure", "4", "51|52|53|54"),
        ("Advil", "depression", "5", "61|62|63|64|65"),
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

    # FOUR denied rows never arrive; the three undenied rows do. Every denied row resolved fine
    # against the fullmap, so a Tablassert reindex regression (or an over-matching one) shows up
    # as extra or missing edges here.
    assert len(edges) == 3
    by_object = {edge["object"]: edge for edge in edges}
    assert set(by_object) == {"HP:0002315", "MONDO:0005044", "MONDO:0002050"}
    assert "UMLS:C0812393" not in by_object  # the junk tied CURIE is excluded, never emitted
    edge = by_object["HP:0002315"]
    assert (edge["subject"], edge["predicate"]) == ("CHEBI:5855", "biolink:applied_to_treat")
    assert edge["original_subject"] == "Advil"
    assert edge["original_object"] == "headache"
    assert int(edge["number_of_cases"]) == 3
    # The QUALIFIED relative survives with its real disease object intact.
    qualified = by_object["MONDO:0005044"]
    assert qualified["original_object"] == "high blood pressure"
    assert int(qualified["number_of_cases"]) == 4
    # The tied "depression" row survives ONLY as the real disease, not the junk phrase concept.
    depression = by_object["MONDO:0002050"]
    assert depression["original_object"] == "depression"
    assert int(depression["number_of_cases"]) == 5
