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

Twelve more rows exercise the v1.24.1 audit additions (water / bare eye-and-clear / GHB):

8. ``water`` / ``headache`` is DROPPED by the SUBJECT ``exclude_regex`` — the fixture resolves
   "water" to CHEBI:15377 (SmallMolecule) fine, so only the water-CURIE exclusion removes the
   row; the wording carries no subject denylist entry.
9. ``WATER PILL`` / ``headache`` is DROPPED by the subject WORDING denylist while 10.
   ``diuretic`` / ``headache`` SURVIVES on the same UMLS:C0012798 (Diuretics) — the class CURIE
   is legitimate, only the unnamed-diuretic placeholder wording is denied.
11. ``sodium oxybate`` / ``headache`` is DROPPED (its only candidate CHEBI:30830 is excluded)
   while 12. ``4-HYDROXYBUTANOIC ACID`` / ``headache`` SURVIVES as UMLS:C0673093 (biolink:Drug)
   — the exclusion must fall the tied wording through to the correct identifier, not just
   delete the edge.
13. ``Advil`` / ``clear`` is DROPPED by the object wording denylist (resolves NCIT:C48621
   Negative Surgical Margin) while 14. ``Advil`` / ``Clear cell renal cell carcinoma`` SURVIVES
   (MONDO:0005005) — the qualified relative keeps its own edges.
15. ``Advil`` / ``eye`` is DROPPED (resolves MONDO:0005328 + MONDO:0004659) while 16.
   ``Advil`` / ``Eye disorder`` SURVIVES on the very same MONDO:0005328 — proof the eye denial
   is wording-level, not CURIE-level.
17. ``Advil`` / ``watery eyes`` is DROPPED while 18. ``Advil`` / ``excessive tearing`` SURVIVES
   on the same MONDO:0001793, and 19. ``Advil`` / ``water intoxication`` is DROPPED (resolves
   MONDO:0022007) — the water-string object denials, same wording-only shape.

The fullmap intentionally makes every denied row resolvable; without the denylist filters the
build would emit seven edges, and each assertion below fails.
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
        "CHEBI:15377",
        "UMLS:C0012798",
        "CHEBI:30830",
        "CHEBI:42639",
        "UMLS:C0673093",
        "NCIT:C48621",
        "MONDO:0005005",
        "MONDO:0005328",
        "MONDO:0004659",
        "MONDO:0001793",
        "MONDO:0022007",
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
                # v1.24.1 audit fixtures — resolvable on purpose so each drop below is
                # attributable to a denylist filter or CURIE exclusion, not empty resolution.
                _synonym("CHEBI:15377", "Water", ["water"], "SmallMolecule"),
                _synonym("UMLS:C0012798", "Diuretics", ["WATER PILL", "diuretic"], "Drug"),
                # TIED GHB family: "4-hydroxybutanoic acid" names the excluded CHEBI pair AND
                # the correct Drug identifier the subject exclude_regex must fall through to.
                _synonym("CHEBI:30830", "gamma-Hydroxybutyric acid", ["sodium oxybate", "4-hydroxybutanoic acid"], "SmallMolecule"),
                _synonym("CHEBI:42639", "gamma-Butyrolactone", ["4-hydroxybutanoic acid"], "SmallMolecule"),
                _synonym("UMLS:C0673093", "4-hydroxybutyric acid", ["4-hydroxybutanoic acid"], "Drug"),
                _synonym("NCIT:C48621", "Negative Surgical Margin", ["clear"], "PhenotypicFeature"),
                _synonym("MONDO:0005005", "clear cell carcinoma", ["Clear cell renal cell carcinoma"], "Disease"),
                # TIED eye family: "eye" reaches BOTH eye CURIEs; "Eye disorder" the shared
                # MONDO:0005328 alone, proving the denial is wording-level, not CURIE-level.
                _synonym("MONDO:0005328", "eye disorder", ["eye", "Eye disorder"], "Disease"),
                _synonym("MONDO:0004659", "eye carcinoma in situ", ["eye"], "Disease"),
                _synonym("MONDO:0001793", "excessive tearing", ["watery eyes", "excessive tearing"], "Disease"),
                _synonym("MONDO:0022007", "water intoxication", ["water intoxication"], "Disease"),
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
        ("water", "headache", "7", "71|72|73|74|75|76|77"),
        ("WATER PILL", "headache", "8", "81|82|83|84|85|86|87|88"),
        ("diuretic", "headache", "2", "91|92"),
        ("sodium oxybate", "headache", "3", "101|102|103"),
        ("4-HYDROXYBUTANOIC ACID", "headache", "2", "111|112"),
        ("Advil", "clear", "4", "121|122|123|124"),
        ("Advil", "Clear cell renal cell carcinoma", "2", "131|132"),
        ("Advil", "eye", "5", "141|142|143|144|145"),
        ("Advil", "Eye disorder", "3", "151|152|153"),
        ("Advil", "watery eyes", "2", "161|162"),
        ("Advil", "excessive tearing", "2", "171|172"),
        ("Advil", "water intoxication", "1", "181"),
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
            primary_knowledge_source="infores:drugapprovals-kp",
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

    # Same guard as test_kgx_end_to_end's in-process build: Tablassert spawns two cpu-count-sized
    # fork Pools per build, and fork inherits xdist thread locks. Pin the section pools to
    # spawn/Pool(1); ingestion, fullmap resolution and KGX compilation all stay real.
    monkeypatch.setattr("tablassert.cli.Pool", lambda: multiprocessing.get_context("spawn").Pool(processes=1))
    build_pipeline(graph, PipelineProgress(total_stages=6))
    version = dakp_tablassert.graph_config()["version"]
    edges_path = tmp_path / "kgx" / f"{dakp_tablassert.GRAPH_NAME}_{version}.edges.ndjson"
    edges = [json.loads(line) for line in edges_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # TWELVE denied/unresolved-family rows never arrive; the eight survivors do. Every denied
    # row resolved fine against the fullmap, so a Tablassert reindex/exclusion regression (or an
    # over-matching one) shows up as extra or missing edges here.
    assert len(edges) == 8
    by_pair = {(edge["subject"], edge["object"]): edge for edge in edges}
    assert set(by_pair) == {
        ("CHEBI:5855", "HP:0002315"),
        ("CHEBI:5855", "MONDO:0005044"),
        ("CHEBI:5855", "MONDO:0002050"),
        ("UMLS:C0012798", "HP:0002315"),
        ("UMLS:C0673093", "HP:0002315"),
        ("CHEBI:5855", "MONDO:0005005"),
        ("CHEBI:5855", "MONDO:0005328"),
        ("CHEBI:5855", "MONDO:0001793"),
    }
    assert all(edge["object"] != "UMLS:C0812393" for edge in edges)  # the junk tied CURIE is excluded, never emitted
    edge = by_pair[("CHEBI:5855", "HP:0002315")]
    assert (edge["subject"], edge["predicate"]) == ("CHEBI:5855", "biolink:applied_to_treat")
    assert edge["original_subject"] == "Advil"
    assert edge["original_object"] == "headache"
    assert int(edge["number_of_cases"]) == 3
    # The QUALIFIED relative survives with its real disease object intact.
    qualified = by_pair[("CHEBI:5855", "MONDO:0005044")]
    assert qualified["original_object"] == "high blood pressure"
    assert int(qualified["number_of_cases"]) == 4
    # The tied "depression" row survives ONLY as the real disease, not the junk phrase concept.
    depression = by_pair[("CHEBI:5855", "MONDO:0002050")]
    assert depression["original_object"] == "depression"
    assert int(depression["number_of_cases"]) == 5

    # v1.24.1 audit guarantees: no water/GHB/eye/clear identifier or wording survives anywhere.
    subjects = {edge["subject"] for edge in edges}
    assert not subjects & {"CHEBI:15377", "CHEBI:30830", "CHEBI:42639"}
    assert {edge["original_subject"] for edge in edges} == {"Advil", "diuretic", "4-HYDROXYBUTANOIC ACID"}
    assert {edge["original_object"] for edge in edges} == {
        "headache",
        "high blood pressure",
        "depression",
        "Clear cell renal cell carcinoma",
        "Eye disorder",
        "excessive tearing",
    }
    # The GHB-excluded wording FALLS THROUGH to the correct Drug identifier (not deleted).
    ghb = by_pair[("UMLS:C0673093", "HP:0002315")]
    assert ghb["original_subject"] == "4-HYDROXYBUTANOIC ACID"
    assert int(ghb["number_of_cases"]) == 2
    # The diuretic-class CURIE keeps its legitimate wording while the placeholder is denied.
    diuretic = by_pair[("UMLS:C0012798", "HP:0002315")]
    assert diuretic["original_subject"] == "diuretic"
    assert int(diuretic["number_of_cases"]) == 2
    # "Eye disorder" rides the very CURIE (MONDO:0005328) the bare "eye" wording was denied on.
    eye_disorder = by_pair[("CHEBI:5855", "MONDO:0005328")]
    assert eye_disorder["original_object"] == "Eye disorder"
    assert int(eye_disorder["number_of_cases"]) == 3
    # "excessive tearing" rides the very CURIE (MONDO:0001793) "watery eyes" was denied on.
    tearing = by_pair[("CHEBI:5855", "MONDO:0001793")]
    assert tearing["original_object"] == "excessive tearing"
    assert int(tearing["number_of_cases"]) == 2
