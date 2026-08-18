"""Pass 3 tests for ``dakp_pipeline.assertions.contraindications``: warning-section mining.

Covers: evidence indexing of the boxed-warning (LOINC ``34066-1``) and warnings/precautions
(``43685-7``, ``34071-1``, ``42232-9``) sections; hard-trigger-only acceptance for
warning-section mentions (soft caution language and explicit negation rejected); the
singleton-ingredient discipline applied to Pass 3 sets; and production multi-GPU dispatch
routing through ``mine_passes_multi_gpu`` when a third pass has work. Inputs are tiny parquet
tables built in tmp so no heavy NER deps are needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dakp_pipeline.assertions.contraindications import build_contraindication_rows
from dakp_pipeline.assertions.evidence import build_dailymed_evidence
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.ner.ner import DiseaseNER

BOXED_WARNING_LOINC = "34066-1"
WARNINGS_AND_PRECAUTIONS_LOINC = "43685-7"
WARNINGS_LOINC = "34071-1"
PRECAUTIONS_LOINC = "42232-9"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _sections(tmp_path: Path, rows: list[tuple[str, str, str, str]]) -> ArtifactRef:
    """spl_sections.parquet from (spl_set_id, spl_document_id, loinc_code, text) rows."""
    frame = pl.DataFrame(
        {
            "spl_set_id": [r[0] for r in rows],
            "spl_document_id": [r[1] for r in rows],
            "loinc_code": [r[2] for r in rows],
            "clean_text": [r[3] for r in rows],
        }
    )
    path = tmp_path / "spl_sections.parquet"
    frame.write_parquet(path)
    return _ref(path)


def _ingredients(tmp_path: Path, rows: list[tuple[str, str, str, str]]) -> ArtifactRef:
    """spl_ingredients.parquet from (role, spl_set_id, ingredient_name, ingredient_unii) rows."""
    frame = pl.DataFrame(
        {
            "role": [r[0] for r in rows],
            "spl_set_id": [r[1] for r in rows],
            "ingredient_name": [r[2] for r in rows],
            "ingredient_unii": [r[3] for r in rows],
        }
    )
    path = tmp_path / "spl_ingredients.parquet"
    frame.write_parquet(path)
    return _ref(path)


# --- evidence indexing --------------------------------------------------------


def test_evidence_indexes_warning_sections(tmp_path: Path) -> None:
    """The 4 warning LOINCs land in the right evidence maps (boxed vs warnings/precautions)."""
    sections = _sections(
        tmp_path,
        [
            ("SET-BW", "SET-BW#34066-1", BOXED_WARNING_LOINC, "boxed warning text"),
            ("SET-WP", "SET-WP#43685-7", WARNINGS_AND_PRECAUTIONS_LOINC, "warnings and precautions text"),
            ("SET-W", "SET-W#34071-1", WARNINGS_LOINC, "warnings text"),
            ("SET-P", "SET-P#42232-9", PRECAUTIONS_LOINC, "precautions text"),
        ],
    )

    evidence = build_dailymed_evidence([sections])
    assert evidence.boxed_warning_docs == {"SET-BW": [("SET-BW#34066-1", "boxed warning text")]}
    assert evidence.warning_docs == {
        "SET-WP": [("SET-WP#43685-7", "warnings and precautions text")],
        "SET-W": [("SET-W#34071-1", "warnings text")],
        "SET-P": [("SET-P#42232-9", "precautions text")],
    }
    assert evidence.contraindication_docs == {}
    assert evidence.indication_docs == {}


# --- Pass 3 acceptance / rejection --------------------------------------------


def test_pass3_boxed_warning_hard_trigger_accepted(tmp_path: Path) -> None:
    """A boxed warning with hard prohibition language yields an edge with 34066-1 provenance."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#34066-1", BOXED_WARNING_LOINC, "Do not use in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1
    assert rows[0]["subject_text"] == "DrugX"
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#34066-1"


def test_pass3_warnings_section_hard_trigger_accepted(tmp_path: Path) -> None:
    """A warnings-and-precautions section (43685-7) with hard language also yields an edge."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#43685-7", WARNINGS_AND_PRECAUTIONS_LOINC, "Never use in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#43685-7"


def test_pass3_soft_language_rejected(tmp_path: Path) -> None:
    """Soft caution prose in a warning section is NOT promoted to a contraindication edge."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#34066-1", BOXED_WARNING_LOINC, "Use with caution in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    assert build_contraindication_rows([sections, ingredients], ner) == []


def test_pass3_explicit_negation_rejected(tmp_path: Path) -> None:
    """Explicit negation wins even inside a warning section."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#34071-1", WARNINGS_LOINC, "DrugX is not contraindicated in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    assert build_contraindication_rows([sections, ingredients], ner) == []


# --- singleton-ingredient discipline ------------------------------------------


def test_pass3_multi_ingredient_skipped(tmp_path: Path) -> None:
    """Combination products contribute no Pass 3 edges (no single attributable subject)."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#34066-1", BOXED_WARNING_LOINC, "Do not use in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-A", "DrugY", "UNII:Y")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    assert build_contraindication_rows([sections, ingredients], ner) == []


def test_pass3_set_without_ingredients_skipped(tmp_path: Path) -> None:
    """A warning section on a set with no active-ingredient rows contributes nothing."""
    sections = _sections(tmp_path, [("SET-Z", "SET-Z#42232-9", PRECAUTIONS_LOINC, "Do not use in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    assert build_contraindication_rows([sections, ingredients], ner) == []


# --- production dispatch -------------------------------------------------------


def test_build_rows_dispatches_passes_multi_gpu_when_pass3_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Production NER + devices + work in all three passes: mine_passes_multi_gpu is called."""
    sections = _sections(
        tmp_path,
        [
            ("SET-A", "SET-A#34070-3", "34070-3", "asthma"),
            ("SET-B", "SET-B#34067-9", "34067-9", "contraindicated in patients with diabetes"),
            ("SET-C", "SET-C#34066-1", BOXED_WARNING_LOINC, "Do not use in patients with epilepsy."),
        ],
    )
    ingredients = _ingredients(
        tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-B", "DrugY", "UNII:Y"), ("active", "SET-C", "DrugZ", "UNII:Z")]
    )
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease", "diabetes": "disease", "epilepsy": "disease"})

    called: list[dict[str, Any]] = []

    def fake_passes(passes, ner_arg, devs):
        called.append({"passes": [len(p) for p in passes], "devices": tuple(devs)})
        offline = DiseaseNER(gazetteer=ner_arg._gazetteer)
        return {(s, d): offline.extract(t) for p in passes for s, d, t in p}

    import dakp_pipeline.assertions.contraindications as contra_mod

    monkeypatch.setattr(contra_mod, "mine_passes_multi_gpu", fake_passes)

    rows = build_contraindication_rows([sections, ingredients], ner, devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"))
    assert called == [{"passes": [1, 1, 1], "devices": ("cuda:0", "cuda:1", "cuda:2", "cuda:3")}]
    assert {r["subject_text"] for r in rows} == {"DrugX", "DrugY", "DrugZ"}
