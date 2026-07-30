"""Edge-case tests for ``dakp_pipeline.assertions.contraindications`` (drive to 100% branch coverage).

Targets the uncovered lines: backend resolution when the fixture lacks an ontology (85->87),
a contraindication set with no active ingredient (114), a blank mined span (119), ingredient
rows with missing fields / duplicates (145, 148), the object-CURIE-already-set resolution skip
(185->exit), and the empty-scores ``_max_score`` guard (223). Inputs are tiny parquet tables
built in tmp so no ``[ner]`` extra or network is needed.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.assertions.contraindications import (
    _active_ingredients_by_set,
    _category_for_type,
    _max_score,
    build_contraindication_rows,
    resolve_ner_backend,
)
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.ner.backends import CONTRAINDICATION_DISEASE_TYPES, EntitySpan, MockNERBackend, NERBackend

CONTRA_LOINC = "34070-3"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _sections(tmp_path: Path, rows: list[tuple[str, str, str]]) -> ArtifactRef:
    """spl_sections.parquet from (spl_set_id, spl_document_id, text) rows (all contraindication)."""
    frame = pl.DataFrame(
        {
            "spl_set_id": [r[0] for r in rows],
            "spl_document_id": [r[1] for r in rows],
            "clean_text": [r[2] for r in rows],
            "loinc_code": [CONTRA_LOINC for _ in rows],
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


class _BlankSpanBackend:
    """A backend that extracts a single whitespace-only span (to exercise the blank-skip)."""

    def extract(self, text: str, types: object) -> list[EntitySpan]:
        return [EntitySpan(text="   ", start=0, end=3, type="disease", score=1.0)]


# --- backend resolution: fixture without an ontology ----------------------------


def test_resolve_ner_backend_fixture_without_ontology_falls_back_to_mock(tmp_path: Path) -> None:
    # fixture_root exists but has no ontology/disease_map.tsv -> empty MockNERBackend.
    backend = resolve_ner_backend(tmp_path, {})
    assert isinstance(backend, MockNERBackend)
    assert backend.extract("asthma", list(CONTRAINDICATION_DISEASE_TYPES)) == []


def test_resolve_ner_backend_unknown_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown ner_backend"):
        resolve_ner_backend(tmp_path, {"ner_backend_name": "bogus"})


# --- set without an active ingredient is skipped (114) --------------------------


def test_contraindication_set_without_active_ingredient_is_skipped(tmp_path: Path, disease_map: dict[str, dict[str, str]]) -> None:
    sections = _sections(tmp_path, [("SET-X", "SET-X#d", "asthma"), ("SET-Y", "SET-Y#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-Y", "DrugY", "UNII:Y")])  # SET-X has none
    backend = MockNERBackend({"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], backend, disease_map)
    # SET-X is skipped (no active ingredient); only SET-Y -> DrugY mines a row.
    assert [r["subject_text"] for r in rows] == ["DrugY"]
    assert rows[0]["object_text"] == "asthma"


# --- blank mined span is skipped (119) ------------------------------------------


def test_blank_mined_span_is_skipped(tmp_path: Path, disease_map: dict[str, dict[str, str]]) -> None:
    sections = _sections(tmp_path, [("SET-Y", "SET-Y#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-Y", "DrugY", "UNII:Y")])
    rows = build_contraindication_rows([sections, ingredients], _BlankSpanBackend(), disease_map)
    assert rows == []  # the whitespace-only span yields no object_text -> skipped


# --- _active_ingredients_by_set: missing fields + duplicates (145, 148) ---------


def test_active_ingredients_skip_missing_fields_inactive_and_duplicates(tmp_path: Path) -> None:
    ingredients = _ingredients(
        tmp_path,
        [
            ("active", "", "NoSet", "UNII:1"),  # missing set_id -> skipped (145)
            ("active", "SET", "", "UNII:2"),  # missing name -> skipped (145)
            ("inactive", "SET", "Inactive", "UNII:3"),  # not active -> skipped
            ("active", "SET", "DrugY", "UNII:Y"),  # kept
            ("active", "SET", "DrugY", "UNII:Y"),  # exact duplicate -> skipped (148)
            ("active", "SET", "drugy", "UNII:Y"),  # case-insensitive duplicate key -> skipped (148)
            ("active", "SET", "OtherDrug", "UNII:Z"),  # kept
        ],
    )
    by_set = _active_ingredients_by_set([ingredients])
    assert by_set == {"SET": [("DrugY", "UNII:Y"), ("OtherDrug", "UNII:Z")]}  # sorted, deduped


def test_active_ingredients_empty_without_ingredients_table(tmp_path: Path) -> None:
    sections = _sections(tmp_path, [("SET-Y", "SET-Y#d", "asthma")])
    assert _active_ingredients_by_set([sections]) == {}  # no spl_ingredients.parquet present
    assert _active_ingredients_by_set([]) == {}


# --- object-CURIE-already-set resolution skip (185->exit) -----------------------


def test_second_observation_of_same_pair_skips_resolution(tmp_path: Path, disease_map: dict[str, dict[str, str]]) -> None:
    # Two sets share the SAME active ingredient and both mention 'asthma' -> the (DrugX, asthma)
    # aggregate is accumulated twice; the second time object_curie is already set (185->exit).
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "asthma"), ("SET-B", "SET-B#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-B", "DrugX", "UNII:X")])
    backend = MockNERBackend({"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], backend, disease_map)
    assert len(rows) == 1
    row = rows[0]
    assert row["subject_text"] == "DrugX"
    assert row["object_text"] == "asthma"
    assert row["object_curie"] == "MONDO:0004979"  # resolved once, from the disease baseline
    assert row["supporting_spl_sets"] == "SET-A|SET-B"  # both observations unioned
    assert row["source_score"] == "1"


# --- phenotype object category (adversarial) ------------------------------------


def test_phenotype_span_yields_phenotypic_feature_category(tmp_path: Path, disease_map: dict[str, dict[str, str]]) -> None:
    sections = _sections(tmp_path, [("SET-Y", "SET-Y#d", "headache")])
    ingredients = _ingredients(tmp_path, [("active", "SET-Y", "DrugY", "UNII:Y")])
    backend = MockNERBackend({"headache": "phenotype"})
    rows = build_contraindication_rows([sections, ingredients], backend, disease_map)
    assert len(rows) == 1
    # 'headache' resolves in the baseline to a PhenotypicFeature; category follows the baseline.
    assert rows[0]["object_category"] == "PhenotypicFeature"
    assert _category_for_type("phenotype") == "PhenotypicFeature"
    assert _category_for_type("disease") == "Disease"
    assert _category_for_type("chemical") == "Disease"  # anything non-phenotype -> Disease


# --- _max_score guard (223) -----------------------------------------------------


def test_max_score_empty_and_nonempty() -> None:
    assert _max_score([]) == ""  # defensive guard: no scores -> empty string
    assert _max_score([0.5, 0.9, 0.7]) == "0.9"
    assert _max_score([1.0]) == "1"


def test_backend_protocol_conformance_of_test_double() -> None:
    assert isinstance(_BlankSpanBackend(), NERBackend)
