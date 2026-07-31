"""Edge-case tests for ``dakp_pipeline.assertions.contraindications`` (drive to 100% branch coverage).

Targets: ``default_ner`` fallback when the fixture lacks an ontology (and when fixture_root is
None); a contraindication set with no active ingredient; a blank mined span; ingredient rows with
missing fields / duplicates in the shared evidence cache; a second observation of the same
(subject, object) pair unioning support; the empty-scores ``_max_score`` guard; and the shaper honoring / ignoring an injected
``params["ner"]``. Inputs are tiny parquet tables built in tmp so no heavy NER deps are needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.assertions.contraindications import ContraindicationsShaper, _max_score, build_contraindication_rows, default_ner
from dakp_pipeline.assertions.evidence import build_dailymed_evidence
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.ner import DiseaseNER, Mention
from dakp_pipeline.paths import Workdir

CONTRA_LOINC = "34070-3"
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


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


class _BlankNER(DiseaseNER):
    """A backend that extracts a single whitespace-only span (to exercise the blank-skip)."""

    def extract(self, text: str) -> list[Mention]:
        return [Mention(text="   ", start=0, end=3, type="disease", score=1.0)]


def _ctx(tmp_path: Path, params: Mapping[str, Any]) -> TaskContext:
    context = TaskContext(workdir=tmp_path / "work", fixture_root=FIXTURE_ROOT, params=params)
    Workdir(context.workdir).create()
    return context


# --- default_ner fallbacks ------------------------------------------------------


def test_default_ner_fixture_without_ontology_falls_back_to_embedded(tmp_path: Path) -> None:
    # fixture_root exists but has no ontology/disease_map.tsv -> embedded-gazetteer backend.
    ner = default_ner(tmp_path)
    assert isinstance(ner, DiseaseNER)
    # The embedded gazetteer still recognizes common terms.
    assert [m.text for m in ner.extract("asthma")] == ["asthma"]


def test_default_ner_none_fixture_uses_embedded() -> None:
    ner = default_ner(None)
    assert isinstance(ner, DiseaseNER)
    assert [m.text for m in ner.extract("headache")] == ["headache"]


# --- set without an active ingredient is skipped --------------------------------


def test_contraindication_set_without_active_ingredient_is_skipped(tmp_path: Path) -> None:
    sections = _sections(tmp_path, [("SET-X", "SET-X#d", "asthma"), ("SET-Y", "SET-Y#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-Y", "DrugY", "UNII:Y")])  # SET-X has none
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    # SET-X is skipped (no active ingredient); only SET-Y -> DrugY mines a row.
    assert [r["subject_text"] for r in rows] == ["DrugY"]
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["object_curie"] == ""


# --- blank mined span is skipped ------------------------------------------------


def test_blank_mined_span_is_skipped(tmp_path: Path) -> None:
    sections = _sections(tmp_path, [("SET-Y", "SET-Y#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-Y", "DrugY", "UNII:Y")])
    assert build_contraindication_rows([sections, ingredients], _BlankNER()) == []  # whitespace-only -> skipped


# --- DailyMedEvidence.active_ingredients_by_set: missing fields + duplicates ----


def test_active_ingredients_skip_missing_fields_inactive_and_duplicates(tmp_path: Path) -> None:
    ingredients = _ingredients(
        tmp_path,
        [
            ("active", "", "NoSet", "UNII:1"),  # missing set_id -> skipped
            ("active", "SET", "", "UNII:2"),  # missing name -> skipped
            ("inactive", "SET", "Inactive", "UNII:3"),  # not active -> skipped
            ("active", "SET", "DrugY", "UNII:Y"),  # kept
            ("active", "SET", "DrugY", "UNII:Y"),  # exact duplicate -> skipped
            ("active", "SET", "drugy", "UNII:Y"),  # case-insensitive duplicate key -> skipped
            ("active", "SET", "OtherDrug", "UNII:Z"),  # kept
        ],
    )
    evidence = build_dailymed_evidence([ingredients])
    assert evidence.active_ingredients_by_set == {"SET": [("DrugY", "UNII:Y"), ("OtherDrug", "UNII:Z")]}  # sorted, deduped
    assert evidence.set_ingredient == {"SET": ("DrugY", "UNII:Y")}  # first active ingredient retained for treatment fallback


def test_active_ingredients_empty_without_ingredients_table(tmp_path: Path) -> None:
    sections = _sections(tmp_path, [("SET-Y", "SET-Y#d", "asthma")])
    assert build_dailymed_evidence([sections]).active_ingredients_by_set == {}  # no spl_ingredients.parquet present
    assert build_dailymed_evidence([]).active_ingredients_by_set == {}


# --- second observation of the same pair unions support -------------------------


def test_second_observation_of_same_pair_unions_support(tmp_path: Path) -> None:
    # Two sets share the SAME active ingredient and both mention 'asthma' -> one aggregated row.
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "asthma"), ("SET-B", "SET-B#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-B", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1
    row = rows[0]
    assert row["subject_text"] == "DrugX"
    assert row["object_text"] == "asthma"
    assert row["object_curie"] == ""
    assert row["supporting_spl_sets"] == "SET-A|SET-B"  # both observations unioned
    assert row["source_score"] == "1"


# --- _max_score guard -----------------------------------------------------------


def test_max_score_empty_and_nonempty() -> None:
    assert _max_score([]) == ""  # defensive guard: no scores -> empty string
    assert _max_score([0.5, 0.9, 0.7]) == "0.9"
    assert _max_score([1.0]) == "1"


# --- shaper: injected / ignored ner param ---------------------------------------


def test_shaper_uses_injected_ner_param(tmp_path: Path) -> None:
    from dakp_pipeline.extract import spl_xml

    ctx = _ctx(tmp_path, {"ner": DiseaseNER(gazetteer={"asthma": "disease", "liver disease": "disease"})})
    refs = spl_xml.extract([_ref(FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz")], ctx)
    out = ContraindicationsShaper().transform(refs, ctx)
    assert len(out) == 1
    # The injected gazetteer mines BOTH contraindication sections.
    from dakp_pipeline.io import schemas

    subjects = sorted(schemas.read_table(out[0].uri)["subject_text"].to_list())
    assert subjects == ["Examplestatin", "Ibuprofen"]


def test_shaper_ignores_non_backend_ner_param_and_falls_back(tmp_path: Path) -> None:
    from dakp_pipeline.extract import spl_xml
    from dakp_pipeline.io import schemas

    # A non-DiseaseNER "ner" param is ignored; the shaper falls back to default_ner(fixture_root).
    ctx = _ctx(tmp_path, {"ner": "not a backend"})
    refs = spl_xml.extract([_ref(FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz")], ctx)
    out = ContraindicationsShaper().transform(refs, ctx)
    frame = schemas.read_table(out[0].uri)
    assert frame.height == 1  # fixture gazetteer -> Ibuprofen -> asthma only
    assert frame.row(0, named=True)["subject_text"] == "Ibuprofen"
