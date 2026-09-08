"""Edge-case tests for the Airflow-native runtime helpers (``dakp_pipeline.runtime``).

Covers the small private surface the end-to-end harness skips: the
``runtime._load_disease_map`` missing-file and blank-text-row branches (plus the empty
name/category defaults). Relocated from the retired ``test_pipeline_edge.py`` when the
pure-Python ``run_pipeline`` runner was deleted — ``_load_disease_map`` is live DAG code
and keeps its full branch coverage.
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.runtime import _load_disease_map


def test_load_disease_map_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _load_disease_map(tmp_path) == {}


def test_load_disease_map_skips_blank_text_and_applies_defaults(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology"
    ontology.mkdir()
    # blank `text` rows are skipped; empty name/category fall back to text / "Disease".
    (ontology / "disease_map.tsv").write_text(
        "text\tcurie\tname\tcategory\nalpha\tMONDO:1\tAlpha\tDisease\n\tMONDO:2\tSkipped\tDisease\nbeta\tMONDO:3\t\t\n", encoding="utf-8"
    )
    mapping = _load_disease_map(tmp_path)
    assert set(mapping) == {"alpha", "beta"}
    assert mapping["alpha"] == {"curie": "MONDO:1", "name": "Alpha", "category": "Disease"}
    # empty name -> defaults to the text; empty category -> defaults to "Disease".
    assert mapping["beta"] == {"curie": "MONDO:3", "name": "beta", "category": "Disease"}
