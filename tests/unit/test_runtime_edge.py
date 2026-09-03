"""Edge-case tests for the Airflow-native runtime helpers (``dakp_pipeline.runtime``).

Covers the small private surface the end-to-end harness skips: the
``runtime._load_disease_map`` missing-file and blank-text-row branches (plus the empty
name/category defaults) and the ``write_build_summary`` ``legacy_tsv`` section (populated vs
the deferred-empty default). Relocated from the retired ``test_pipeline_edge.py`` when the
pure-Python ``run_pipeline`` runner was deleted — ``_load_disease_map`` is live DAG code
and keeps its full branch coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.paths import Workdir
from dakp_pipeline.runtime import _load_disease_map, write_build_summary
from dakp_pipeline.translator import ContractReport, RegressionReport


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


# --- write_build_summary: the legacy_tsv section ----------------------------------


def _ref(path: Path, rows: int) -> ArtifactRef:
    path.write_text("stub\n", encoding="utf-8")
    return ArtifactRef(uri=path, blake3="b3:" + "0" * 64, media_type="text/tab-separated-values", rows=rows)


def test_build_summary_legacy_tsv_section_lists_exported_files(tmp_path: Path) -> None:
    wd = Workdir(tmp_path)
    wd.create()
    nodes = _ref(wd.kgx / "dakp_0.1.0.nodes.tsv", 5)
    edges = _ref(wd.kgx / "dakp_0.1.0.edges.tsv", 12)
    summary_path = write_build_summary(wd, [], [], ContractReport(ok=True), RegressionReport(ok=True), legacy_tsv_refs=[nodes, edges])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["legacy_tsv"] == {
        "exported": True,
        "files": [
            {"name": "dakp_0.1.0.nodes", "path": str(nodes.uri), "rows": 5, "artifact_id": nodes.blake3},
            {"name": "dakp_0.1.0.edges", "path": str(edges.uri), "rows": 12, "artifact_id": edges.blake3},
        ],
    }


def test_build_summary_legacy_tsv_section_defaults_to_not_exported(tmp_path: Path) -> None:
    wd = Workdir(tmp_path)
    wd.create()
    # No legacy_tsv_refs (deferred handoff): exported False, empty files.
    summary_path = write_build_summary(wd, [], [], ContractReport(ok=True), RegressionReport(ok=True))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["legacy_tsv"] == {"exported": False, "files": []}
