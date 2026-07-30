"""Edge-case tests for the pipeline result types + runtime helpers.

Covers the small public surface the end-to-end tests skip: ``TableResult.exists()``, the
``PipelineResult.table()`` KeyError path (both the "<none>" and the populated-available
messages), the ``runtime.write_build_summary`` regression-violation serialization, and the
``runtime._load_disease_map`` empty-file / blank-text-row branches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline.config import load_profile
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.paths import Workdir
from dakp_pipeline.pipeline import PipelineResult, TableResult
from dakp_pipeline.runtime import _load_disease_map, write_build_summary
from dakp_pipeline.translator.contract import ContractReport
from dakp_pipeline.translator.regression import RegressionReport, RegressionViolation


def test_table_result_exists_reflects_filesystem(tmp_path: Path) -> None:
    present = tmp_path / "here.tsv"
    present.write_text("x\n")
    assert TableResult("here", present, 1).exists() is True
    assert TableResult("gone", tmp_path / "gone.tsv", 0).exists() is False


def test_pipeline_result_table_unknown_name_empty(tmp_path: Path) -> None:
    result = PipelineResult(workdir=Workdir(tmp_path), profile=load_profile("mock"))
    with pytest.raises(KeyError, match=r"available: <none>"):
        result.table("missing")


def test_pipeline_result_table_unknown_name_lists_available(tmp_path: Path) -> None:
    table = TableResult("approved_treats_assertions", tmp_path / "a.tsv", 0)
    result = PipelineResult(workdir=Workdir(tmp_path), profile=load_profile("mock"), tables={"approved_treats_assertions": table})
    assert result.table("approved_treats_assertions") is table
    with pytest.raises(KeyError, match="approved_treats_assertions"):
        result.table("nope")


def test_write_build_summary_serializes_regression_violations(tmp_path: Path) -> None:
    wd = Workdir(tmp_path)
    wd.create()
    ref = ArtifactRef(uri=wd.tabular / "approved_treats_assertions.tsv", blake3="b3:aa", media_type="text/tab-separated-values", rows=1)
    report = ContractReport(ok=True, tables={"approved_treats_assertions": {"rows": 1, "missing_columns": []}})
    regression = RegressionReport(
        ok=False,
        families_seen=["biolink:treats"],
        row_count=1,
        violations=[RegressionViolation("biolink:treats", "knowledge_level", "1 row(s): expected 'knowledge_assertion'")],
    )
    summary = write_build_summary(wd, "mock", [ref], [], report, regression)
    text = summary.read_text(encoding="utf-8")
    assert '"translator_regression"' in text
    assert '"knowledge_level"' in text
    assert "expected 'knowledge_assertion'" in text


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
