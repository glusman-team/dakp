"""DAG wiring tests — import-safe WITHOUT airflow (the base ``uv sync`` path).

``dakp_pipeline.dags.dakp_build`` must import cleanly whether or not apache-airflow is
installed; its pure helpers (:data:`STAGE_CALLABLES`, :data:`DAG_PARAMS`) are always
tested, and the Airflow task graph is asserted only when airflow is importable.
"""

from __future__ import annotations

import pytest

from dakp_pipeline.dags import dakp_build

_REQUIRED_PARAMS = {"profile", "quarter_limit", "force", "fixture_root", "workdir", "log_level"}

_EXPECTED_TASK_IDS = {
    "acquire_dailymed",
    "acquire_faers",
    "acquire_drugsfda",
    "acquire_medi",
    "extract_dailymed",
    "extract_faers",
    "extract_drugsfda",
    "extract_medi",
    "shape_treatment_tables",
    "shape_faers_use_tables",
    "shape_contraindication_tables",
    "generate_tablassert_configs",
    "run_tablassert",
    "write_build_summary",
}


def test_module_imports_without_airflow() -> None:
    # Importing already succeeded at module top; assert the guard resolved and the pure
    # helpers are present regardless of whether airflow is installed.
    assert isinstance(dakp_build._AIRFLOW_AVAILABLE, bool)
    assert dakp_build.DAG_ID == "dakp_build"
    assert callable(dakp_build._ctx_from_params)


def test_dag_params_expose_required_knobs() -> None:
    assert set(dakp_build.DAG_PARAMS) >= _REQUIRED_PARAMS
    assert dakp_build.DAG_PARAMS["profile"] in {"mock", "sample", "wenceslaus_full"}


def test_stage_callable_mapping_matches_run_pipeline() -> None:
    """The wiring manifest is complete and uses the exact callables ``run_pipeline`` uses."""
    from dakp_pipeline import pipeline, tablassert
    from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
    from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
    from dakp_pipeline.sources import dailymed, drugsfda, faers, medi
    from dakp_pipeline.tablassert import configs as tablassert_configs
    from dakp_pipeline.translator import contract as translator_contract

    expected = {
        "acquire_dailymed": dailymed.fetch,
        "acquire_faers": faers.fetch,
        "acquire_drugsfda": drugsfda.fetch,
        "acquire_medi": medi.fetch,
        "extract_dailymed": spl_xml.extract,
        "extract_faers": faers_ascii.extract,
        "extract_drugsfda": drugsfda_products.extract,
        "extract_medi": pipeline._extract_medi,
        "shape_treatment_tables": approved_treats.transform,
        "shape_faers_use_tables": observed_uses.transform,
        "shape_contraindication_tables": contraindications.transform,
        "generate_tablassert_configs": tablassert_configs.generate,
        "run_tablassert": tablassert.run,
        "validate_contract": translator_contract.validate,
        "write_build_summary": pipeline._write_build_summary,
    }
    assert expected == dakp_build.STAGE_CALLABLES
    # Completeness: every PLAN.md stage (acquire/extract/shape/handoff/summary) is wired.
    assert set(dakp_build.STAGE_CALLABLES) == set(expected)


def test_dag_package_helpers_reflect_airflow_state() -> None:
    from dakp_pipeline.dags import airflow_available, get_dag

    assert airflow_available() is dakp_build._AIRFLOW_AVAILABLE
    if airflow_available():
        assert get_dag() is dakp_build.dag_obj
    else:
        with pytest.raises(RuntimeError, match="apache-airflow is not installed"):
            get_dag()


def test_dag_task_graph() -> None:
    if not dakp_build._AIRFLOW_AVAILABLE:
        pytest.skip("apache-airflow not installed; DAG graph construction needs the airflow extra")

    dag = dakp_build.dag_obj
    assert {t.task_id for t in dag.tasks} == _EXPECTED_TASK_IDS

    def upstream(task_id: str) -> set[str]:
        return set(dag.get_task(task_id).upstream_task_ids)

    def downstream(task_id: str) -> set[str]:
        return set(dag.get_task(task_id).downstream_task_ids)

    # Acquisition feeds its own extractor (download -> extract).
    assert upstream("extract_dailymed") == {"acquire_dailymed"}
    assert upstream("extract_faers") == {"acquire_faers"}
    assert upstream("extract_drugsfda") == {"acquire_drugsfda"}
    assert upstream("extract_medi") == {"acquire_medi"}

    # Shapers join the extracts run_pipeline joins (treatment: dm+drugsfda; uses: faers+dm;
    # contraindication: medi+dm).
    assert upstream("shape_treatment_tables") == {"extract_dailymed", "extract_drugsfda"}
    assert upstream("shape_faers_use_tables") == {"extract_faers", "extract_dailymed"}
    assert upstream("shape_contraindication_tables") == {"extract_medi", "extract_dailymed"}

    shapes = {"shape_treatment_tables", "shape_faers_use_tables", "shape_contraindication_tables"}
    assert upstream("generate_tablassert_configs") == shapes
    assert upstream("run_tablassert") == shapes | {"generate_tablassert_configs"}
    assert upstream("write_build_summary") == shapes | {"run_tablassert"}

    # The summary is terminal.
    assert downstream("write_build_summary") == set()
