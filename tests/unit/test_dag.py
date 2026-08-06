"""DAG wiring tests for the Airflow-native ``dakp_build`` DAG (Airflow 3 is a hard dependency).

The DAG always imports and constructs (no optional-extra guard). These tests assert the module
constants, the 13-task graph, that the three ``extract_*`` tasks are native Go SDK stubs routed to
the ``golang`` queue, and that acquisition runs on the download pool.
"""

from __future__ import annotations

# ``dakp_build`` (and thus ``import airflow``) is pulled in lazily via the ``dakp_build`` fixture so
# pytest-xdist workers don't pay the Airflow import during per-worker collection (see conftest.py).

_EXPECTED_TASK_IDS = {
    "acquire_dailymed",
    "acquire_faers",
    "acquire_drugsfda",
    "acquire_ner_models",
    "extract_dailymed",
    "extract_faers",
    "extract_drugsfda",
    "shape_treatment_tables",
    "shape_faers_use_tables",
    "shape_contraindication_tables",
    "generate_tablassert_configs",
    "run_tablassert",
    "write_build_summary",
}

_GO_STUB_IDS = {"extract_dailymed", "extract_faers", "extract_drugsfda"}
_ACQUIRE_IDS = {"acquire_dailymed", "acquire_faers", "acquire_drugsfda", "acquire_ner_models"}


def test_module_constants(dakp_build) -> None:
    assert dakp_build.DAG_ID == "dakp_build"
    assert dakp_build.GO_QUEUE == "golang"
    assert dakp_build.DOWNLOAD_POOL == "dakp_download"
    assert dakp_build.EXTRACT_POOL == "dakp_extract"
    assert dakp_build.CONFIG_VARIABLE == "dakp_config"


def test_dag_object_and_task_ids(dakp_build) -> None:
    dag = dakp_build.dag_obj
    assert dag.dag_id == "dakp_build"
    assert {t.task_id for t in dag.tasks} == _EXPECTED_TASK_IDS


def test_extract_tasks_are_go_stubs_on_golang_queue(dakp_build) -> None:
    dag = dakp_build.dag_obj
    for task_id in _GO_STUB_IDS:
        task = dag.get_task(task_id)
        assert task.queue == dakp_build.GO_QUEUE
        assert task.pool == dakp_build.EXTRACT_POOL
        assert type(task).__name__ == "_StubOperator"


def test_acquisition_tasks_use_download_pool(dakp_build) -> None:
    dag = dakp_build.dag_obj
    for task_id in _ACQUIRE_IDS:
        assert dag.get_task(task_id).pool == dakp_build.DOWNLOAD_POOL


def test_dag_task_graph(dakp_build) -> None:
    dag = dakp_build.dag_obj

    def upstream(task_id: str) -> set[str]:
        return set(dag.get_task(task_id).upstream_task_ids)

    def downstream(task_id: str) -> set[str]:
        return set(dag.get_task(task_id).downstream_task_ids)

    # Acquisition feeds its own extractor (download -> native Go extract).
    assert upstream("extract_dailymed") == {"acquire_dailymed"}
    assert upstream("extract_faers") == {"acquire_faers"}
    assert upstream("extract_drugsfda") == {"acquire_drugsfda"}

    # Shapers join the extracts (treatment: dm+drugsfda+faers; uses: faers+dm; contraindication: dm + NER models).
    assert upstream("shape_treatment_tables") == {"extract_dailymed", "extract_drugsfda", "extract_faers"}
    assert upstream("shape_faers_use_tables") == {"extract_faers", "extract_dailymed"}
    assert upstream("shape_contraindication_tables") == {"extract_dailymed", "acquire_ner_models"}

    shapes = {"shape_treatment_tables", "shape_faers_use_tables", "shape_contraindication_tables"}
    assert upstream("generate_tablassert_configs") == shapes
    assert upstream("run_tablassert") == shapes | {"generate_tablassert_configs"}
    assert upstream("write_build_summary") == shapes | {"run_tablassert"}

    # The summary is terminal.
    assert downstream("write_build_summary") == set()
