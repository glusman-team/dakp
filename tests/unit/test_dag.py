"""DAG wiring tests for the Airflow-native ``dakp_build`` DAG (Airflow 3 is a hard dependency).

The DAG always imports and constructs (no optional-extra guard). These tests assert the module
constants, the 15-task graph, the visual TaskGroups (with unprefixed/stable task IDs), that the
DailyMed/FAERS/Drugs@FDA ``extract_*`` tasks are native Go SDK stubs routed to the ``golang``
queue while the EMA extract is a plain Python task, and that acquisition/extraction resource
pools are configured for the 50 GB RAM budget.
"""

from __future__ import annotations

# ``dakp_build`` (and thus ``import airflow``) is pulled in lazily via the ``dakp_build`` fixture so
# pytest-xdist workers don't pay the Airflow import during per-worker collection (see conftest.py).

_EXPECTED_TASK_IDS = {
    "acquire_dailymed",
    "acquire_faers",
    "acquire_drugsfda",
    "acquire_ema",
    "acquire_ner_models",
    "extract_dailymed",
    "extract_faers",
    "extract_drugsfda",
    "extract_ema",
    "shape_treatment_tables",
    "shape_faers_use_tables",
    "shape_contraindication_tables",
    "generate_tablassert_configs",
    "run_tablassert",
    "write_build_summary",
}

_GO_STUB_IDS = {"extract_dailymed", "extract_faers", "extract_drugsfda"}
_ACQUIRE_IDS = {"acquire_dailymed", "acquire_faers", "acquire_drugsfda", "acquire_ema", "acquire_ner_models"}
_EXPECTED_GROUP_MEMBERS = {
    "acquire": _ACQUIRE_IDS,
    "extract": _GO_STUB_IDS | {"extract_ema"},
    "shape": {"shape_treatment_tables", "shape_faers_use_tables", "shape_contraindication_tables"},
    "tablassert": {"generate_tablassert_configs", "run_tablassert"},
    "summary": {"write_build_summary"},
}
_EXPECTED_EXTRACT_POOL_SLOTS = {"extract_dailymed": 3, "extract_faers": 3, "extract_drugsfda": 1, "extract_ema": 1}


def test_module_constants(dakp_build) -> None:
    assert dakp_build.DAG_ID == "dakp_build"
    assert dakp_build.GO_QUEUE == "golang"
    assert dakp_build.DOWNLOAD_POOL == "dakp_download"
    assert dakp_build.EXTRACT_POOL == "dakp_extract"
    assert dakp_build.DAILYMED_EXTRACT_POOL_SLOTS == 3
    assert dakp_build.FAERS_EXTRACT_POOL_SLOTS == 3
    assert dakp_build.DRUGSFDA_EXTRACT_POOL_SLOTS == 1
    assert dakp_build.EMA_EXTRACT_POOL_SLOTS == 1
    assert dakp_build.CONFIG_VARIABLE == "dakp_config"


def test_dag_object_and_task_ids(dakp_build) -> None:
    dag = dakp_build.dag_obj
    assert dag.dag_id == "dakp_build"
    assert {t.task_id for t in dag.tasks} == _EXPECTED_TASK_IDS


def test_task_groups_preserve_stable_task_ids(dakp_build) -> None:
    dag = dakp_build.dag_obj
    assert set(dag.task_group.children) == set(_EXPECTED_GROUP_MEMBERS)
    for group_id, members in _EXPECTED_GROUP_MEMBERS.items():
        grouped = {task.task_id for task in dag.tasks if task.task_group.group_id == group_id}
        assert grouped == members
    # prefix_group_id=False keeps Airflow history/log task IDs unchanged.
    assert all("." not in task.task_id for task in dag.tasks)


def test_extract_tasks_are_go_stubs_on_golang_queue(dakp_build) -> None:
    dag = dakp_build.dag_obj
    for task_id in _GO_STUB_IDS:
        task = dag.get_task(task_id)
        assert task.queue == dakp_build.GO_QUEUE
        assert task.pool == dakp_build.EXTRACT_POOL
        assert task.pool_slots == _EXPECTED_EXTRACT_POOL_SLOTS[task_id]
        assert type(task).__name__ == "_StubOperator"


def test_extract_ema_is_a_python_task_on_the_extract_pool(dakp_build) -> None:
    # The EMA xlsx parse stays in Python (polars): a real task, not a Go stub.
    task = dakp_build.dag_obj.get_task("extract_ema")
    assert type(task).__name__ != "_StubOperator"
    assert task.queue != dakp_build.GO_QUEUE
    assert task.pool == dakp_build.EXTRACT_POOL
    assert task.pool_slots == _EXPECTED_EXTRACT_POOL_SLOTS["extract_ema"]


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

    # Acquisition feeds its own extractor (download -> native Go extract / Python EMA parse).
    assert upstream("extract_dailymed") == {"acquire_dailymed"}
    assert upstream("extract_faers") == {"acquire_faers"}
    assert upstream("extract_drugsfda") == {"acquire_drugsfda"}
    assert upstream("extract_ema") == {"acquire_ema"}

    # Shapers join the extracts (treatment: dm+drugsfda+faers+ema + NER models for EPAR mining; uses: faers+dm; contraindication: dm + NER models).
    assert upstream("shape_treatment_tables") == {"extract_dailymed", "extract_drugsfda", "extract_faers", "extract_ema", "acquire_ner_models"}
    assert upstream("shape_faers_use_tables") == {"extract_faers", "extract_dailymed"}
    assert upstream("shape_contraindication_tables") == {"extract_dailymed", "acquire_ner_models"}

    shapes = {"shape_treatment_tables", "shape_faers_use_tables", "shape_contraindication_tables"}
    assert upstream("generate_tablassert_configs") == shapes
    assert upstream("run_tablassert") == shapes | {"generate_tablassert_configs"}
    assert upstream("write_build_summary") == shapes | {"run_tablassert"}

    # The summary is terminal.
    assert downstream("write_build_summary") == set()
