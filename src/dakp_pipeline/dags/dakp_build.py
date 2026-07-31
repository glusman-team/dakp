"""``dakp_build`` Airflow DAG — the full pipeline task graph (Airflow-native).

This is the **only** orchestrator: the pure-Python ``run_pipeline`` CLI runner is retired (see
plans/airflow-native-go-workers.md). The heavy parsing/extraction runs as **native Airflow Go SDK
bundle workers** (``go/cmd/dakp-bundle``): the three ``extract_*`` tasks are ``@task.stub(queue=
"golang")`` declarations whose Go implementations the ExecutableCoordinator forks per task instance.
Every other stage (acquisition, assertion shaping, Tablassert handoff, translator contract +
regression, build summary) is a real Python TaskFlow task reusing the existing stage modules.

Tasks pass ``list[ArtifactRef]`` manifests over XCom (serialized to JSON dicts via
:mod:`dakp_pipeline.io.xcom` so the native Go workers read/write the same manifests); heavy bytes
move through the BLAKE3 content-addressed filesystem store. Run config (workdir / profile /
fixtures) comes from the ``dakp_config`` Airflow Variable, set by the one-command orchestrator
(``dakp up``) and shared by the Python tasks and the Go bundle.
"""

from __future__ import annotations

from typing import Any

from airflow.models import Variable
from airflow.sdk import dag, task
from pendulum import datetime

from dakp_pipeline import acquire, runtime, tablassert
from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.io.xcom import refs_from_xcom, refs_to_xcom
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import configs as tablassert_configs
from dakp_pipeline.translator import contract as translator_contract
from dakp_pipeline.translator import regression

# --- DAG-level constants ---------------------------------------------------------

DAG_ID = "dakp_build"

#: Queue routed to the Go coordinator (airflow.cfg: [sdk] queue_to_coordinator = {"golang": "go"}).
GO_QUEUE = "golang"
#: Pool bounding concurrent source downloads (network I/O).
DOWNLOAD_POOL = "dakp_download"
#: Pool bounding concurrent raw->interim parses (CPU I/O) — the Go extract stubs run here.
EXTRACT_POOL = "dakp_extract"

#: Airflow Variable (JSON) holding the per-run config (workdir / profile / fixture_root / limits).
CONFIG_VARIABLE = "dakp_config"


def _cfg() -> dict[str, Any]:  # pragma: no cover - reads the Airflow Variable at task runtime
    """Read the per-run config dict from the ``dakp_config`` Variable."""
    return Variable.get(CONFIG_VARIABLE, deserialize_json=True)  # type: ignore[no-any-return]


def _ctx() -> TaskContext:  # pragma: no cover - builds the context at task runtime
    """Build the per-task :class:`TaskContext` from the ``dakp_config`` Variable."""
    return runtime.build_context_from_config(_cfg())


@dag(dag_id=DAG_ID, start_date=datetime(2026, 1, 1), schedule=None, catchup=False, tags=["dakp", "drug-approvals"])
def dakp_build() -> None:  # pragma: no cover - Airflow task graph; task bodies execute only under an Airflow runtime
    """Full DAKP build DAG: acquire -> extract (native Go) -> shape -> Tablassert handoff -> summary."""

    # -- acquisition (Python; download pool) ------------------------------------
    @task(pool=DOWNLOAD_POOL)
    def acquire_dailymed() -> list[dict[str, Any]]:
        return refs_to_xcom(acquire.acquire_dailymed(_ctx()))

    @task(pool=DOWNLOAD_POOL)
    def acquire_faers() -> list[dict[str, Any]]:
        return refs_to_xcom(acquire.acquire_faers(_ctx()))

    @task(pool=DOWNLOAD_POOL)
    def acquire_drugsfda() -> list[dict[str, Any]]:
        return refs_to_xcom(acquire.acquire_drugsfda(_ctx()))

    @task(pool=DOWNLOAD_POOL)
    def acquire_ner_models() -> list[dict[str, Any]]:
        return refs_to_xcom(acquire.acquire_ner_models(_ctx()))

    # -- extraction (native Go SDK bundle workers) ------------------------------
    # No Python body: the ExecutableCoordinator forks the Go bundle, which reads the upstream
    # acquire_* ArtifactRefs from XCom, parses with internal/{dailymed,faers,drugsfda}, writes the
    # interim parquet + TSV handoff into the store, and pushes the output ArtifactRefs as XCom.
    @task.stub(queue=GO_QUEUE, pool=EXTRACT_POOL)
    def extract_dailymed(raw: Any) -> list[dict[str, Any]]: ...

    @task.stub(queue=GO_QUEUE, pool=EXTRACT_POOL)
    def extract_faers(raw: Any) -> list[dict[str, Any]]: ...

    @task.stub(queue=GO_QUEUE, pool=EXTRACT_POOL)
    def extract_drugsfda(raw: Any) -> list[dict[str, Any]]: ...

    # -- assertion shaping (Python) ---------------------------------------------
    @task
    def shape_treatment_tables(dm_ext: Any, drugsfda_ext: Any) -> list[dict[str, Any]]:
        refs = [*refs_from_xcom(dm_ext), *refs_from_xcom(drugsfda_ext)]
        return refs_to_xcom(approved_treats.transform(refs, _ctx()))

    @task
    def shape_faers_use_tables(faers_ext: Any, dm_ext: Any) -> list[dict[str, Any]]:
        refs = [*refs_from_xcom(faers_ext), *refs_from_xcom(dm_ext)]
        return refs_to_xcom(observed_uses.transform(refs, _ctx()))

    @task
    def shape_contraindication_tables(dm_ext: Any, ner_models: Any) -> list[dict[str, Any]]:
        # ``ner_models`` is an ordering dependency only: the NER backend lazily loads weights cached
        # by acquire_ner_models, so mining must run after acquisition — the model refs are not inputs.
        del ner_models
        return refs_to_xcom(contraindications.transform(refs_from_xcom(dm_ext), _ctx()))

    # -- tablassert handoff (Python) --------------------------------------------
    @task
    def generate_tablassert_configs(approved: Any, uses: Any, contra: Any) -> list[dict[str, Any]]:
        refs = [*refs_from_xcom(approved), *refs_from_xcom(uses), *refs_from_xcom(contra)]
        return refs_to_xcom(tablassert_configs.generate(refs, _ctx()))

    @task
    def run_tablassert(approved: Any, uses: Any, contra: Any, configs: Any) -> list[dict[str, Any]]:
        assertion_refs = [*refs_from_xcom(approved), *refs_from_xcom(uses), *refs_from_xcom(contra)]
        return refs_to_xcom(tablassert.run(assertion_refs, refs_from_xcom(configs), _ctx()))

    # -- translator contract + regression + build summary (Python) --------------
    @task
    def write_build_summary(approved: Any, uses: Any, contra: Any, kgx: Any) -> str:
        ctx = _ctx()
        assertion_refs = [*refs_from_xcom(approved), *refs_from_xcom(uses), *refs_from_xcom(contra)]
        report = translator_contract.validate(assertion_refs)
        regression_report = regression.check_assertion_tables(assertion_refs)
        summary = runtime.write_build_summary(Workdir(ctx.workdir), ctx.profile, assertion_refs, refs_from_xcom(kgx), report, regression_report)
        return str(summary)

    # -- wiring -----------------------------------------------------------------
    dm_raw = acquire_dailymed()
    faers_raw = acquire_faers()
    drugsfda_raw = acquire_drugsfda()
    ner_models = acquire_ner_models()

    dm_ext = extract_dailymed(dm_raw)
    faers_ext = extract_faers(faers_raw)
    drugsfda_ext = extract_drugsfda(drugsfda_raw)

    approved = shape_treatment_tables(dm_ext, drugsfda_ext)
    uses = shape_faers_use_tables(faers_ext, dm_ext)
    contra = shape_contraindication_tables(dm_ext, ner_models)

    configs = generate_tablassert_configs(approved, uses, contra)
    kgx = run_tablassert(approved, uses, contra, configs)
    write_build_summary(approved, uses, contra, kgx)


# Register the DAG (Airflow scans the dags folder for module-level DAGs).
dag_obj: Any = dakp_build()

__all__ = ["CONFIG_VARIABLE", "DAG_ID", "DOWNLOAD_POOL", "EXTRACT_POOL", "GO_QUEUE", "dag_obj", "dakp_build"]
