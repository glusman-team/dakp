"""``dakp_build`` Airflow TaskFlow DAG — the full PLAN.md task graph.

This is a thin wrapper around the same stage functions that
:func:`dakp_pipeline.pipeline.run_pipeline` calls. It is import-safe **without** Airflow
installed: the ``airflow`` and ``pendulum`` imports are guarded by ``try/except
ImportError`` with no-op decorator fallbacks, so the module loads under the base
``uv sync`` (pyright/tested path). Real execution requires ``uv sync --extra airflow``.

Milestone-1 status (per PLAN.md "Milestone 6"): the full 11-task graph is in place and
each task delegates to the real stage function. Production XCom serialization of
``ArtifactRef`` (which carries :class:`~pathlib.Path` fields) is finalized when the DAG
is exercised end-to-end in Milestone 6; for the skeleton, tasks return ``ArtifactRef``
lists and ``params`` carries the workdir/profile/fixture_root.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, medi
from dakp_pipeline.tablassert import configs as tablassert_configs
from dakp_pipeline.tablassert import run as tablassert_run

try:
    from airflow.decorators import dag, task  # type: ignore[import-not-found]
    from pendulum import datetime  # type: ignore[import-not-found]

    _AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the airflow extra
    _AIRFLOW_AVAILABLE = False

    def dag(*_args: Any, **_kwargs: Any) -> Any:
        def _wrap(func: Any) -> Any:
            return func

        return _wrap

    def task(func: Any = None, **_kwargs: Any) -> Any:
        if func is None:

            def _decorator(f: Any) -> Any:
                return f

            return _decorator
        return func

    def datetime(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        import datetime as _stdlib

        return _stdlib.datetime(*args, **kwargs)


def _ctx_from_params(params: Mapping[str, Any]) -> TaskContext:
    """Build a :class:`TaskContext` from Airflow DAG params."""
    p = dict(params or {})
    workdir = Path(p.get("workdir", "data"))
    Workdir(workdir).create()
    fixture_root = Path(p["fixture_root"]) if p.get("fixture_root") else None
    return TaskContext(
        profile=str(p.get("profile", "mock")),
        workdir=workdir,
        fixture_root=fixture_root,
        threads=int(p.get("threads", 1)),
        memory_budget_gb=int(p.get("memory_budget_gb", 1)),
        params=p,
    )


@dag(
    dag_id="dakp_build",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dakp", "drug-approvals"],
    params={"profile": "mock", "fixture_root": "tests/fixtures/pipeline", "workdir": "data", "quarter_limit": 1, "threads": 8},
)
def dakp_build() -> None:
    """Full DAKP build DAG: acquire -> extract -> shape -> Tablassert handoff -> summary."""

    @task
    def acquire_sources(**kwargs: Any) -> dict[str, list[ArtifactRef]]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        return {"dailymed": dailymed.fetch(ctx), "faers": faers.fetch(ctx), "drugsfda": drugsfda.fetch(ctx), "medi": medi.fetch(ctx)}

    @task
    def extract_dailymed(sources: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        return spl_xml.extract(sources["dailymed"], ctx)

    @task
    def extract_faers(sources: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        return faers_ascii.extract(sources["faers"], ctx)

    @task
    def extract_drugsfda(sources: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        return drugsfda_products.extract(sources["drugsfda"], ctx)

    @task
    def extract_medi(sources: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        # MEDI is already a clean table in Milestone 1; extraction is a passthrough here.
        # (params intentionally unused for this no-op stage.)
        _ = kwargs.get("params", {})
        return sources["medi"]

    @task
    def shape_treatment_tables(extracted: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        return approved_treats.transform([*extracted["dailymed"], *extracted["drugsfda"]], ctx)

    @task
    def shape_faers_use_tables(extracted: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        return observed_uses.transform([*extracted["faers"], *extracted["dailymed"]], ctx)

    @task
    def shape_contraindication_tables(extracted: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        return contraindications.transform([*extracted["medi"], *extracted["dailymed"]], ctx)

    @task
    def generate_tablassert_configs(assertions: dict[str, list[ArtifactRef]], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        refs = [*assertions["approved"], *assertions["uses"], *assertions["contra"]]
        return tablassert_configs.generate(refs, ctx)

    @task
    def run_tablassert(assertions: dict[str, list[ArtifactRef]], config_refs: list[ArtifactRef], **kwargs: Any) -> list[ArtifactRef]:
        ctx = _ctx_from_params(kwargs.get("params", {}))
        refs = [*assertions["approved"], *assertions["uses"], *assertions["contra"]]
        return tablassert_run(refs, config_refs, ctx)

    @task
    def write_build_summary(kgx_refs: list[ArtifactRef], **kwargs: Any) -> str:
        # The pure-Python runner owns the canonical summary writer; this task records the
        # handoff path. Full summary aggregation under Airflow lands in Milestone 6.
        params = kwargs.get("params", {})
        workdir = Path(dict(params).get("workdir", "data"))
        summary = workdir / "data" / "reports" / "build_summary.json"
        return str(summary)

    sources = acquire_sources()
    dm = extract_dailymed(sources)
    faers = extract_faers(sources)
    drugsfda = extract_drugsfda(sources)
    medi_ext = extract_medi(sources)
    extracted = {"dailymed": dm, "faers": faers, "drugsfda": drugsfda, "medi": medi_ext}

    approved = shape_treatment_tables(extracted)
    uses = shape_faers_use_tables(extracted)
    contra = shape_contraindication_tables(extracted)
    assertion_inputs = {"approved": approved, "uses": uses, "contra": contra}

    configs = generate_tablassert_configs(assertion_inputs)
    kgx = run_tablassert(assertion_inputs, configs)
    write_build_summary(kgx)


# Register the DAG when Airflow is present (Airflow scans dags/ for module-level DAGs).
if _AIRFLOW_AVAILABLE:
    dag_obj = dakp_build()

__all__ = ["dakp_build"]
