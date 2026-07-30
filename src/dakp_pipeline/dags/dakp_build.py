"""``dakp_build`` Airflow TaskFlow DAG — the full PLAN.md task graph (Milestone 6).

This is a thin TaskFlow wrapper around the **same stage functions** that
:func:`dakp_pipeline.pipeline.run_pipeline` calls, so DAG behavior is identical to the
canonical Airflow-independent runner. Tasks pass :class:`ArtifactRef` manifest metadata
(paths + BLAKE3 ids), never in-memory dataframes.

Import-safe **without** Airflow: the ``airflow``/``pendulum`` imports are guarded by
``try/except ImportError`` with no-op decorator fallbacks, so the module (and its pure
helpers — :data:`STAGE_CALLABLES`, :data:`DAG_PARAMS`, :func:`_ctx_from_params`) loads
under the base ``uv sync``. Real execution requires ``uv sync --extra airflow``.

Design notes
------------
* **Single source of truth.** Every task body calls through :data:`STAGE_CALLABLES`, a
  flat ``stage name -> leaf callable`` manifest mirroring ``run_pipeline``'s stages.
  ``tests/unit/test_dag.py`` asserts the manifest is complete and matches the runner.
* **Delegation, not duplication.** Context construction and the build summary reuse
  ``run_pipeline``'s own helpers (``pipeline._build_context``,
  ``pipeline._write_build_summary``) so results are identical. They are reached via
  attribute access on the ``pipeline`` module (not ``from … import _private``).
* **Pools (conceptual).** Every acquisition task (DailyMed / Drugs@FDA / FAERS source
  downloads, NER model caching, and ontology/fullmap acquisition) runs on the
  ``dakp_download`` pool and extraction tasks on the ``dakp_extract`` pool so a deployment
  can bound concurrent network downloads and CPU-heavy parses independently (create the
  pools via the Airflow CLI/UI; the names are referenced here, not provisioned at import
  time). ``Profile.download.concurrency`` documents the intended download-pool slot count.
* **Acquisition is delegated** to :mod:`dakp_pipeline.acquire`, the shared download-to-store
  layer (fetchers + NER model cache + ontology/fullmap). Acquisition tasks return
  :class:`ArtifactRef` manifests (paths + BLAKE3 ids), never dataframes, and are idempotent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dakp_pipeline import acquire, pipeline, tablassert
from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.config import load_profile
from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import configure_logging
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, drugsfda, faers
from dakp_pipeline.tablassert import configs as tablassert_configs
from dakp_pipeline.translator import contract as translator_contract

try:
    from airflow.decorators import dag, task  # type: ignore[import-not-found]
    from pendulum import datetime  # type: ignore[import-not-found]  # pragma: no cover - airflow-only

    _AIRFLOW_AVAILABLE = True  # pragma: no cover - airflow-only
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


# --- DAG-level constants (importable/testable without airflow) -------------------

DAG_ID = "dakp_build"

#: Pool bounding concurrent source downloads (network I/O).
DOWNLOAD_POOL = "dakp_download"
#: Pool bounding concurrent raw->interim parses (CPU I/O).
EXTRACT_POOL = "dakp_extract"

#: DAG-level params (PLAN.md "Phase 2": dev/full build knobs). ``profile`` selects
#: mock | sample | prod; the rest override profile defaults or steer I/O.
DAG_PARAMS: dict[str, Any] = {
    "profile": "mock",
    "quarter_limit": 1,
    "force": False,
    "fixture_root": "tests/fixtures/pipeline",
    "workdir": "data",
    "log_level": "INFO",
}

#: Flat ``stage name -> leaf callable`` manifest mirroring ``run_pipeline``'s stages.
#: Task bodies call through this dict (single source of truth); the unit tests assert it
#: is complete and identical to the callables the canonical runner uses.
STAGE_CALLABLES: dict[str, Callable[..., Any]] = {
    # acquisition (sources/)
    "acquire_dailymed": dailymed.fetch,
    "acquire_faers": faers.fetch,
    "acquire_drugsfda": drugsfda.fetch,
    # extraction (extract/)
    "extract_dailymed": spl_xml.extract,
    "extract_faers": faers_ascii.extract,
    "extract_drugsfda": drugsfda_products.extract,
    # assertion shaping (assertions/)
    "shape_treatment_tables": approved_treats.transform,
    "shape_faers_use_tables": observed_uses.transform,
    "shape_contraindication_tables": contraindications.transform,
    # tablassert handoff (tablassert/)
    "generate_tablassert_configs": tablassert_configs.generate,
    "run_tablassert": tablassert.run,
    # translator-readiness contract + build summary (translator/ + pipeline)
    "validate_contract": translator_contract.validate,
    "write_build_summary": pipeline._write_build_summary,
}


def _ctx_from_params(params: Mapping[str, Any] | None = None) -> TaskContext:
    """Build a :class:`TaskContext` from Airflow DAG params, exactly as ``run_pipeline``.

    Resolves the workdir, configures logging at ``log_level``, applies ``quarter_limit``
    and ``force`` as profile overrides, then delegates to ``pipeline._build_context`` so
    the disease map / ``mock_sources`` / ``run_tablassert`` params match the runner.
    """
    p = dict(params or {})
    wd = Workdir(Path(str(p.get("workdir", "data"))))
    wd.create()
    configure_logging(wd.root, level=str(p.get("log_level", "INFO")), for_airflow=_AIRFLOW_AVAILABLE)

    overrides: dict[str, object] = {}
    if p.get("quarter_limit") is not None:
        overrides["quarter_limit"] = int(p["quarter_limit"])
    if p.get("force") is not None:
        overrides["force"] = bool(p["force"])
    profile = load_profile(str(p.get("profile", "mock")), **overrides)
    # Forward download-layer source overrides (config.DownloadConfig) to the fetchers via params.
    extra: dict[str, object] = {}
    if profile.download.drugsfda_url:
        extra["drugsfda_url"] = profile.download.drugsfda_url
    return pipeline._build_context(profile, wd, p.get("fixture_root"), extra or None)


# The task graph below only executes inside an Airflow runtime (the optional extra, not
# installed in CI). Each @task body is a thin wrapper around a STAGE_CALLABLES entry plus
# _ctx_from_params; those wrapped callables and the context helper are tested directly, so the
# Airflow-only wiring is excluded from coverage rather than dead code.
@dag(dag_id=DAG_ID, start_date=datetime(2026, 1, 1), schedule=None, catchup=False, tags=["dakp", "drug-approvals"], params=DAG_PARAMS)
def dakp_build() -> None:  # pragma: no cover - Airflow task graph; task bodies run only under an Airflow runtime
    """Full DAKP build DAG: acquire -> extract -> shape -> Tablassert handoff -> summary."""

    # -- acquisition (download pool) -------------------------------------------
    # Each task delegates to the shared dakp_pipeline.acquire layer, which invokes the real
    # source fetchers / NER model cache / ontology downloader and returns ArtifactRef manifests.
    @task(pool=DOWNLOAD_POOL)
    def acquire_dailymed(**context: Any) -> list[ArtifactRef]:
        return acquire.acquire_dailymed(_ctx_from_params(context.get("params")))

    @task(pool=DOWNLOAD_POOL)
    def acquire_faers(**context: Any) -> list[ArtifactRef]:
        return acquire.acquire_faers(_ctx_from_params(context.get("params")))

    @task(pool=DOWNLOAD_POOL)
    def acquire_drugsfda(**context: Any) -> list[ArtifactRef]:
        return acquire.acquire_drugsfda(_ctx_from_params(context.get("params")))

    @task(pool=DOWNLOAD_POOL)
    def acquire_ner_models(**context: Any) -> list[ArtifactRef]:
        return acquire.acquire_ner_models(_ctx_from_params(context.get("params")))

    @task(pool=DOWNLOAD_POOL)
    def acquire_ontologies(**context: Any) -> list[ArtifactRef]:
        return acquire.acquire_ontologies(_ctx_from_params(context.get("params")))

    # -- extraction (extract pool) ---------------------------------------------
    @task(pool=EXTRACT_POOL)
    def extract_dailymed(raw: list[ArtifactRef], **context: Any) -> list[ArtifactRef]:
        return STAGE_CALLABLES["extract_dailymed"](raw, _ctx_from_params(context.get("params")))

    @task(pool=EXTRACT_POOL)
    def extract_faers(raw: list[ArtifactRef], **context: Any) -> list[ArtifactRef]:
        return STAGE_CALLABLES["extract_faers"](raw, _ctx_from_params(context.get("params")))

    @task(pool=EXTRACT_POOL)
    def extract_drugsfda(raw: list[ArtifactRef], **context: Any) -> list[ArtifactRef]:
        return STAGE_CALLABLES["extract_drugsfda"](raw, _ctx_from_params(context.get("params")))

    # -- assertion shaping -----------------------------------------------------
    @task
    def shape_treatment_tables(dm_ext: list[ArtifactRef], drugsfda_ext: list[ArtifactRef], **context: Any) -> list[ArtifactRef]:
        return STAGE_CALLABLES["shape_treatment_tables"]([*dm_ext, *drugsfda_ext], _ctx_from_params(context.get("params")))

    @task
    def shape_faers_use_tables(faers_ext: list[ArtifactRef], dm_ext: list[ArtifactRef], **context: Any) -> list[ArtifactRef]:
        return STAGE_CALLABLES["shape_faers_use_tables"]([*faers_ext, *dm_ext], _ctx_from_params(context.get("params")))

    @task
    def shape_contraindication_tables(dm_ext: list[ArtifactRef], ner_models: list[ArtifactRef], **context: Any) -> list[ArtifactRef]:
        # Contraindications are text-mined from the DailyMed SPL contraindication sections
        # (NER backend resolved in pipeline._build_context); no separate source extract.
        # ``ner_models`` is an ordering dependency only: the NER backend lazily loads weights
        # cached by acquire_ner_models, so mining must run after acquisition — the model refs
        # are not transform inputs.
        del ner_models
        return STAGE_CALLABLES["shape_contraindication_tables"]([*dm_ext], _ctx_from_params(context.get("params")))

    # -- tablassert handoff ----------------------------------------------------
    @task
    def generate_tablassert_configs(
        approved: list[ArtifactRef], uses: list[ArtifactRef], contra: list[ArtifactRef], **context: Any
    ) -> list[ArtifactRef]:
        return STAGE_CALLABLES["generate_tablassert_configs"]([*approved, *uses, *contra], _ctx_from_params(context.get("params")))

    @task
    def run_tablassert(
        approved: list[ArtifactRef],
        uses: list[ArtifactRef],
        contra: list[ArtifactRef],
        configs: list[ArtifactRef],
        ontologies: list[ArtifactRef],
        **context: Any,
    ) -> list[ArtifactRef]:
        # ``ontologies`` (fullmap redb / term lists) is an ordering dependency: Tablassert
        # resolves canonical entities against the fullmap acquired by acquire_ontologies, so
        # the handoff must run after acquisition. Tablassert reads the fullmap itself; the refs
        # are not passed to the handoff callable.
        del ontologies
        return STAGE_CALLABLES["run_tablassert"]([*approved, *uses, *contra], configs, _ctx_from_params(context.get("params")))

    # -- translator contract + build summary -----------------------------------
    @task
    def write_build_summary(
        approved: list[ArtifactRef], uses: list[ArtifactRef], contra: list[ArtifactRef], kgx: list[ArtifactRef], **context: Any
    ) -> str:
        ctx = _ctx_from_params(context.get("params"))
        assertion_refs = [*approved, *uses, *contra]
        report = STAGE_CALLABLES["validate_contract"](assertion_refs)
        summary = STAGE_CALLABLES["write_build_summary"](Workdir(ctx.workdir), ctx.profile, assertion_refs, kgx, report)
        return str(summary)

    dm_raw = acquire_dailymed()
    faers_raw = acquire_faers()
    drugsfda_raw = acquire_drugsfda()
    ner_models = acquire_ner_models()
    ontologies = acquire_ontologies()

    dm_ext = extract_dailymed(dm_raw)
    faers_ext = extract_faers(faers_raw)
    drugsfda_ext = extract_drugsfda(drugsfda_raw)

    approved = shape_treatment_tables(dm_ext, drugsfda_ext)
    uses = shape_faers_use_tables(faers_ext, dm_ext)
    contra = shape_contraindication_tables(dm_ext, ner_models)

    configs = generate_tablassert_configs(approved, uses, contra)
    kgx = run_tablassert(approved, uses, contra, configs, ontologies)
    write_build_summary(approved, uses, contra, kgx)


# Register the DAG when Airflow is present (Airflow scans dags/ for module-level DAGs).
dag_obj: Any = None
if _AIRFLOW_AVAILABLE:  # pragma: no cover - DAG registration needs the airflow extra
    dag_obj = dakp_build()

__all__ = ["DAG_ID", "DAG_PARAMS", "DOWNLOAD_POOL", "EXTRACT_POOL", "STAGE_CALLABLES", "dag_obj", "dakp_build"]
