"""``dakp_build`` Airflow DAG — the full pipeline task graph (Airflow-native).

This is the **only** orchestrator (the former pure-Python pipeline runner is retired). The heavy
parsing/extraction runs as **native Airflow Go SDK
bundle workers** (``go/cmd/dakp-bundle``): the three ``extract_*`` tasks are ``@task.stub(queue=
"golang")`` declarations whose Go implementations the ExecutableCoordinator forks per task instance.
Every other stage (acquisition, assertion shaping, Tablassert handoff, translator contract +
regression, build summary) is a real Python TaskFlow task reusing the existing stage modules.

Tasks pass ``list[ArtifactRef]`` manifests over XCom (serialized to JSON dicts via
:mod:`dakp_pipeline.io.xcom` so the native Go workers read/write the same manifests); heavy bytes
move through the BLAKE3 content-addressed filesystem store. Run config (workdir / fixtures /
limits) comes from the ``dakp_config`` Airflow Variable, set by the one-command orchestrator
(``dakp up``) and shared by the Python tasks and the Go bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from airflow.sdk import TaskGroup, Variable, dag, task
from pendulum import datetime

from dakp_pipeline.logging_setup import logger, stats, step

# --- DAG-level constants ---------------------------------------------------------

DAG_ID = "dakp_build"

#: Queue routed to the Go coordinator (airflow.cfg: [sdk] queue_to_coordinator = {"golang": "go"}).
GO_QUEUE = "golang"
#: Pool bounding concurrent source downloads (network I/O).
DOWNLOAD_POOL = "dakp_download"
#: Pool bounding concurrent raw->interim parses (CPU I/O) — the Go extract stubs run here.
EXTRACT_POOL = "dakp_extract"

#: Pool-slot weights for the 4-slot ``dakp_extract`` pool on wenceslaus. FAERS and DailyMed are
#: memory-heavy enough that they should not overlap under the 50 GB pipeline RAM budget, while the
#: tiny Drugs@FDA parse can run beside either heavy extraction.
DAILYMED_EXTRACT_POOL_SLOTS = 3
FAERS_EXTRACT_POOL_SLOTS = 3
DRUGSFDA_EXTRACT_POOL_SLOTS = 1

#: Airflow Variable (JSON) holding the per-run config (workdir / fixture_root / threads / limits).
CONFIG_VARIABLE = "dakp_config"

_DAG_DOC_MD = """
### DAKP build stages

The DAG is organized into five visual TaskGroups while preserving the historical task IDs:

1. **acquire** — network/model acquisition, bounded by `dakp_download`.
2. **extract** — native Go SDK stubs on the `golang` queue, bounded by `dakp_extract`.
3. **shape** — Python assertion-table shaping over artifact manifests.
4. **tablassert** — config generation and optional real Tablassert KGX handoff.
5. **summary** — terminal translator validation/regression/build-summary task.

The `dakp_extract` pool has 4 slots. DailyMed and FAERS extraction each consume 3 slots, and
Drugs@FDA consumes 1 slot. This allows the small Drugs@FDA extract to overlap with either heavy
extract while preventing DailyMed and FAERS from extracting concurrently under the 50 GB memory
budget on wenceslaus.
"""

_ACQUIRE_DOC_MD = """Acquire raw source/model artifacts and return small `ArtifactRef` manifests via XCom."""
_EXTRACT_DOC_MD = """Native Go SDK bundle extraction; heavy payloads stay in the content-addressed store."""
_SHAPE_DOC_MD = """Shape interim artifacts into DAKP assertion TSVs without moving table bytes through XCom."""
_TABLASSERT_DOC_MD = """Generate Tablassert configs and optionally run the installed Tablassert CLI."""
_SUMMARY_DOC_MD = """Terminal build-summary stage: translator validation, regression checks, and report JSON."""


@dataclass(frozen=True)
class AcquireOutputs:
    """Task handles produced by the acquisition stage."""

    dailymed: Any
    faers: Any
    drugsfda: Any
    ner_models: Any


@dataclass(frozen=True)
class ExtractOutputs:
    """Task handles produced by the native extraction stage."""

    dailymed: Any
    faers: Any
    drugsfda: Any


@dataclass(frozen=True)
class AssertionOutputs:
    """Task handles produced by the assertion-shaping stage."""

    approved: Any
    uses: Any
    contraindications: Any


@dataclass(frozen=True)
class TablassertOutputs:
    """Task handles produced by the Tablassert handoff stage."""

    configs: Any
    kgx: Any


def _cfg() -> dict[str, Any]:  # pragma: no cover - reads the Airflow Variable at task runtime
    """Read the per-run config dict from the ``dakp_config`` Variable."""
    return Variable.get(CONFIG_VARIABLE, deserialize_json=True)  # type: ignore[no-any-return]


def _ctx() -> Any:  # pragma: no cover - builds the context at task runtime
    """Build the per-task ``TaskContext`` from the ``dakp_config`` Variable."""
    from dakp_pipeline import runtime

    return runtime.build_context_from_config(_cfg())


def _refs_to_xcom(refs: Any) -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
    """Serialize ArtifactRefs for XCom, importing the serializer only inside task execution."""
    from dakp_pipeline.io.xcom import refs_to_xcom

    return refs_to_xcom(refs)


def _refs_from_xcom(items: Any) -> list[Any]:  # pragma: no cover - body executes only under the Airflow task runtime
    """Deserialize ArtifactRefs from XCom, importing the serializer only inside task execution."""
    from dakp_pipeline.io.xcom import refs_from_xcom

    return refs_from_xcom(items)


def _build_acquire_stage() -> AcquireOutputs:
    """Create the acquisition TaskGroup and return its task handles."""
    with TaskGroup(group_id="acquire", prefix_group_id=False, tooltip="Acquire raw inputs", doc_md=_ACQUIRE_DOC_MD):

        @task(pool=DOWNLOAD_POOL, doc_md="Download/cache DailyMed SPL artifacts; returns `ArtifactRef` manifests only.")
        def acquire_dailymed() -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline import acquire

            ctx = _ctx()
            with step(logger, "task acquire_dailymed"):
                refs = acquire.acquire_dailymed(ctx)
                stats(logger, "task acquire_dailymed", output_refs=len(refs))
                return _refs_to_xcom(refs)

        @task(pool=DOWNLOAD_POOL, doc_md="Download/cache FAERS quarterly ASCII artifacts; returns `ArtifactRef` manifests only.")
        def acquire_faers() -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline import acquire

            ctx = _ctx()
            with step(logger, "task acquire_faers"):
                refs = acquire.acquire_faers(ctx)
                stats(logger, "task acquire_faers", output_refs=len(refs))
                return _refs_to_xcom(refs)

        @task(pool=DOWNLOAD_POOL, doc_md="Download/cache the Drugs@FDA data-files ZIP; returns `ArtifactRef` manifests only.")
        def acquire_drugsfda() -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline import acquire

            ctx = _ctx()
            with step(logger, "task acquire_drugsfda"):
                refs = acquire.acquire_drugsfda(ctx)
                stats(logger, "task acquire_drugsfda", output_refs=len(refs))
                return _refs_to_xcom(refs)

        @task(pool=DOWNLOAD_POOL, doc_md="Ensure the production GLiNER checkpoint is cached before contraindication mining.")
        def acquire_ner_models() -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline import acquire

            ctx = _ctx()
            with step(logger, "task acquire_ner_models"):
                refs = acquire.acquire_ner_models(ctx)
                stats(logger, "task acquire_ner_models", output_refs=len(refs))
                return _refs_to_xcom(refs)

        return AcquireOutputs(dailymed=acquire_dailymed(), faers=acquire_faers(), drugsfda=acquire_drugsfda(), ner_models=acquire_ner_models())


def _build_extract_stage(raw: AcquireOutputs) -> ExtractOutputs:
    """Create the native Go extraction TaskGroup and return its task handles."""
    with TaskGroup(group_id="extract", prefix_group_id=False, tooltip="Native Go extraction", doc_md=_EXTRACT_DOC_MD):
        # No Python body: the ExecutableCoordinator forks the Go bundle, which reads the upstream
        # acquire_* ArtifactRefs from XCom, parses with internal/{dailymed,faers,drugsfda}, writes
        # the interim parquet + TSV handoff into the store, and pushes output ArtifactRefs as XCom.
        @task.stub(queue=GO_QUEUE, pool=EXTRACT_POOL, pool_slots=DAILYMED_EXTRACT_POOL_SLOTS, doc_md=_EXTRACT_DOC_MD)
        def extract_dailymed(raw_refs: Any) -> list[dict[str, Any]]: ...

        @task.stub(queue=GO_QUEUE, pool=EXTRACT_POOL, pool_slots=FAERS_EXTRACT_POOL_SLOTS, doc_md=_EXTRACT_DOC_MD)
        def extract_faers(raw_refs: Any) -> list[dict[str, Any]]: ...

        @task.stub(queue=GO_QUEUE, pool=EXTRACT_POOL, pool_slots=DRUGSFDA_EXTRACT_POOL_SLOTS, doc_md=_EXTRACT_DOC_MD)
        def extract_drugsfda(raw_refs: Any) -> list[dict[str, Any]]: ...

        return ExtractOutputs(dailymed=extract_dailymed(raw.dailymed), faers=extract_faers(raw.faers), drugsfda=extract_drugsfda(raw.drugsfda))


def _build_shape_stage(extracts: ExtractOutputs, ner_models: Any) -> AssertionOutputs:
    """Create the assertion-shaping TaskGroup and return assertion task handles."""
    with TaskGroup(group_id="shape", prefix_group_id=False, tooltip="Shape assertion tables", doc_md=_SHAPE_DOC_MD):

        @task(doc_md="Shape FDA-approved treatment assertions from DailyMed, Drugs@FDA, and FAERS refs.")
        def shape_treatment_tables(
            dm_ext: Any, drugsfda_ext: Any, faers_ext: Any
        ) -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline.assertions import approved_treats

            ctx = _ctx()
            with step(logger, "task shape_treatment_tables"):
                dailymed_refs, drugsfda_refs, faers_refs = _refs_from_xcom(dm_ext), _refs_from_xcom(drugsfda_ext), _refs_from_xcom(faers_ext)
                stats(
                    logger,
                    "task shape_treatment_tables",
                    dailymed_refs=len(dailymed_refs),
                    drugsfda_refs=len(drugsfda_refs),
                    faers_refs=len(faers_refs),
                )
                out = approved_treats.transform([*dailymed_refs, *drugsfda_refs, *faers_refs], ctx)
                stats(logger, "task shape_treatment_tables", output_refs=len(out))
                return _refs_to_xcom(out)

        @task(
            doc_md="Shape FAERS observed-use assertions from FAERS cases + DailyMed refs, cross-referenced with the approved-treats table for the approval status."
        )
        def shape_faers_use_tables(
            faers_ext: Any, dm_ext: Any, approved: Any
        ) -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline.assertions import observed_uses

            ctx = _ctx()
            with step(logger, "task shape_faers_use_tables"):
                faers_refs, dailymed_refs, approved_refs = _refs_from_xcom(faers_ext), _refs_from_xcom(dm_ext), _refs_from_xcom(approved)
                stats(
                    logger,
                    "task shape_faers_use_tables",
                    faers_refs=len(faers_refs),
                    dailymed_refs=len(dailymed_refs),
                    approved_refs=len(approved_refs),
                )
                out = observed_uses.transform([*faers_refs, *dailymed_refs, *approved_refs], ctx)
                stats(logger, "task shape_faers_use_tables", output_refs=len(out))
                return _refs_to_xcom(out)

        @task(doc_md="Mine contraindication assertions from DailyMed refs after production NER models are cached.")
        def shape_contraindication_tables(
            dm_ext: Any, ner_models_ref: Any
        ) -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            # ``ner_models_ref`` is an ordering dependency: the production NER lazily loads the
            # GLiNER weights cached by acquire_ner_models, so mining runs after acquisition (the
            # model refs aren't inputs). The shaper owns its internal two-pass indication-section
            # mining and 4-GPU dispatch; the DAG only needs DailyMed refs + model-cache ordering.
            del ner_models_ref
            from dakp_pipeline.assertions import contraindications
            from dakp_pipeline.io.contracts import TaskContext
            from dakp_pipeline.ner.ner import DiseaseNER

            ctx = _ctx()
            # Production composite NER (curated gazetteer anchors + GLiNER zero-shot recall) mines
            # contraindication diseases far beyond any fixed gazetteer. It loads the GLiNER
            # checkpoint acquire_ner_models cached under the workdir. Offline tests inject their
            # own deterministic backend (or fall back to the gazetteer); this production wiring is
            # DAG-only. Keep ``contraindications.transform`` as a runtime module lookup so tests and
            # concurrent indication-parser work can monkeypatch the module attribute.
            with step(logger, "task shape_contraindication_tables"):
                dailymed_refs = _refs_from_xcom(dm_ext)
                stats(logger, "task shape_contraindication_tables", dailymed_refs=len(dailymed_refs))
                ner = DiseaseNER(offline=False, workdir=ctx.workdir)
                ctx = TaskContext(workdir=ctx.workdir, fixture_root=ctx.fixture_root, params={**ctx.params, "ner": ner})
                out = contraindications.transform(dailymed_refs, ctx)
                stats(logger, "task shape_contraindication_tables", output_refs=len(out))
                return _refs_to_xcom(out)

        # The observed-uses task consumes the produced approved-treats table (approval-status
        # cross-reference), so it runs after shape_treatment_tables.
        approved = shape_treatment_tables(extracts.dailymed, extracts.drugsfda, extracts.faers)
        return AssertionOutputs(
            approved=approved,
            uses=shape_faers_use_tables(extracts.faers, extracts.dailymed, approved),
            contraindications=shape_contraindication_tables(extracts.dailymed, ner_models),
        )


def _build_tablassert_stage(assertions: AssertionOutputs) -> TablassertOutputs:
    """Create the Tablassert handoff TaskGroup and return its task handles."""
    with TaskGroup(group_id="tablassert", prefix_group_id=False, tooltip="Tablassert handoff", doc_md=_TABLASSERT_DOC_MD):

        @task(doc_md="Generate Graph/table YAML configs for the assertion TSVs.")
        def generate_tablassert_configs(
            approved: Any, uses: Any, contra: Any
        ) -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline import tablassert

            ctx = _ctx()
            with step(logger, "task generate_tablassert_configs"):
                refs = [*_refs_from_xcom(approved), *_refs_from_xcom(uses), *_refs_from_xcom(contra)]
                stats(logger, "task generate_tablassert_configs", assertion_refs=len(refs))
                out = tablassert.generate(refs, ctx)
                stats(logger, "task generate_tablassert_configs", output_refs=len(out))
                return _refs_to_xcom(out)

        @task(doc_md="Run the installed Tablassert CLI when a fullmap is configured; otherwise emit a deferred handoff report.")
        def run_tablassert(
            approved: Any, uses: Any, contra: Any, configs: Any
        ) -> list[dict[str, Any]]:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline import tablassert

            ctx = _ctx()
            with step(logger, "task run_tablassert"):
                assertion_refs = [*_refs_from_xcom(approved), *_refs_from_xcom(uses), *_refs_from_xcom(contra)]
                config_refs = _refs_from_xcom(configs)
                stats(logger, "task run_tablassert", assertion_refs=len(assertion_refs), config_refs=len(config_refs))
                out = tablassert.run(assertion_refs, config_refs, ctx)
                stats(logger, "task run_tablassert", output_refs=len(out))
                return _refs_to_xcom(out)

        configs = generate_tablassert_configs(assertions.approved, assertions.uses, assertions.contraindications)
        kgx = run_tablassert(assertions.approved, assertions.uses, assertions.contraindications, configs)
        return TablassertOutputs(configs=configs, kgx=kgx)


def _build_summary_stage(assertions: AssertionOutputs, kgx: Any) -> Any:
    """Create the terminal summary TaskGroup and return its task handle."""
    with TaskGroup(group_id="summary", prefix_group_id=False, tooltip="Build summary", doc_md=_SUMMARY_DOC_MD):

        @task(doc_md="Terminal task: validate assertion tables, run regression guards, and write build_summary.json.")
        def write_build_summary(
            approved: Any, uses: Any, contra: Any, kgx_refs: Any
        ) -> str:  # pragma: no cover - body executes only under the Airflow task runtime
            from dakp_pipeline import runtime, translator
            from dakp_pipeline.paths import Workdir

            ctx = _ctx()
            with step(logger, "task write_build_summary"):
                assertion_refs = [*_refs_from_xcom(approved), *_refs_from_xcom(uses), *_refs_from_xcom(contra)]
                kgx_ref_list = _refs_from_xcom(kgx_refs)
                stats(logger, "task write_build_summary", assertion_refs=len(assertion_refs), kgx_refs=len(kgx_ref_list))
                report = translator.validate(assertion_refs)
                regression_report = translator.check_assertion_tables(assertion_refs)
                summary = runtime.write_build_summary(Workdir(ctx.workdir), assertion_refs, kgx_ref_list, report, regression_report)
                stats(logger, "task write_build_summary", summary_path=str(summary))
                return str(summary)

        return write_build_summary(assertions.approved, assertions.uses, assertions.contraindications, kgx)


@dag(dag_id=DAG_ID, start_date=datetime(2026, 1, 1), schedule=None, catchup=False, tags=["dakp", "drug-approvals"], doc_md=_DAG_DOC_MD)
def dakp_build() -> None:  # pragma: no cover - Airflow task graph; task bodies execute only under an Airflow runtime
    """Full DAKP build DAG: acquire -> extract (native Go) -> shape -> Tablassert handoff -> summary."""
    acquired = _build_acquire_stage()
    extracted = _build_extract_stage(acquired)
    assertions = _build_shape_stage(extracted, acquired.ner_models)
    tablassert_outputs = _build_tablassert_stage(assertions)
    _build_summary_stage(assertions, tablassert_outputs.kgx)


# Register the DAG (Airflow scans the dags folder for module-level DAGs).
dag_obj: Any = dakp_build()

__all__ = [
    "CONFIG_VARIABLE",
    "DAG_ID",
    "DAILYMED_EXTRACT_POOL_SLOTS",
    "DOWNLOAD_POOL",
    "DRUGSFDA_EXTRACT_POOL_SLOTS",
    "EXTRACT_POOL",
    "FAERS_EXTRACT_POOL_SLOTS",
    "GO_QUEUE",
    "dag_obj",
    "dakp_build",
]
