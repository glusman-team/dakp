"""Pure-Python pipeline runner — an internal test/dev harness (NOT a user-facing entrypoint).

The production orchestrator is the Airflow DAG (:mod:`dakp_pipeline.dags.dakp_build`); the CLI that
used to wrap this runner is retired (see plans/airflow-native-go-workers.md). ``run_pipeline`` is
kept as a fast, Airflow-free way to exercise the stage functions and the **pure-Python reference
extractors** end-to-end (the native Go extractors are validated by ``uv run dakp up`` + the Go parity
tests). It wires every stage exactly as the DAG does and shares the same context/summary helpers
(:mod:`dakp_pipeline.runtime`), so its outputs match the DAG's.

Fully monkeypatchable: fetchers and ``dakp_pipeline.tablassert.run`` are resolved through their
owning module/package at call time, so ``monkeypatch.setattr(dailymed, "fetch", ...)`` and
``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` take effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dakp_pipeline import tablassert as _tablassert
from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.config import Profile, load_profile
from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
from dakp_pipeline.logging_setup import bind, configure_logging
from dakp_pipeline.paths import Workdir
from dakp_pipeline.runtime import build_context, write_build_summary
from dakp_pipeline.sources import dailymed, drugsfda, faers
from dakp_pipeline.tablassert import configs as _tablassert_configs
from dakp_pipeline.translator import contract as translator_contract
from dakp_pipeline.translator import regression


@dataclass
class TableResult:
    """Summary of one produced tabular output."""

    name: str
    path: Path
    rows: int

    def exists(self) -> bool:
        return self.path.exists()


@dataclass
class PipelineResult:
    """Return value of :func:`run_pipeline`. Exposes table summaries for assertions."""

    workdir: Workdir
    profile: Profile
    tables: dict[str, TableResult] = field(default_factory=dict)
    build_summary: Path | None = None

    def table(self, name: str) -> TableResult:
        if name not in self.tables:
            available = ", ".join(sorted(self.tables)) or "<none>"
            msg = f"No table named {name!r} in result; available: {available}"
            raise KeyError(msg)
        return self.tables[name]


def run_pipeline(
    profile: str = "mock", fixture_root: Path | str | None = None, workdir: Path | str = "data", params: Mapping[str, Any] | None = None
) -> PipelineResult:
    """Run the DAKP pipeline end-to-end with the pure-Python extractors (mock needs no network).

    Stages: acquire -> extract -> shape assertions -> generate Tablassert configs -> Tablassert
    handoff -> translator contract + regression -> build summary. This is an Airflow-free test/dev
    harness; real Airflow execution is the DAG's job (``uv run dakp up``).
    """
    resolved_profile = load_profile(profile)
    wd = Workdir(Path(workdir))
    wd.create()

    configure_logging(wd.root, level="INFO", for_airflow=False)
    ctx = build_context(resolved_profile, wd, fixture_root, params)

    log = bind(task_id="run_pipeline", profile=profile, workdir=str(wd.root))
    log.info("pipeline start")

    # 1. Acquire sources (mock profile ingests fixtures into the content-addressed store).
    dm_raw = dailymed.fetch(ctx)
    faers_raw = faers.fetch(ctx)
    drugsfda_raw = drugsfda.fetch(ctx)
    log.info("sources acquired", dailymed=len(dm_raw), faers=len(faers_raw), drugsfda=len(drugsfda_raw))

    # 2. Extract raw -> interim parquet tables (pure-Python reference extractors).
    dm_ext = spl_xml.extract(dm_raw, ctx)
    faers_ext = faers_ascii.extract(faers_raw, ctx)
    drugsfda_ext = drugsfda_products.extract(drugsfda_raw, ctx)

    # 3. Shape assertion tables (uncompressed TSV, Tablassert-facing).
    approved = approved_treats.transform([*dm_ext, *drugsfda_ext], ctx)
    uses = observed_uses.transform([*faers_ext, *dm_ext], ctx)
    contra = contraindications.transform([*dm_ext], ctx)
    assertion_refs = [*approved, *uses, *contra]

    # 4. Generate Tablassert Graph + per-table configs.
    config_refs = _tablassert_configs.generate(assertion_refs, ctx)

    # 5. Tablassert handoff (mock writes a deferred-handoff manifest; no KGX compiler).
    kgx_refs = _tablassert.run(assertion_refs, config_refs, ctx)

    # 6. Translator-readiness contract + regression + build summary.
    report = translator_contract.validate(assertion_refs)
    regression_report = regression.check_assertion_tables(assertion_refs)
    build_summary = write_build_summary(wd, profile, assertion_refs, kgx_refs, report, regression_report)

    tables = {ref.uri.stem: TableResult(ref.uri.stem, ref.uri, ref.rows or 0) for ref in assertion_refs}
    log.info("pipeline complete", tables=len(tables), build_summary=str(build_summary))
    return PipelineResult(workdir=wd, profile=resolved_profile, tables=tables, build_summary=build_summary)


__all__ = ["PipelineResult", "TableResult", "run_pipeline"]
