"""Shared stage-harness for the DAKP integration tests.

Reproduces the Airflow DAG's (``dakp_pipeline.dags.dakp_build``) Python stage wiring so the
integration tests call the stage functions DIRECTLY through ONE harness instead of through a
duplicated runner. Production orchestration is the Airflow DAG; this module is test-only, lives
outside the coverage ``source`` (``src/``), and adds no run-config concepts of its own beyond the
explicit ``params`` it forwards to :func:`dakp_pipeline.runtime.build_context`.

WHY a harness and not per-test wiring: the four end-to-end integration tests (semantic-equivalence,
offline-pipeline, prod-smoke, KGX) all run the identical acquire -> extract -> shape -> Tablassert ->
contract/regression -> summary sequence. Centralizing it in one place means a stage signature
change touches one call site, the byte-determinism re-run uses the exact same path as the first
run, and monkeypatch boundaries stay identical across tests.

Fully monkeypatchable: the source fetchers are resolved through their owning MODULE at call time
(``dailymed.fetch``/``faers.fetch``/``drugsfda.fetch``) and ``tablassert.run`` through its owning
MODULE (``dakp_pipeline.tablassert``), so ``monkeypatch.setattr(dailymed, "fetch", ...)`` and
``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` take effect exactly as they did with
the retired runner. Fetchers always run their real (download) branches; :func:`install_fixture_fetchers`
is the shared offline stand-in that routes them to the tiny pipeline fixtures via ``ctx.fixture()``.

NOT collected by pytest (no ``test_`` prefix); imported by the integration tests as a sibling
module (``from harness import run_stages``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline import tablassert as _tablassert
from dakp_pipeline import translator
from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import configure_logging
from dakp_pipeline.paths import Workdir
from dakp_pipeline.runtime import build_context, write_build_summary
from dakp_pipeline.sources import dailymed, drugsfda, faers


@dataclass
class TableOutput:
    """One produced assertion table: its path on disk + row count."""

    name: str
    path: Path
    rows: int


@dataclass
class StageResult:
    """Return value of :func:`run_stages`: produced table summaries + the build summary path."""

    workdir: Workdir
    tables: dict[str, TableOutput] = field(default_factory=dict)
    build_summary: Path | None = None

    def table(self, name: str) -> TableOutput:
        if name not in self.tables:
            available = ", ".join(sorted(self.tables)) or "<none>"
            msg = f"No table named {name!r} in result; available: {available}"
            raise KeyError(msg)
        return self.tables[name]


def install_fixture_fetchers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the three source fetchers to the tiny pipeline fixtures via ``ctx.fixture()``.

    Offline stand-in for real acquisition: fetchers always run their real (download) branches now,
    so integration tests that want the fixture pipeline monkeypatch the module-level ``fetch``
    boundaries (exactly as the unit tests do). Loads the pipeline fixture set: one DailyMed SPL,
    the three Drugs@FDA tables, and every FAERS ``.txt`` family under ``fixture_root/faers``.
    """
    monkeypatch.setattr(dailymed, "fetch", lambda ctx: [ctx.fixture("dailymed/dailymed_spl.xml.gz")])
    monkeypatch.setattr(
        drugsfda,
        "fetch",
        lambda ctx: [
            ctx.fixture("drugsfda/drugsfda_products.tsv"),
            ctx.fixture("drugsfda/drugsfda_applications.tsv"),
            ctx.fixture("drugsfda/drugsfda_submissions.tsv"),
        ],
    )

    def _faers_fixtures(ctx: TaskContext) -> list[ArtifactRef]:
        assert ctx.fixture_root is not None
        return [ctx.fixture(f"faers/{path.name}") for path in sorted((ctx.fixture_root / "faers").glob("*.txt"))]

    monkeypatch.setattr(faers, "fetch", _faers_fixtures)


def run_stages(*, workdir: Path | str, fixture_root: Path | str | None, params: Mapping[str, Any] | None = None) -> StageResult:
    """Wire the DAKP stages exactly as the Airflow DAG does, end-to-end.

    Stages: acquire -> extract -> shape assertions -> generate Tablassert configs -> Tablassert
    handoff -> translator contract + regression -> build summary. The same sequence the DAG drives;
    the only difference is this runs Airflow-free in-process so the tests exercise the real stage
    functions (and the pure-Python reference extractors) with full monkeypatch control. ``params``
    carries the explicit run behavior (``run_tablassert``, ``quarter_limit``, ``release_limit``,
    ``fullmap``); fetchers run their real branches unless a test monkeypatches them (see
    :func:`install_fixture_fetchers`).
    """
    wd = Workdir(Path(workdir))
    wd.create()

    configure_logging(wd.root, level="INFO", for_airflow=False)
    ctx = build_context(wd, fixture_root, params)

    # 1. Acquire sources (real fetchers; offline tests monkeypatch the module-level fetch).
    dm_raw = dailymed.fetch(ctx)
    faers_raw = faers.fetch(ctx)
    drugsfda_raw = drugsfda.fetch(ctx)

    # 2. Extract raw -> interim parquet tables (pure-Python reference extractors).
    dm_ext = spl_xml.extract(dm_raw, ctx)
    faers_ext = faers_ascii.extract(faers_raw, ctx)
    drugsfda_ext = drugsfda_products.extract(drugsfda_raw, ctx)

    # 3. Shape assertion tables (uncompressed TSV, Tablassert-facing).
    approved = approved_treats.transform([*dm_ext, *drugsfda_ext, *faers_ext], ctx)
    uses = observed_uses.transform([*faers_ext, *dm_ext], ctx)
    contra = contraindications.transform([*dm_ext], ctx)
    assertion_refs = [*approved, *uses, *contra]

    # 4. Generate Tablassert Graph + per-table configs.
    config_refs = _tablassert.generate(assertion_refs, ctx)

    # 5. Tablassert handoff (deferred unless run_tablassert is set; no local KGX compiler).
    kgx_refs = _tablassert.run(assertion_refs, config_refs, ctx)

    # 6. Translator-readiness contract + regression + build summary.
    report = translator.validate(assertion_refs)
    regression_report = translator.check_assertion_tables(assertion_refs)
    build_summary = write_build_summary(wd, assertion_refs, kgx_refs, report, regression_report)

    tables = {ref.uri.stem: TableOutput(ref.uri.stem, ref.uri, ref.rows or 0) for ref in assertion_refs}
    return StageResult(workdir=wd, tables=tables, build_summary=build_summary)


__all__ = ["StageResult", "TableOutput", "install_fixture_fetchers", "run_stages"]
