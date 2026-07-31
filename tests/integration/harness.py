"""Shared stage-harness for the DAKP integration tests.

Reproduces the Airflow DAG's (``dakp_pipeline.dags.dakp_build``) Python stage wiring — the same
call sequence the retired pure-Python runner (``pipeline.py``) used to exercise — so the
integration tests call the stage functions DIRECTLY through ONE harness instead of through a
duplicated runner. Production orchestration is the Airflow DAG; this module is test-only, lives
outside the coverage ``source`` (``src/``), and adds no profile/machine concepts of its own beyond
forwarding ``profile`` to the still-existing :func:`dakp_pipeline.config.load_profile`.

WHY a harness and not per-test wiring: the four end-to-end integration tests (semantic-equivalence,
mock-pipeline, prod-smoke, KGX) all run the identical acquire -> extract -> shape -> Tablassert ->
contract/regression -> summary sequence. Centralizing it in one place means a stage signature
change touches one call site, the byte-determinism re-run uses the exact same path as the first
run, and monkeypatch boundaries stay identical across tests.

Fully monkeypatchable: the source fetchers are resolved through their owning MODULE at call time
(``dailymed.fetch``/``faers.fetch``/``drugsfda.fetch``) and ``tablassert.run`` through its owning
PACKAGE (``dakp_pipeline.tablassert.run``), so ``monkeypatch.setattr(dailymed, "fetch", ...)`` and
``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` take effect exactly as they did with
the retired runner.

NOT collected by pytest (no ``test_`` prefix); imported by the integration tests as a sibling
module (``from harness import run_stages``).
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
from dakp_pipeline.logging_setup import configure_logging
from dakp_pipeline.paths import Workdir
from dakp_pipeline.runtime import build_context, write_build_summary
from dakp_pipeline.sources import dailymed, drugsfda, faers
from dakp_pipeline.tablassert import configs as _tablassert_configs
from dakp_pipeline.translator import contract as translator_contract
from dakp_pipeline.translator import regression


@dataclass
class TableOutput:
    """One produced assertion table: its path on disk + row count."""

    name: str
    path: Path
    rows: int


@dataclass
class StageResult:
    """Return value of :func:`run_stages`: produced table summaries + the resolved profile + build summary.

    ``.profile`` exposes the resolved :class:`~dakp_pipeline.config.Profile` so a smoke test can still
    assert the driven profile (``result.profile.name == "prod"``) while profiles remain (a later
    story removes them).
    """

    workdir: Workdir
    profile: Profile
    tables: dict[str, TableOutput] = field(default_factory=dict)
    build_summary: Path | None = None

    def table(self, name: str) -> TableOutput:
        if name not in self.tables:
            available = ", ".join(sorted(self.tables)) or "<none>"
            msg = f"No table named {name!r} in result; available: {available}"
            raise KeyError(msg)
        return self.tables[name]


def run_stages(
    *, workdir: Path | str, fixture_root: Path | str | None, profile: str = "mock", params: Mapping[str, Any] | None = None
) -> StageResult:
    """Wire the DAKP stages exactly as the Airflow DAG does, end-to-end (no network for ``mock``).

    Stages: acquire -> extract -> shape assertions -> generate Tablassert configs -> Tablassert
    handoff -> translator contract + regression -> build summary. The same sequence the DAG drives;
    the only difference is this runs Airflow-free in-process so the tests exercise the real stage
    functions (and the pure-Python reference extractors) with full monkeypatch control.
    """
    resolved_profile = load_profile(profile)
    wd = Workdir(Path(workdir))
    wd.create()

    configure_logging(wd.root, level="INFO", for_airflow=False)
    ctx = build_context(resolved_profile, wd, fixture_root, params)

    # 1. Acquire sources (mock profile ingests fixtures into the content-addressed store).
    dm_raw = dailymed.fetch(ctx)
    faers_raw = faers.fetch(ctx)
    drugsfda_raw = drugsfda.fetch(ctx)

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

    tables = {ref.uri.stem: TableOutput(ref.uri.stem, ref.uri, ref.rows or 0) for ref in assertion_refs}
    return StageResult(workdir=wd, profile=resolved_profile, tables=tables, build_summary=build_summary)


__all__ = ["StageResult", "TableOutput", "run_stages"]
