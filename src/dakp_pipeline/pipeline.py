"""Pure-Python pipeline runner — the single source of truth for orchestration.

``run_pipeline`` wires every stage end-to-end and is fully monkeypatchable: fetchers and
``dakp_pipeline.tablassert.run`` are always resolved through their owning module/package at
call time, so ``monkeypatch.setattr(dailymed, "fetch", ...)`` and
``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` take effect (PLAN.md sketch).

The Airflow DAG (:mod:`dakp_pipeline.dags.dakp_build`) is a thin TaskFlow wrapper around
these same stage functions; this runner is what the CLI and tests exercise.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline import tablassert as _tablassert
from dakp_pipeline.assertions import approved_treats, contraindications, observed_uses
from dakp_pipeline.config import Profile, load_profile
from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import bind, configure_logging
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, drugsfda, faers
from dakp_pipeline.tablassert import configs as _tablassert_configs
from dakp_pipeline.translator import contract as translator_contract


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
    profile: str = "mock",
    fixture_root: Path | str | None = None,
    workdir: Path | str = "data",
    run_airflow: bool = False,
    params: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Run the DAKP pipeline end-to-end (mock profile needs no network/Tablassert).

    Stages: acquire -> extract -> shape assertions -> generate Tablassert configs ->
    Tablassert handoff -> write build summary. The ``run_airflow`` flag only toggles
    Airflow-targeted logging; actual Airflow task decomposition lives in the DAG module.
    """
    resolved_profile = load_profile(profile)
    wd = Workdir(Path(workdir))
    wd.create()

    if run_airflow:
        try:
            import airflow  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - requires airflow extra
            msg = "run_airflow=True requires the airflow extra; install with `uv sync --extra airflow`"
            raise RuntimeError(msg) from exc

    configure_logging(wd.root, level="INFO", for_airflow=run_airflow)
    ctx = _build_context(resolved_profile, wd, fixture_root, params)

    log = bind(task_id="run_pipeline", profile=profile, workdir=str(wd.root))
    log.info("pipeline start")

    # 1. Acquire sources (mock profile ingests fixtures into the content-addressed store).
    dm_raw = dailymed.fetch(ctx)
    faers_raw = faers.fetch(ctx)
    drugsfda_raw = drugsfda.fetch(ctx)
    log.info("sources acquired", dailymed=len(dm_raw), faers=len(faers_raw), drugsfda=len(drugsfda_raw))

    # 2. Extract raw -> interim parquet tables.
    dm_ext = spl_xml.extract(dm_raw, ctx)
    faers_ext = faers_ascii.extract(faers_raw, ctx)
    drugsfda_ext = drugsfda_products.extract(drugsfda_raw, ctx)

    # 3. Shape assertion tables (uncompressed TSV, Tablassert-facing). Contraindications are
    #    text-mined from the DailyMed SPL contraindication sections via the configured NER
    #    backend (resolved in _build_context and passed through ctx.params).
    approved = approved_treats.transform([*dm_ext, *drugsfda_ext], ctx)
    uses = observed_uses.transform([*faers_ext, *dm_ext], ctx)
    contra = contraindications.transform([*dm_ext], ctx)
    assertion_refs = [*approved, *uses, *contra]

    # 4. Generate Tablassert Graph + per-table configs.
    config_refs = _tablassert_configs.generate(assertion_refs, ctx)

    # 5. Tablassert handoff (mock writes a deferred-handoff manifest; no KGX compiler).
    kgx_refs = _tablassert.run(assertion_refs, config_refs, ctx)

    # 6. Translator-readiness contract + build summary.
    report = translator_contract.validate(assertion_refs)
    build_summary = _write_build_summary(wd, profile, assertion_refs, kgx_refs, report)

    tables = {ref.uri.stem: TableResult(ref.uri.stem, ref.uri, ref.rows or 0) for ref in assertion_refs}
    log.info("pipeline complete", tables=len(tables), build_summary=str(build_summary))
    return PipelineResult(workdir=wd, profile=resolved_profile, tables=tables, build_summary=build_summary)


# --- helpers --------------------------------------------------------------------


def _build_context(profile: Profile, wd: Workdir, fixture_root: Path | str | None, extra: Mapping[str, Any] | None) -> TaskContext:
    fixture = Path(fixture_root) if fixture_root is not None else None
    disease_map = _load_disease_map(fixture) if fixture is not None else {}
    params: dict[str, Any] = {
        "disease_map": disease_map,
        "mock_sources": profile.mock_sources,
        "run_tablassert": profile.run_tablassert,
        "quarter_limit": profile.quarter_limit,
        "release_limit": profile.release_limit,
        "force": profile.force,
        "use_go_workers": profile.use_go_workers,
    }
    if extra:
        params.update(extra)
    # Configure the NER backend for DailyMed contraindication mining: the offline dictionary
    # baseline by default, or a real GLiNER/SciSpacy backend when ``ner_backend_name`` selects
    # one (needs the [ner] extra). Built once here and passed to the shaper via ctx.params.
    params["ner_backend"] = contraindications.resolve_ner_backend(fixture, params)
    return TaskContext(
        profile=profile.name, workdir=wd.root, fixture_root=fixture, threads=profile.threads, memory_budget_gb=profile.memory_budget_gb, params=params
    )


def _load_disease_map(fixture_root: Path) -> dict[str, dict[str, str]]:
    """Load the lexical disease dictionary (fast baseline) from the ontology fixture."""
    path = fixture_root / "ontology" / "disease_map.tsv"
    if not path.exists():
        return {}
    frame = pl.read_csv(path, separator="\t")
    mapping: dict[str, dict[str, str]] = {}
    for rec in frame.iter_rows(named=True):
        text = str(rec.get("text", "") or "").strip()
        if not text:
            continue
        mapping[text] = {
            "curie": str(rec.get("curie", "") or ""),
            "name": str(rec.get("name", text) or text),
            "category": str(rec.get("category", "Disease") or "Disease"),
        }
    return mapping


def _write_build_summary(
    wd: Workdir, profile: str, assertion_refs: list[ArtifactRef], kgx_refs: list[ArtifactRef], report: translator_contract.ContractReport
) -> Path:
    summary_path = wd.reports / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "dakp.build_summary.v1",
        "profile": profile,
        "generated_at": datetime.now(UTC).isoformat(),
        "workdir": str(wd.root),
        "tables": [{"name": ref.uri.stem, "path": str(ref.uri), "rows": ref.rows, "artifact_id": ref.blake3} for ref in assertion_refs],
        "tablassert": {"handoff_refs": [str(ref.uri) for ref in kgx_refs]},
        "translator_contract": {"ok": report.ok, "problems": report.problems, "tables": report.tables},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


__all__ = ["PipelineResult", "TableResult", "run_pipeline"]
