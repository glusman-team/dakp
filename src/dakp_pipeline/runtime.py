"""Airflow-native runtime helpers — the canonical home for run-context + build-summary logic.

The DAG is the only orchestrator (the former pure-Python pipeline runner is retired). Tasks build
their :class:`TaskContext` from the ``dakp_config``
Airflow Variable (the single source of run config, shared by the Python tasks and the native Go
bundle workers) via :func:`build_context_from_config`, and the final summary is written by
:func:`write_build_summary`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline import translator
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import configure_logging, logger, stats
from dakp_pipeline.paths import Workdir

__all__ = ["build_context", "build_context_from_config", "write_build_summary"]


def build_context_from_config(cfg: Mapping[str, Any]) -> TaskContext:
    """Build a :class:`TaskContext` from the ``dakp_config`` Variable dict.

    Resolves the workdir, configures logging, and builds the run params directly from the config:
    ``run_tablassert`` is DERIVED from fullmap presence (a fullmap path triggers the real Tablassert
    handoff; absent => deferred, never an error), and ``quarter_limit`` / ``release_limit`` /
    ``dailymed_max_age_days`` / ``drugsfda_max_age_days`` / ``force`` / ``fullmap`` / ``drugsfda_url`` are forwarded when set.
    ``fullmap`` is resolved to an absolute path so relative paths passed at ``dakp up`` time are
    anchored to the caller's CWD, not the Airflow worker's CWD at task-run time.
    Delegates to :func:`build_context` so the disease map is loaded from the fixture root exactly
    as the test harness does.
    """
    wd = Workdir(Path(str(cfg["workdir"])))
    wd.create()
    configure_logging(wd.root, level=str(cfg.get("log_level", "INFO")), for_airflow=True)

    params: dict[str, Any] = {
        "run_tablassert": cfg.get("fullmap") is not None,
        "quarter_limit": int(cfg["quarter_limit"]) if cfg.get("quarter_limit") is not None else None,
        "release_limit": int(cfg["release_limit"]) if cfg.get("release_limit") is not None else None,
        "dailymed_max_age_days": float(cfg["dailymed_max_age_days"]) if cfg.get("dailymed_max_age_days") is not None else None,
        "drugsfda_max_age_days": float(cfg["drugsfda_max_age_days"]) if cfg.get("drugsfda_max_age_days") is not None else None,
        "force": bool(cfg["force"]) if cfg.get("force") is not None else False,
    }
    if cfg.get("fullmap") is not None:
        params["fullmap"] = str(Path(str(cfg["fullmap"])).resolve())
    if cfg.get("drugsfda_url") is not None:
        params["drugsfda_url"] = str(cfg["drugsfda_url"])
    return build_context(wd, cfg.get("fixture_root"), params)


def build_context(wd: Workdir, fixture_root: Path | str | None, params: Mapping[str, Any] | None) -> TaskContext:
    """Assemble the per-task :class:`TaskContext` from a workdir + fixtures + explicit run params.

    Loads the lexical disease map from ``fixture_root`` (when present) and merges it under
    ``params["disease_map"]`` ahead of the caller-supplied params.
    """
    fixture = Path(fixture_root) if fixture_root is not None else None
    disease_map = _load_disease_map(fixture) if fixture is not None else {}
    merged: dict[str, Any] = {"disease_map": disease_map}
    if params:
        merged.update(params)
    # The contraindication shaper builds its own deterministic offline NER backend from
    # ctx.fixture_root (single composite backend). A real production DiseaseNER (offline=False,
    # needs the NER dependencies) may be injected via params["ner"].
    return TaskContext(workdir=wd.root, fixture_root=fixture, params=merged)


def write_build_summary(
    wd: Workdir,
    assertion_refs: list[ArtifactRef],
    kgx_refs: list[ArtifactRef],
    report: translator.ContractReport,
    regression_report: translator.RegressionReport,
) -> Path:
    """Write the build-summary JSON under the workdir's reports dir and return its path."""
    summary_path = wd.reports / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "dakp.build_summary.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "workdir": str(wd.root),
        "tables": [{"name": ref.uri.stem, "path": str(ref.uri), "rows": ref.rows, "artifact_id": ref.blake3} for ref in assertion_refs],
        "tablassert": {"handoff_refs": [str(ref.uri) for ref in kgx_refs]},
        "translator_contract": {"ok": report.ok, "problems": report.problems, "tables": report.tables},
        "translator_regression": {
            "ok": regression_report.ok,
            "families_seen": regression_report.families_seen,
            "row_count": regression_report.row_count,
            "violations": [
                {"family": violation.family, "invariant": violation.invariant, "message": violation.message}
                for violation in regression_report.violations
            ],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    event = "build_summary"
    stats(logger, event, path=str(summary_path), tables=len(assertion_refs), handoff_refs=len(kgx_refs))
    for ref in assertion_refs:
        stats(logger, event, table=ref.uri.stem, rows=ref.rows if ref.rows is not None else "-", blake3=ref.blake3)
    stats(logger, event, contract_ok=report.ok, contract_problems=len(report.problems))
    stats(logger, event, regression_ok=regression_report.ok, regression_violations=len(regression_report.violations))
    return summary_path


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
