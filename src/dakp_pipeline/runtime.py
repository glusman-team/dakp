"""Airflow-native runtime helpers — the canonical home for run-context + build-summary logic.

The DAG is the only orchestrator (the pure-Python ``run_pipeline`` CLI runner is retired — see
plans/airflow-native-go-workers.md). Tasks build their :class:`TaskContext` from the ``dakp_config``
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

from dakp_pipeline.config import Profile, load_profile
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import configure_logging
from dakp_pipeline.paths import Workdir
from dakp_pipeline.translator import contract as translator_contract
from dakp_pipeline.translator import regression

__all__ = ["build_context", "build_context_from_config", "write_build_summary"]


def build_context_from_config(cfg: Mapping[str, Any]) -> TaskContext:
    """Build a :class:`TaskContext` from the ``dakp_config`` Variable dict.

    Resolves the workdir, configures logging, applies the ``quarter_limit`` / ``release_limit`` /
    ``force`` overrides to the profile, and delegates to :func:`build_context` so disease-map /
    ``mock_sources`` / ``run_tablassert`` params are populated exactly as the canonical runner did.
    """
    overrides: dict[str, object] = {}
    if cfg.get("quarter_limit") is not None:
        overrides["quarter_limit"] = int(cfg["quarter_limit"])
    if cfg.get("release_limit") is not None:
        overrides["release_limit"] = int(cfg["release_limit"])
    if cfg.get("force") is not None:
        overrides["force"] = bool(cfg["force"])
    profile = load_profile(str(cfg.get("profile", "mock")), **overrides)

    wd = Workdir(Path(str(cfg["workdir"])))
    wd.create()
    configure_logging(wd.root, level=str(cfg.get("log_level", "INFO")), for_airflow=True)

    extra: dict[str, object] = {}
    if profile.download.drugsfda_url:
        extra["drugsfda_url"] = profile.download.drugsfda_url
    return build_context(profile, wd, cfg.get("fixture_root"), extra or None)


def build_context(profile: Profile, wd: Workdir, fixture_root: Path | str | None, extra: Mapping[str, Any] | None) -> TaskContext:
    """Assemble the per-task :class:`TaskContext` (params dict) from a profile + workdir + fixtures."""
    fixture = Path(fixture_root) if fixture_root is not None else None
    disease_map = _load_disease_map(fixture) if fixture is not None else {}
    params: dict[str, Any] = {
        "disease_map": disease_map,
        "mock_sources": profile.mock_sources,
        "run_tablassert": profile.run_tablassert,
        "quarter_limit": profile.quarter_limit,
        "release_limit": profile.release_limit,
        "force": profile.force,
    }
    if extra:
        params.update(extra)
    # The contraindication shaper builds its own deterministic offline NER backend from
    # ctx.fixture_root (single composite backend). A real production DiseaseNER (offline=False,
    # needs the NER dependencies) may be injected via params["ner"].
    return TaskContext(
        profile=profile.name, workdir=wd.root, fixture_root=fixture, threads=profile.threads, memory_budget_gb=profile.memory_budget_gb, params=params
    )


def write_build_summary(
    wd: Workdir,
    profile: str,
    assertion_refs: list[ArtifactRef],
    kgx_refs: list[ArtifactRef],
    report: translator_contract.ContractReport,
    regression_report: regression.RegressionReport,
) -> Path:
    """Write the build-summary JSON under the workdir's reports dir and return its path."""
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
