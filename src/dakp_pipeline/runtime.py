"""Airflow-native runtime helpers.

The DAG is the only orchestrator (the pure-Python ``run_pipeline`` CLI runner is retired — see
plans/airflow-native-go-workers.md). Tasks build their :class:`TaskContext` from the ``dakp_config``
Airflow Variable (the single source of run config, shared by the Python tasks and the native Go
bundle workers) via :func:`build_context_from_config`, and the final summary is written by
:func:`write_build_summary`.

These currently delegate to :mod:`dakp_pipeline.pipeline`'s context/summary implementation; that
implementation moves here in full when ``pipeline.py`` is deleted (Phase 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dakp_pipeline import pipeline
from dakp_pipeline.config import load_profile
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import configure_logging
from dakp_pipeline.paths import Workdir
from dakp_pipeline.translator import contract as translator_contract
from dakp_pipeline.translator import regression

__all__ = ["build_context_from_config", "write_build_summary"]


def build_context_from_config(cfg: dict[str, Any]) -> TaskContext:
    """Build a :class:`TaskContext` from the ``dakp_config`` Variable dict.

    Mirrors the old ``_ctx_from_params``: resolves the workdir, configures logging, applies the
    ``quarter_limit`` / ``release_limit`` / ``force`` overrides to the profile, and delegates to
    ``pipeline._build_context`` so disease-map / ``mock_sources`` / ``run_tablassert`` params match
    the canonical runner.
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
    fixture_root = cfg.get("fixture_root")
    return pipeline._build_context(profile, wd, fixture_root, extra or None)


def write_build_summary(
    cfg: dict[str, Any],
    assertion_refs: list[ArtifactRef],
    kgx_refs: list[ArtifactRef],
    report: translator_contract.ContractReport,
    regression_report: regression.RegressionReport,
) -> Path:
    """Write the build summary JSON (delegates to ``pipeline._write_build_summary``)."""
    wd = Workdir(Path(str(cfg["workdir"])))
    return pipeline._write_build_summary(wd, str(cfg.get("profile", "mock")), assertion_refs, kgx_refs, report, regression_report)
