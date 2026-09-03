"""Airflow DAGs for the DAKP pipeline.

Airflow 3 is a hard dependency (the pipeline is Airflow-native), so :mod:`dakp_pipeline.dags.dakp_build` always imports and
constructs the DAG at module load. :func:`get_dag` returns the constructed DAG object; the
one-command ``uv run dakp up`` orchestrator (:mod:`dakp_pipeline.cli`) triggers it via ``airflow dags
trigger dakp_pipeline``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["airflow_available", "get_dag"]


def airflow_available() -> bool:
    """Return ``True`` if apache-airflow is importable (it is a hard dependency, so always True)."""
    import importlib.util

    return importlib.util.find_spec("airflow") is not None


def get_dag() -> Any:
    """Return the constructed ``dakp_pipeline`` Airflow DAG object."""
    from dakp_pipeline.dags import dakp_build

    return dakp_build.dag_obj
