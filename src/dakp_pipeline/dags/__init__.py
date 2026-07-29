"""Airflow DAGs for the DAKP pipeline.

The :mod:`dakp_pipeline.dags.dakp_build` module imports cleanly with or without Airflow;
the helpers below give a small, explicit entry point for getting/running the DAG and fail
with a clear message (not an opaque ``ImportError``) when the optional airflow extra is
absent. Imports are lazy so ``import dakp_pipeline.dags`` stays lightweight.
"""

from __future__ import annotations

from typing import Any

__all__ = ["airflow_available", "get_dag", "run_dag"]


def airflow_available() -> bool:
    """Return ``True`` if apache-airflow is importable (the optional extra is installed)."""
    from dakp_pipeline.dags.dakp_build import _AIRFLOW_AVAILABLE

    return _AIRFLOW_AVAILABLE


def get_dag() -> Any:
    """Return the constructed ``dakp_build`` Airflow DAG object.

    Raises:
        RuntimeError: if apache-airflow is not installed (install ``uv sync --extra airflow``).
    """
    from dakp_pipeline.dags import dakp_build

    if not dakp_build._AIRFLOW_AVAILABLE:
        msg = "apache-airflow is not installed; install the optional extra with `uv sync --extra airflow`"
        raise RuntimeError(msg)
    return dakp_build.dag_obj


def run_dag(**params: Any) -> Any:
    """Execute the DAG in-process via Airflow's ``DAG.test()`` (local development only).

    Any keyword arguments are passed as the DAG run's ``run_conf`` (overriding DAG params).
    Production runs should use ``airflow dags trigger dakp_build`` instead.

    Raises:
        RuntimeError: if apache-airflow is not installed.
    """
    dag = get_dag()
    return dag.test(run_conf=params) if params else dag.test()
