"""DAKP pipeline: reproducible, content-addressed build of treatment and contraindication assertion tables.

See ``PLAN.md`` for the full approved specification. Milestone 1 ships the project
scaffold, BLAKE3 content-addressed store, and a fully mocked end-to-end pipeline that
runs with no network and no real Tablassert/Airflow installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
