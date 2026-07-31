"""DAKP pipeline: reproducible, content-addressed build of treatment and contraindication assertion tables.

See ``PLAN.md`` for the approved specification. The package ships a BLAKE3
content-addressed store, real source acquisition/extraction, assertion shaping, and Airflow
orchestration. Offline execution is purely a test concern (monkeypatched fetchers + fixtures).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
