"""DAKP pipeline: reproducible, content-addressed build of treatment and contraindication assertion tables.

The package ships a BLAKE3
content-addressed store, real source acquisition/extraction, assertion shaping, and Airflow
orchestration. Offline execution is purely a test concern (monkeypatched fetchers + fixtures).
"""

from __future__ import annotations

#: Package version. MUST match ``project.version`` in ``pyproject.toml`` — the artifact
#: names the release stage publishes (``drug_approvals_kg_*_v<version>``) come from HERE, not
#: from the installed distribution metadata, so a bump that touches only ``pyproject.toml``
#: silently ships the previous version's file names. ``tests/unit/test_version.py`` pins the two
#: together.
__version__ = "1.3.2"

__all__ = ["__version__"]
