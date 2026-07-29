"""Tablassert handoff: generate Graph/table configs and (optionally) run Tablassert.

DAKP does everything up to the shape Tablassert consumes, then generates a Graph config
plus one table config per assertion table. Canonical entity resolution, KGX compilation,
dedup, deterministic IDs, and RIG generation are delegated to ``../Tablassert`` — DAKP
ships **no** local fallback KGX compiler (PLAN.md "Tablassert modeling layer").

``run`` is re-exported here so tests can ``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)``
and the pure-Python runner picks up the patched callable (PLAN.md integration-test sketch).
"""

from __future__ import annotations

from dakp_pipeline.tablassert.configs import generate
from dakp_pipeline.tablassert.run import run

__all__ = ["generate", "run"]
