"""Tablassert handoff: generate Graph/table configs and (optionally) run Tablassert.

DAKP does everything up to the shape Tablassert consumes, then generates a Graph config
plus one table config per assertion table. Canonical entity resolution, KGX compilation,
dedup, deterministic IDs, and RIG generation are delegated to ``../Tablassert`` — DAKP
ships **no** local fallback KGX compiler (PLAN.md "Tablassert modeling layer").
"""

from __future__ import annotations
