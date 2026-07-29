"""Reference Ingest Guide (RIG) content (stub).

The DAKP RIG is preserved/updated from ``../DINGO`` during Milestone 7; for now this
returns a placeholder string so the pipeline shape is complete. Tablassert performs the
real RIG generation where supported.
"""

from __future__ import annotations

_RIG_STUB = """# DAKP Reference Ingest Guide (stub)

Source: infores:multiomics-drugapprovals
Predicates: biolink:treats, biolink:applied_to_treat, biolink:contraindicated_in
Upstream: infores:dailymed, infores:faers, infores:medi

Full RIG content is ported from ../DINGO during Milestone 7.
"""


def rig_text() -> str:
    """Return placeholder RIG content (real content lands in Milestone 7)."""
    return _RIG_STUB


__all__ = ["rig_text"]
