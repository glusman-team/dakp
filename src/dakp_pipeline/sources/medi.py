"""MEDI contraindication list fetcher (stub)."""

from __future__ import annotations

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.sources import ingest_fixtures, require_mock

_MEDI_FIXTURES = ("medi/medi_contraindications.tsv",)


class MEDIFetcher:
    """Acquire the MEDI/matrix contraindication list.

    Mock profile ingests a tiny contraindications fixture; real MEDI acquisition lands
    in Milestone 2.
    """

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        require_mock(ctx, "medi")
        return ingest_fixtures(ctx, _MEDI_FIXTURES, namespace="medi")


fetch = MEDIFetcher().fetch

__all__ = ["MEDIFetcher", "fetch"]
