"""FAERS quarterly extract fetcher (stub)."""

from __future__ import annotations

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.sources import ingest_fixtures, require_mock

# One FAERS quarter's ASCII files (mock fixture). Real acquisition discovers quarters.
_FAERS_FIXTURES = ("faers/DEMO24Q3.txt", "faers/DRUG24Q3.txt", "faers/INDI24Q3.txt")


class FAERSFetcher:
    """Acquire FAERS quarterly ASCII artifacts.

    Mock profile ingests a single tiny quarter; real quarterly discovery/download with
    ``quarter_limit`` dev mode lands in Milestone 2.
    """

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        require_mock(ctx, "faers")
        return ingest_fixtures(ctx, _FAERS_FIXTURES, namespace="faers")


fetch = FAERSFetcher().fetch

__all__ = ["FAERSFetcher", "fetch"]
