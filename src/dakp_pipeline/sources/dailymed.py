"""DailyMed SPL fetcher (stub)."""

from __future__ import annotations

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.sources import ingest_fixtures, require_mock

# Fixture paths relative to fixture_root (see tests/fixtures/pipeline/).
_DAILYMED_FIXTURES = ("dailymed/dailymed_spl.xml.gz",)


class DailyMedFetcher:
    """Acquire DailyMed SPL full-release artifacts.

    Mock profile ingests the tiny SPL fixture; real full-release acquisition (idempotent
    download, manifest/checksums, no destructive stashing) lands in Milestone 2.
    """

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        require_mock(ctx, "dailymed")
        return ingest_fixtures(ctx, _DAILYMED_FIXTURES, namespace="dailymed")


# Module-level default for monkeypatchability: tests do `monkeypatch.setattr(dailymed, "fetch", ...)`.
fetch = DailyMedFetcher().fetch

__all__ = ["DailyMedFetcher", "fetch"]
