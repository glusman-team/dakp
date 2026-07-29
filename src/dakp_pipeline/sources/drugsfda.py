"""Drugs@FDA product/application fetcher (stub)."""

from __future__ import annotations

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.sources import ingest_fixtures, require_mock

_DRUGSFDA_FIXTURES = ("drugsfda/drugsfda_products.tsv",)


class DrugsFDAFetcher:
    """Acquire Drugs@FDA product/application tables.

    Mock profile ingests a tiny products fixture; real Drugs@FDA download/extract lands
    in Milestone 2.
    """

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        require_mock(ctx, "drugsfda")
        return ingest_fixtures(ctx, _DRUGSFDA_FIXTURES, namespace="drugsfda")


fetch = DrugsFDAFetcher().fetch

__all__ = ["DrugsFDAFetcher", "fetch"]
