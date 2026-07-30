"""FAERS quarterly ASCII acquisition (Milestone 2).

Real acquisition discovers quarterly ASCII zips from the FDA exports index, downloads each
into the BLAKE3 content-addressed store (idempotent — re-downloading identical bytes is a
cache hit, never a duplicate store entry), and honors ``quarter_limit`` (process only the N
most-recent quarters) for dev/sample runs. Unlike the legacy ``getLatest.pl``, this fetcher
never destructively renames or stashes files.

The ``mock`` profile resolves tiny ``$``-delimited fixture files under
``ctx.fixture_root/faers``; the extractor reads those loose ``.txt`` files directly (no
network, no zip).

Monkeypatchable boundaries:

* :func:`discover_quarters` — pure parser of the FDA index HTML -> most-recent-first quarters.
* :meth:`FAERSFetcher.fetch_index` / :meth:`FAERSFetcher.download_quarter` — the network
  calls; tests patch these (or replace the module-level :data:`fetch`) for offline coverage.
"""

from __future__ import annotations

import re
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import ingest_fixtures

# FDA exports index lists quarterly ASCII zips as faers_ascii_<YYYY>q<N>.zip.
FDA_FAERS_INDEX_URL = "https://fis.fda.gov/content/Exports"
_FAERS_ZIP_RE = re.compile(r"faers_ascii_(\d{4})q(\d)\.zip", re.IGNORECASE)
# Quarter label embedded in a FAERS filename, e.g. DEMO24Q3.txt or faers_ascii_2024q3.zip.
_QUARTER_IN_NAME_RE = re.compile(r"(\d{2})q(\d)", re.IGNORECASE)
_DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class QuarterSource:
    """A discovered FAERS quarter: a canonical quarter label and its zip URL."""

    quarter: str  # canonical label, e.g. "24Q3"
    url: str


def discover_quarters(index_html: str, *, base_url: str = FDA_FAERS_INDEX_URL) -> list[QuarterSource]:
    """Parse FDA index HTML into most-recent-first ``(quarter, url)`` entries.

    Pure function (no network). Quarter labels are canonicalized to ``YYQ<N>`` (e.g.
    ``2024q3`` -> ``24Q3``) so they line up with FAERS ASCII filenames and the legacy
    ``\\d\\dQ\\d`` quarter convention used by ``listCases.pl`` / ``drug2indi.pl``.
    """
    found: dict[str, str] = {}
    for match in _FAERS_ZIP_RE.finditer(index_html):
        filename = match.group(0)
        # The zip regex already captures (year, quarter); a match always has both groups.
        label = f"{match.group(1)[-2:]}Q{match.group(2)}"
        url = filename if filename.lower().startswith("http") else f"{base_url.rstrip('/')}/{filename}"
        found[label] = url
    # Lexicographic reverse sort == most-recent first for zero-padded 2-digit years
    # (covers FAERS quarters from 2004 through 2099).
    return [QuarterSource(label, found[label]) for label in sorted(found, reverse=True)]


def _fixture_quarter(name: str) -> str | None:
    """Canonical quarter label for a fixture filename like ``DEMO24Q3.txt``."""
    match = _QUARTER_IN_NAME_RE.search(name)
    return f"{match.group(1)}Q{match.group(2)}".upper() if match else None


def _apply_quarter_limit[T](quarters: list[T], ctx: TaskContext) -> list[T]:
    """Slice to the N most-recent quarters when ``quarter_limit`` is set (<=0 / None = all)."""
    limit = ctx.params.get("quarter_limit")
    if not isinstance(limit, int) or limit <= 0:
        return quarters
    return quarters[:limit]


class FAERSFetcher:
    """Acquire FAERS quarterly ASCII artifacts (real download or mock fixtures)."""

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        refs = self._fetch_fixtures(ctx) if ctx.profile == "mock" else self._fetch_remote(ctx)
        logger.info("faers acquisition complete", profile=ctx.profile, artifacts=len(refs))
        return refs

    # -- mock profile ---------------------------------------------------------
    def _fetch_fixtures(self, ctx: TaskContext) -> list[ArtifactRef]:
        if ctx.fixture_root is None:
            msg = "TaskContext.fixture_root is None; cannot resolve FAERS fixtures"
            raise ValueError(msg)
        faers_dir = ctx.fixture_root / "faers"
        by_quarter: dict[str, list[str]] = defaultdict(list)
        for path in sorted(faers_dir.glob("*.txt")):
            quarter = _fixture_quarter(path.name)
            if quarter is not None:
                by_quarter[quarter].append(path.name)
        quarters = _apply_quarter_limit(sorted(by_quarter, reverse=True), ctx)
        names: list[str] = []
        for quarter in quarters:
            names.extend(f"faers/{name}" for name in sorted(by_quarter[quarter]))
        if not names:
            logger.warning("no FAERS fixtures found", fixture_root=str(faers_dir))
            return []
        return ingest_fixtures(ctx, tuple(names), namespace="faers")

    # -- real profile ---------------------------------------------------------
    def _fetch_remote(self, ctx: TaskContext) -> list[ArtifactRef]:
        index_html = self.fetch_index(ctx)
        quarters = _apply_quarter_limit(discover_quarters(index_html), ctx)
        if not quarters:
            logger.warning("no FAERS quarters discovered from index")
            return []
        refs: list[ArtifactRef] = [self.download_quarter(ctx, source) for source in quarters]
        return refs

    def fetch_index(self, ctx: TaskContext) -> str:
        """Fetch the FDA exports index HTML. Network boundary; monkeypatchable."""
        return _http_get_text(FDA_FAERS_INDEX_URL, timeout=_DEFAULT_TIMEOUT)

    def download_quarter(self, ctx: TaskContext, source: QuarterSource) -> ArtifactRef:
        """Download one quarter zip into the content-addressed store (idempotent by hash)."""
        wd = Workdir(ctx.workdir)
        dest = wd.raw / "downloads" / f"faers_ascii_{source.quarter}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _http_download(source.url, dest, timeout=_DEFAULT_TIMEOUT)
        store = ArtifactStore(wd)
        ref, cache_hit = store.ingest(
            dest, alias=f"faers/faers_ascii_{source.quarter}.zip", source=SourceBlock(url=source.url, retrieved_at=datetime.now(UTC).isoformat())
        )
        logger.info("faers quarter acquired", quarter=source.quarter, cache_hit=cache_hit, artifact_id=ref.blake3)
        return ref


# --- stdlib HTTP helpers (no new deps; downloads.py stays untouched) --------------


def _http_get_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _http_download(url: str, dest: Path, *, timeout: float) -> Path:
    """Download ``url`` to ``dest``. HTTP/URL errors propagate (fail loudly)."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        dest.write_bytes(response.read())
    return dest


def iter_quarter_sources(quarters: Iterable[QuarterSource]) -> Iterable[QuarterSource]:
    """Expose discovered quarters for inspection/tests (identity passthrough)."""
    yield from quarters


fetch = FAERSFetcher().fetch

__all__ = ["FDA_FAERS_INDEX_URL", "FAERSFetcher", "QuarterSource", "discover_quarters", "fetch", "iter_quarter_sources"]
