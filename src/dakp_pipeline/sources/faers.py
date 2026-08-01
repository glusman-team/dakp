"""FAERS quarterly ASCII acquisition.

Discovers quarterly ASCII zips from the FDA exports index, downloads each into the BLAKE3
content-addressed store (idempotent — re-downloading identical bytes is a cache hit, never a
duplicate store entry), and honors ``quarter_limit`` (process only the N most-recent quarters)
for bounded runs. Unlike the legacy ``getLatest.pl``, this fetcher never destructively renames
or stashes files. Offline tests monkeypatch the module-level :data:`fetch` (or the
:meth:`FAERSFetcher.fetch_index` / :meth:`FAERSFetcher.download_quarter` network boundaries).

Monkeypatchable boundaries:

* :func:`discover_quarters` — pure parser of the FDA index HTML -> most-recent-first quarters.
* :meth:`FAERSFetcher.fetch_index` / :meth:`FAERSFetcher.download_quarter` — the network
  calls; tests patch these (or replace the module-level :data:`fetch`) for offline coverage.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.paths import Workdir

# FDA quarterly-data-extract listing page (anchors discovery of the ASCII zips). The old
# ``https://fis.fda.gov/content/Exports`` index now 404s; the listing moved here. The quarterly
# ASCII zips themselves still live under ``https://fis.fda.gov/content/Exports/`` (the download
# base), named ``faers_ascii_<YYYY>q<N>.zip``.
FDA_FAERS_INDEX_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
#: Where the quarterly ASCII zips are downloaded from (the listing page only anchors discovery).
FDA_FAERS_DOWNLOAD_BASE = "https://fis.fda.gov/content/Exports"
_FAERS_ZIP_RE = re.compile(r"faers_ascii_(\d{4})q(\d)\.zip", re.IGNORECASE)
_DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class QuarterSource:
    """A discovered FAERS quarter: a canonical quarter label and its zip URL."""

    quarter: str  # canonical label, e.g. "24Q3"
    url: str


def discover_quarters(index_html: str, *, base_url: str = FDA_FAERS_DOWNLOAD_BASE) -> list[QuarterSource]:
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


def _apply_quarter_limit[T](quarters: list[T], ctx: TaskContext) -> list[T]:
    """Slice to the N most-recent quarters when ``quarter_limit`` is set (<=0 / None = all)."""
    limit = ctx.params.get("quarter_limit")
    if not isinstance(limit, int) or limit <= 0:
        return quarters
    return quarters[:limit]


class FAERSFetcher:
    """Acquire FAERS quarterly ASCII artifacts over the network."""

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        index_html = self.fetch_index(ctx)
        quarters = _apply_quarter_limit(discover_quarters(index_html), ctx)
        if not quarters:
            logger.warning("no FAERS quarters discovered from index")
            return []
        refs = [self.download_quarter(ctx, source) for source in quarters]
        logger.info("faers acquisition complete", artifacts=len(refs))
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


fetch = FAERSFetcher().fetch

__all__ = ["FDA_FAERS_DOWNLOAD_BASE", "FDA_FAERS_INDEX_URL", "FAERSFetcher", "QuarterSource", "discover_quarters", "fetch"]
