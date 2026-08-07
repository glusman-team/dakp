"""FAERS quarterly ASCII acquisition.

Discovers quarterly ASCII zips from the FDA exports index, downloads each into the BLAKE3
content-addressed store, and honors ``quarter_limit`` (process only the N most-recent
quarters) for bounded runs. Acquisition is cache-first: a quarter whose zip is already
content-addressed under its ``faers/faers_ascii_<Q>.zip`` alias is reused WITHOUT touching
the network (FAERS quarterly zips are immutable published snapshots, and fis.fda.gov sends
no ETag/Last-Modified, so alias reuse — not conditional GET — is the skip mechanism);
``force`` (run param) re-downloads everything. Unlike the legacy ``getLatest.pl``, this
fetcher never destructively renames or stashes files. Offline tests monkeypatch the
module-level :data:`fetch` (or the :meth:`FAERSFetcher.fetch_index` /
:meth:`FAERSFetcher.download_quarter` network boundaries).

Monkeypatchable boundaries:

* :func:`discover_quarters` — pure parser of the FDA index HTML -> most-recent-first quarters.
* :meth:`FAERSFetcher.fetch_index` / :meth:`FAERSFetcher.download_quarter` — the network
  calls; tests patch these (or replace the module-level :data:`fetch`) for offline coverage.
"""

from __future__ import annotations

import re
import shutil
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import logger, stats, step
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
#: Narration prefix for every log line this fetcher emits (one stat per line).
_EVENT = "acquire_faers"


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
        with step(logger, _EVENT):
            index_html = self.fetch_index(ctx)
            discovered = discover_quarters(index_html)
            stats(logger, _EVENT, quarters_discovered=len(discovered))
            for source in discovered:
                stats(logger, _EVENT, level="DEBUG", quarter=source.quarter, url=source.url)
            quarters = _apply_quarter_limit(discovered, ctx)
            stats(logger, _EVENT, quarters_to_acquire=len(quarters))
            if not quarters:
                logger.warning("{}: no FAERS quarters discovered from index", _EVENT)
                return []
            refs = [self.download_quarter(ctx, source) for source in quarters]
            stats(logger, _EVENT, artifacts=len(refs))
            return refs

    def fetch_index(self, ctx: TaskContext) -> str:
        """Fetch the FDA exports index HTML. Network boundary; monkeypatchable."""
        stats(logger, f"{_EVENT} index", url=FDA_FAERS_INDEX_URL)
        started = time.monotonic()
        index_html = _http_get_text(FDA_FAERS_INDEX_URL, timeout=_DEFAULT_TIMEOUT)
        stats(logger, f"{_EVENT} index", bytes=len(index_html), elapsed_s=round(time.monotonic() - started, 3))
        return index_html

    def download_quarter(self, ctx: TaskContext, source: QuarterSource) -> ArtifactRef:
        """Acquire one quarter zip into the content-addressed store (cache-first).

        A quarter already present under its alias is reused WITHOUT a network call (FAERS
        quarterly zips are immutable published snapshots; fis.fda.gov sends no ETag or
        Last-Modified, so conditional GET is impossible and alias reuse is the skip).
        ``force`` re-downloads unconditionally. Downloads stream in 1 MiB chunks (a quarter
        zip never sits whole in memory) and the staged copy is removed once content-addressed.
        """
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        alias = f"faers/faers_ascii_{source.quarter}.zip"
        quarter_event = f"{_EVENT} quarter"
        if not bool(ctx.params.get("force", False)):
            cached = store.cached_ref(alias)
            if cached is not None and cached.uri.exists():
                stats(logger, quarter_event, quarter=source.quarter, cache_hit=True, blake3=cached.blake3)
                return cached
        stats(logger, quarter_event, quarter=source.quarter)
        stats(logger, quarter_event, level="DEBUG", url=source.url)
        dest = wd.raw / "downloads" / f"faers_ascii_{source.quarter}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        _http_download(source.url, dest, timeout=_DEFAULT_TIMEOUT)
        stats(logger, quarter_event, bytes=dest.stat().st_size, elapsed_s=round(time.monotonic() - started, 3))
        ref, cache_hit = store.ingest(dest, alias=alias, source=SourceBlock(url=source.url, retrieved_at=datetime.now(UTC).isoformat()))
        dest.unlink(missing_ok=True)  # staged copy no longer needed once content-addressed
        stats(logger, quarter_event, blake3=ref.blake3, cache_hit=cache_hit)
        return ref


# --- stdlib HTTP helpers (no new deps; downloads.py stays untouched) --------------


def _http_get_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _http_download(url: str, dest: Path, *, timeout: float) -> Path:
    """Stream ``url`` to ``dest`` in 1 MiB chunks (a whole quarter zip never sits in memory).
    HTTP/URL errors propagate (fail loudly)."""
    with urllib.request.urlopen(url, timeout=timeout) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    return dest


fetch = FAERSFetcher().fetch

__all__ = ["FDA_FAERS_DOWNLOAD_BASE", "FDA_FAERS_INDEX_URL", "FAERSFetcher", "QuarterSource", "discover_quarters", "fetch"]
