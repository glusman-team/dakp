"""MEDI / matrix contraindication list fetcher.

Acquires the MEDI contraindication list (the legacy ``matrix`` used
``contraindicationList-<version>.xlsx`` from the ``everycure-org/matrix-indication-list``
GitHub releases) behind the :class:`~dakp_pipeline.io.contracts.Fetcher` protocol.

* ``mock`` profile: ingests a tiny TSV fixture under ``fixture_root``. The fixture carries
  the real MEDI column shape so downstream parsing is faithful even though the on-disk
  format is TSV (see :mod:`dakp_pipeline.extract.medi` for the dual TSV/xlsx reader).
* real profiles (``sample`` / ``wenceslaus_full``): download the versioned ``.xlsx``
  release asset via :func:`http_download` — a single monkeypatchable seam — then
  content-address it (BLAKE3).

Idempotent: identical bytes are a cache hit in the content-addressed store (keyed by
BLAKE3, never filename/mtime). ``medi_version`` is captured in the artifact alias and in
the fetch manifest's operation ``config_hash`` so the version is recoverable from
provenance alone.
"""

from __future__ import annotations

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_bytes
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloads import http_download
from dakp_pipeline.io.manifests import OperationBlock, SourceBlock
from dakp_pipeline.paths import Workdir

# Version stamped into the mock fixture artifact (the real list ships per-release tags).
DEFAULT_MEDI_VERSION = "MEDI-0.0-mock"

# everycure-org/matrix-indication-list release asset URL template (real profiles).
# The exact tag/asset naming evolves with the upstream release; override the whole URL
# via ``params["medi_url"]`` or the template via ``params["medi_url_template"]``.
_DEFAULT_RELEASE_URL_TEMPLATE = (
    "https://github.com/everycure-org/matrix-indication-list/releases/download/v{version}/contraindicationList-{version}.xlsx"
)

_FIXTURE_NAME = "medi/medi_contraindications.tsv"
_OPERATION = "fetch_medi"


def resolve_medi_version(ctx: TaskContext) -> str:
    """The MEDI list version to acquire.

    Priority: ``params["medi_version"]`` -> profile default (mock pins a fixture version;
    real profiles default to ``"latest"`` and must resolve to a concrete tag/URL).
    """
    value = ctx.params.get("medi_version")
    if value is not None:
        return str(value)
    return DEFAULT_MEDI_VERSION if ctx.profile == "mock" else "latest"


class MEDIFetcher:
    """Acquire the MEDI/matrix contraindication list (mock fixture or real release asset)."""

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        version = resolve_medi_version(ctx)
        store = ArtifactStore(Workdir(ctx.workdir))
        if ctx.profile == "mock":
            return self._fetch_mock(ctx, store, version)
        return self._fetch_real(ctx, store, version)

    # -- profiles --------------------------------------------------------------
    def _fetch_mock(self, ctx: TaskContext, store: ArtifactStore, version: str) -> list[ArtifactRef]:
        if ctx.fixture_root is None:
            msg = "TaskContext.fixture_root is None; cannot resolve MEDI fixture"
            raise ValueError(msg)
        path = ctx.fixture_root / _FIXTURE_NAME
        if not path.exists():
            msg = f"MEDI contraindication fixture not found: {path}"
            raise FileNotFoundError(msg)
        ref, _ = store.ingest(
            path, alias=f"medi/contraindicationList-{version}", operation=_operation(version), source=SourceBlock(url=f"fixture:{_FIXTURE_NAME}")
        )
        return [ref]

    def _fetch_real(self, ctx: TaskContext, store: ArtifactStore, version: str) -> list[ArtifactRef]:
        url = _resolve_asset_url(ctx, version)
        filename = _asset_filename(version)
        dest = Workdir(ctx.workdir).raw / "medi" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            # Single monkeypatchable seam: tests replace ``medi.http_download`` to avoid
            # the network. ``http_download`` is a Milestone-2 stub until real acquisition
            # lands at the io layer; it raises loudly outside the mock profile.
            http_download(url, dest)
        ref, _ = store.ingest(
            dest,
            alias=f"medi/contraindicationList-{version}",
            operation=_operation(version),
            source=SourceBlock(url=url, retrieved_at=_retrieved_at()),
        )
        return [ref]


# --- helpers --------------------------------------------------------------------


def _operation(version: str) -> OperationBlock:
    """Fetch operation block; ``config_hash`` deterministically encodes ``medi_version``."""
    return OperationBlock(name=_OPERATION, config_hash=hash_bytes(version.encode("utf-8")))


def _resolve_asset_url(ctx: TaskContext, version: str) -> str:
    """Resolve the release-asset URL for ``version`` (explicit URL > template)."""
    explicit = ctx.params.get("medi_url")
    if explicit is not None:
        return str(explicit)
    if version == "latest":
        # Resolving the latest release tag needs the GitHub releases API (network); that
        # acquisition helper is Milestone-2 work. Fail loudly rather than fetching a
        # fabricated URL.
        msg = "MEDI version is 'latest' but latest-tag resolution is not implemented; pin params['medi_version'] or pass params['medi_url']."
        raise NotImplementedError(msg)
    template = ctx.params.get("medi_url_template")
    template_str = str(template) if template is not None else _DEFAULT_RELEASE_URL_TEMPLATE
    return template_str.format(version=version)


def _asset_filename(version: str) -> str:
    return f"contraindicationList-{version}.xlsx"


def _retrieved_at() -> str:
    """Current UTC timestamp for real-download provenance (ISO 8601, second precision)."""
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


fetch = MEDIFetcher().fetch

__all__ = ["DEFAULT_MEDI_VERSION", "MEDIFetcher", "fetch", "resolve_medi_version"]
