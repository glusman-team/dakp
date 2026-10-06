"""EMA EPAR documents-report fetcher (the manifest that enumerates every EPAR document).

The EMA publishes one fixed-name JSON report listing EVERY document attached to every European
public assessment report (EPAR) — product information (the SmPC/label), assessment reports,
variation reports, RMPs and their translations. DAKP acquires it as the crawl manifest for the
product-information (SmPC) PDFs that carry the ``4.1 Therapeutic indications`` and
``4.3 Contraindications`` sections (:mod:`dakp_pipeline.sources.ema_smpc`).

Content-addressed with BLAKE3 and returned as one :class:`ArtifactRef`. A cached copy younger than
the default seven-day freshness window is reused without network I/O; ``force`` bypasses the gate
(the report is regenerated nightly, so the window bounds staleness exactly as the EMA medicines
export does in :mod:`dakp_pipeline.sources.ema`).

The downloaded payload is PARSED before it is ingested: a truncated or non-JSON body fails the
acquisition loudly instead of surfacing as an empty crawl later.

Selection rule (:func:`product_information_documents`) — the English, human, PDF product-information
documents, verified against the live 2026-10-01 report (20,234 records):

* ``type == "product-information"`` (2,220 rows; every other type is an assessment/variation/RMP
  document with no SmPC sections),
* ``document_url`` under ``/en/documents/product-information/`` and ending ``.pdf`` (the ``-\\d+``
  suffix variant is accepted: Drupal serves a few product-information files as ``..._en.pdf-0``
  with ``content-type: application/pdf``; a sibling ``.xls`` member-state URL list is excluded),
* ``ema_product_number`` starting ``EMEA/H/C/`` (human centralised). ``EMEA/V/C/`` rows are
  veterinary and are dropped here, the same split the medicines-registry extractor makes.

That is 1,945 of the 2,220 product-information rows (274 veterinary + 1 ``.xls``). Authorisation
status is deliberately NOT filtered here: the report's ``status`` field is ``"unknown"`` for
essentially every row, so "is this medicine still authorised" is answered by the join against the
medicines registry at extraction/shaping time (which does carry the withdrawn flag).

Idempotent and non-destructive: the only writes are the content-addressed store copy, its alias,
and its manifest. The download target is monkeypatchable via :func:`download_ema_documents` /
``ctx.params["ema_documents_url"]`` / :pyattr:`EmaDocumentsFetcher.url`.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloader import download
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir

#: The EMA EPAR documents report (fixed name, regenerated nightly from the website content).
EMA_DOCUMENTS_URL = "https://www.ema.europa.eu/en/documents/report/documents-output-epar_documents_json-report_en.json"
#: Store alias for the manifest artifact (distinct from the medicines-registry xlsx alias).
DOCUMENTS_ALIAS = "ema/documents/epar_documents_en.json"
_DEFAULT_MAX_AGE_DAYS = 7.0
#: Narration prefix for every log line this fetcher emits (one stat per line).
_EVENT = "acquire_ema_documents"

#: The report's ``type`` value for the SmPC/label document DAKP mines.
PRODUCT_INFORMATION_TYPE = "product-information"
#: Product-number prefix for human centralised procedures (``EMEA/V/C/`` is veterinary).
HUMAN_CENTRALISED_PREFIX = "EMEA/H/C/"
_EN_PRODUCT_INFORMATION_PATH = "/en/documents/product-information/"
#: Accepts ``..._en.pdf`` and the ``..._en.pdf-0`` variant Drupal serves for a few documents.
_PDF_SUFFIX = re.compile(r"\.pdf(?:-\d+)?$", re.IGNORECASE)

#: Every key the report carries per record. Only ``id``/``type``/``document_url`` are structural:
#: the live 2026-10-01 report has 2 of 20,234 records (an orphan-maintenance report and a
#: tracked-changes document) with no ``medicine_name``/``ema_product_number``, and rejecting the
#: whole report for two non-SmPC rows would fail every acquisition. Records missing those keys parse
#: with ``""`` and the selection rule drops them (a product-information row with no product number
#: cannot be joined to an active substance), counted in :func:`product_information_documents`.
_REQUIRED_KEYS = ("id", "type", "document_url")
_OPTIONAL_KEYS = (
    "medicine_name",
    "ema_product_number",
    "name",
    "status",
    "consultation_date",
    "first_published_date",
    "last_updated_date",
    "reference_number",
)


@dataclass(frozen=True)
class EparDocument:
    """One row of the EPAR documents report: a single published document for one medicine."""

    id: str
    type: str
    medicine_name: str
    ema_product_number: str
    document_url: str
    name: str = ""
    status: str = ""
    consultation_date: str = ""
    first_published_date: str = ""
    last_updated_date: str = ""
    reference_number: str = ""

    @property
    def stem(self) -> str:
        """Document filename with the PDF extension removed, retaining Drupal revisions.

        ``name.pdf`` and ``name.pdf-0`` can name different products. Preserve ``-0`` in the
        staging/cache key so parallel downloads and provenance never collide. Ordinary PDF
        aliases remain unchanged, keeping their existing cached artifacts reusable.
        """
        base = self.document_url.rsplit("/", 1)[-1]
        return re.sub(r"\.pdf(?=-\d+$|$)", "", base, flags=re.IGNORECASE) or base


def product_information_documents(documents: list[EparDocument]) -> list[EparDocument]:
    """The crawl selection: English, human, PDF product-information documents (stable order).

    Sorted by ``(ema_product_number, id)`` so the crawl order — and therefore the fetcher's
    narration and any partial-failure report — is deterministic across runs.
    """
    selected = [doc for doc in documents if _is_english_human_product_information(doc)]
    unattributable = [doc for doc in selected if not doc.medicine_name.strip()]
    if unattributable:
        logger.warning(
            "acquire_ema_documents: {} product-information document(s) carry no medicine_name and are skipped: {}",
            len(unattributable),
            ", ".join(sorted(doc.id for doc in unattributable)),
        )
    kept = [doc for doc in selected if doc.medicine_name.strip()]
    return sorted(kept, key=lambda doc: (doc.ema_product_number, doc.id))


def _is_english_human_product_information(doc: EparDocument) -> bool:
    return (
        doc.type == PRODUCT_INFORMATION_TYPE
        and _EN_PRODUCT_INFORMATION_PATH in doc.document_url
        and _PDF_SUFFIX.search(doc.document_url) is not None
        and doc.ema_product_number.startswith(HUMAN_CENTRALISED_PREFIX)
    )


def parse_documents(payload: Any) -> list[EparDocument]:
    """Validate the report shape and return one :class:`EparDocument` per record.

    Raises ``ValueError`` naming the offending key/record: a payload that is not an object, a
    ``data`` member that is not a list, a record that is not an object, or a record missing a
    required key. Never returns a partial list — a malformed report is an acquisition failure.
    """
    if not isinstance(payload, dict):
        msg = f"EMA documents report: expected a JSON object at the top level, got {type(payload).__name__}"
        raise ValueError(msg)
    data = payload.get("data")
    if not isinstance(data, list):
        msg = f"EMA documents report: 'data' must be a list of records, got {type(data).__name__}"
        raise ValueError(msg)
    documents: list[EparDocument] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            msg = f"EMA documents report: record {index} must be an object, got {type(record).__name__}"
            raise ValueError(msg)
        missing = [key for key in _REQUIRED_KEYS if not str(record.get(key, "")).strip()]
        if missing:
            msg = f"EMA documents report: record {index} (id={record.get('id', '?')!r}) missing required key(s): {', '.join(missing)}"
            raise ValueError(msg)
        documents.append(EparDocument(**{key: str(record.get(key, "") or "") for key in (*_REQUIRED_KEYS, *_OPTIONAL_KEYS)}))
    return documents


def load_documents(path: Path) -> list[EparDocument]:
    """Read and validate a downloaded documents report."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"EMA documents report: cannot read {path}: {exc}"
        raise ValueError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"EMA documents report: {path} is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
        raise ValueError(msg) from exc
    return parse_documents(payload)


class EmaDocumentsFetcher:
    """Acquire the EPAR documents report over the network."""

    url: str = EMA_DOCUMENTS_URL

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, _EVENT):
            store = ArtifactStore(Workdir(ctx.workdir))
            url = str(ctx.params.get("ema_documents_url", self.url))
            force = bool(ctx.params.get("force", False))
            stats(logger, _EVENT, url=url, force=force)
            cached = store.cached_ref(DOCUMENTS_ALIAS)
            manifest = store.read_manifest(cached.blake3) if cached is not None else None
            age = _cache_age_days(manifest)
            max_age = _max_age_days(ctx)
            if (
                not force
                and max_age is not None
                and cached is not None
                and cached.uri.exists()
                and manifest is not None
                and manifest.source.url == url
                and age is not None
                and age < max_age
            ):
                stats(logger, _EVENT, cache_hit=True, age_days=round(age, 2), max_age_days=max_age, blake3=cached.blake3)
                return [cached]

            with tempfile.NamedTemporaryFile(prefix="ema-documents-", suffix=".json", delete=False) as handle:
                dest = Path(handle.name)
            try:
                started = time.monotonic()
                download_ema_documents(url, dest)
                # Fail loudly on a truncated/HTML body BEFORE it lands in the store: an empty
                # crawl three stages later would be far harder to attribute.
                documents = load_documents(dest)
                stats(
                    logger,
                    _EVENT,
                    bytes=dest.stat().st_size,
                    elapsed_s=round(time.monotonic() - started, 3),
                    records=len(documents),
                    product_information=len(product_information_documents(documents)),
                )
                ref, cache_hit = store.ingest(
                    dest,
                    media_type="application/json",
                    alias=DOCUMENTS_ALIAS,
                    source=SourceBlock(url=url, retrieved_at=datetime.now(UTC).isoformat()),
                )
            finally:
                dest.unlink(missing_ok=True)

            stats(logger, _EVENT, blake3=ref.blake3, cache_hit=cache_hit)
            return [ref]


def _max_age_days(ctx: TaskContext) -> float | None:
    """Resolve the documents-report cache window; ``None``/non-positive disables the gate."""
    value = ctx.params.get("ema_documents_max_age_days")
    if value is None:
        return _DEFAULT_MAX_AGE_DAYS
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None
    return _DEFAULT_MAX_AGE_DAYS


def _cache_age_days(manifest: object | None) -> float | None:
    """Return the age of a cached manifest in days, or ``None`` when provenance is unusable."""
    retrieved_at = getattr(getattr(manifest, "source", None), "retrieved_at", None)
    if not retrieved_at:
        return None
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(retrieved_at)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return None


def download_ema_documents(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
    """Download ``url`` to ``dest`` (aria2c-accelerated, stdlib fallback). Monkeypatchable.

    Tests replace this to serve the committed fixture report without network; the real path is
    covered by the offline prod-smoke test and the downloader unit tests.
    """
    return download(url, dest, timeout=timeout, headers={"User-Agent": "dakp-pipeline/0.1"})


fetch = EmaDocumentsFetcher().fetch

__all__ = [
    "DOCUMENTS_ALIAS",
    "EMA_DOCUMENTS_URL",
    "EmaDocumentsFetcher",
    "EparDocument",
    "download_ema_documents",
    "fetch",
    "load_documents",
    "parse_documents",
    "product_information_documents",
]
