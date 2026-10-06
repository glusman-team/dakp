"""EMA product-information (SmPC) section extractor.

Turns the crawled product-information PDFs (:mod:`dakp_pipeline.sources.ema_smpc`) plus the EPAR
documents report that enumerated them (:mod:`dakp_pipeline.sources.ema_documents`) into one
normalized interim parquet table (``data/interim/ema/smpc_sections.parquet``): one row per
(medicine, mined section), carrying the section text verbatim for the assertion-shaping stage to
mine with the DiseaseNER.

Why this is a separate stage from the shaping: the SmPC is the EU equivalent of the DailyMed SPL
label, and the pipeline already separates "cut the label into sections" (:mod:`extract.spl_xml`)
from "mine the sections into assertions". Keeping the same split means the EU contraindication and
indication passes reuse the existing mining, corroboration and aggregation machinery untouched.

Design notes:

* **Annex I only.** A product-information PDF is Annex I (SmPC) + Annex II + Annex III (labelling
  and package leaflet). The leaflet repeats consumer-facing wording ("What X is and what it is used
  for", "Do not use X if ...") that is NOT the regulatory section text, so extraction is cut to
  Annex I: from the first ``ANNEX I`` line to the next ``ANNEX <roman>`` marker.
* **QRD headings, tolerant match.** The EU Query Review Document template numbers the sections, so
  ``4.1 Therapeutic indications`` / ``4.3 Contraindications`` / ``4.4 Special warnings and special
  precautions for use`` are located by a line-start ``<number> <title>`` pattern with a
  case-insensitive title-prefix check. That covers the observed lower-case form, the older
  upper-case form, the dotted ``4.3.`` variant, and long titles that the text extraction wrapped.
  A section ends at the NEXT numbered heading of any kind, so a missing 4.2 cannot swallow 4.3.
* **One row per (product, section).** A medicine can carry two product-information documents in the
  report (observed: the pre-endorsement 2021 COVID-19 Vaccine AstraZeneca document alongside the
  2024 Vaxzevria one), so duplicates are resolved newest-``last_updated_date``-wins and the
  superseded document is recorded as a warning instead of producing two conflicting rows.
* **Nothing is dropped silently.** An unreadable PDF, an empty text layer, a missing expected
  section and a superseded document each produce a structured warning row with a stable code and a
  count, exactly as the FAERS and SPL extractors do.
* **Fast by construction.** pypdf text extraction is the whole cost (the 2026-10 production
  corpus: 1,945 documents, 119k pages, ~5,300 CPU-seconds; ~89 minutes serial). Two levers keep
  it to minutes with identical rows and warnings on that live corpus: (1) a document is read page by page and stops
  at the first Annex II-V marker after Annex I, the exact boundary :func:`cut_sections` cuts at,
  so the labelling and leaflet pages (most of the document) are never decoded; (2) documents are
  parsed on a spawn process pool, largest first, and reassembled in input order, so the rows and
  warnings are identical to a serial run.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import bind, stats
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources.ema_documents import EparDocument, load_documents, product_information_documents

# --- normalized column contract ---------------------------------------------------

SMPC_SECTIONS_COLUMNS: list[str] = [
    "source_record_id",
    "ema_product_number",
    "medicine_name",
    "section_kind",
    "section_title",
    "section_text",
    "document_url",
    "last_updated_date",
    "pdf_path",
]
SMPC_WARNINGS_COLUMNS: list[str] = ["ema_product_number", "code", "message", "count"]

#: section_kind -> the QRD number and the title prefix that confirms it. The prefix check keeps a
#: stray "4.3" line (a cross-reference, a table row) from being read as a section heading.
_MINED_SECTIONS: dict[str, tuple[str, str]] = {
    "indications": ("4.1", "therapeutic indication"),
    "contraindications": ("4.3", "contraindication"),
    "warnings": ("4.4", "special warning"),
}
#: section_kind order in the interim table (also the QRD order).
_SECTION_ORDER: tuple[str, ...] = ("indications", "contraindications", "warnings")
#: QRD number -> (section_kind, title prefix), the reverse index used while scanning headings.
_BY_NUMBER: dict[str, tuple[str, str]] = {number: (kind, prefix) for kind, (number, prefix) in _MINED_SECTIONS.items()}

_ANNEX_I = re.compile(r"^[ \t]*ANNEX[ \t]+I[ \t]*$", re.IGNORECASE | re.MULTILINE)
_NEXT_ANNEX = re.compile(r"^[ \t]*ANNEX[ \t]+(?:II|III|IV|V)\b", re.IGNORECASE | re.MULTILINE)
#: A numbered QRD heading at the start of a line: "4.3 Contraindications", "4.3.", "5. PHARMACOLOGICAL".
_HEADING = re.compile(r"^[ \t]*(\d{1,2}(?:\.\d{1,2})?)\.?[ \t]+(\S.*)$", re.MULTILINE)
_WS = re.compile(r"\s+")
#: Symbol/Wingdings bullets extract as Unicode private-use code points (observed: U+F0B7 before
#: every contraindication bullet in a live EMA SmPC). They are layout, not text.
_PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
#: A line that is only a page number (the PDF footer) - it would otherwise trail a section body.
_PAGE_NUMBER_LINE = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.MULTILINE)

#: Narration prefix for every log line this extractor emits (one stat per line).
_EVENT = "extract_ema_smpc"
# Change when parsing semantics or output schemas change; worker count cannot change output.
_PARSE_CACHE_VERSION = "smpc-annex-i-v1"
_MAX_PARSE_WORKERS = 32


class EmaSmpcExtractor:
    """Parse the crawled product-information PDFs into the normalized section table."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        log = bind(task_id=_EVENT)

        manifest_ref = next((ref for ref in inputs if ref.uri.suffix.lower() == ".json"), None)
        if manifest_ref is None:
            msg = "no EMA documents report (json) among the inputs: the crawl manifest carries the per-document metadata"
            raise ValueError(msg)
        pdf_refs = [ref for ref in inputs if ref.uri.suffix.lower() == ".pdf"]
        if not pdf_refs:
            msg = "no EMA product-information PDFs among the inputs"
            raise ValueError(msg)

        input_ids = [ref.blake3 for ref in inputs]
        cache_inputs = [*input_ids, _PARSE_CACHE_VERSION, *(str(ref.uri) for ref in inputs)]
        if not ctx.params.get("force"):
            cached = store.find_by_operation(_EVENT, cache_inputs)
            if cached is not None:
                stats(log, _EVENT, cache_hit=True, outputs=len(cached))
                return cached

        documents = {doc.stem: doc for doc in product_information_documents(load_documents(manifest_ref.uri))}
        rows, warnings = parse_documents(pdf_refs, documents, workers=_workers(ctx))

        operation = OperationBlock(name=_EVENT)
        interim_dir = wd.interim / "ema"
        sections_fp = schemas.schema_fingerprint(SMPC_SECTIONS_COLUMNS)
        warnings_fp = schemas.schema_fingerprint(SMPC_WARNINGS_COLUMNS)
        refs = [
            _write_parquet(
                rows, SMPC_SECTIONS_COLUMNS, interim_dir / "smpc_sections.parquet", store, operation, sections_fp, len(warnings), input_ids
            ),
            _write_parquet(
                warnings, SMPC_WARNINGS_COLUMNS, interim_dir / "smpc_warnings.parquet", store, operation, warnings_fp, len(warnings), input_ids
            ),
        ]
        store.record_operation(_EVENT, cache_inputs, refs)
        stats(log, _EVENT, pdfs=len(pdf_refs), rows=len(rows), warnings=len(warnings), outputs=len(refs), sections=",".join(_SECTION_ORDER))
        return refs


def parse_documents(
    pdf_refs: list[ArtifactRef], documents: dict[str, EparDocument], *, workers: int = 1
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse every crawled PDF into section rows + warning rows (pure; deterministic order).

    ``documents`` maps the manifest's document stem to its record, so each PDF can be attributed to
    a medicine, a product number and a source URL. A PDF whose stem is not in the manifest is a
    warning, not a row: without the metadata the section could not be joined to an active substance
    or carry an approval id, and inventing one would be worse than reporting the gap.

    ``workers > 1`` reads the PDFs on a spawn process pool (:func:`_read_texts`); the result is
    identical to a serial run because texts are reassembled in input order before anything else.
    """
    rows: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    known = [(ref, documents[ref.uri.stem]) for ref in pdf_refs if ref.uri.stem in documents]
    # Identity preserves distinct refs with equal content; `known` keeps every ref alive.
    texts = dict(zip((id(ref) for ref, _ in known), _read_texts([ref.uri for ref, _ in known], workers), strict=True))
    for ref in pdf_refs:
        stem = ref.uri.stem
        doc = documents.get(stem)
        if doc is None:
            warnings.append(_warning("", "unknown_document", f"PDF {ref.uri.name} is not in the documents report selection"))
            continue
        text, error = texts[id(ref)]
        if error is not None:
            warnings.append(_warning(doc.ema_product_number, "pdf_unreadable", f"{doc.medicine_name}: {error}"))
            continue
        if not text.strip():
            warnings.append(_warning(doc.ema_product_number, "no_text", f"{doc.medicine_name}: the PDF has no extractable text layer"))
            continue
        sections, section_warnings = cut_sections(text)
        warnings.extend(_warning(doc.ema_product_number, code, message) for code, message in section_warnings)
        for kind in _SECTION_ORDER:
            found = sections.get(kind)
            if found is None:
                continue
            title, body = found
            rows.append(
                {
                    "source_record_id": f"{doc.ema_product_number}#{kind}",
                    "ema_product_number": doc.ema_product_number,
                    "medicine_name": doc.medicine_name,
                    "section_kind": kind,
                    "section_title": title,
                    "section_text": body,
                    "document_url": doc.document_url,
                    "last_updated_date": doc.last_updated_date,
                    "pdf_path": str(ref.uri),
                }
            )
    return _dedupe_superseded(rows, warnings), _sorted_warnings(warnings)


def read_pdf_text(path: Path) -> str:
    """Extract the text layer of one PDF (newline-joined pages).

    An empty owner password is accepted: EMA publishes some product-information files with an
    encryption dictionary that carries no user password, which pypdf refuses to read until
    ``decrypt`` is called. A password-protected document is an error the caller records.
    """
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_annex_i_text(path: Path) -> str:
    """The text layer up to and including the page that ends Annex I (else the whole document).

    :func:`cut_sections` keeps only the text between the FIRST ``ANNEX I`` line and the first
    Annex II-V marker after it, so decoding stops at the page that carries that marker: the
    labelling and package-leaflet pages after it can never reach a row. Markers are line-scoped
    and pages are newline-joined, so a per-page search finds exactly the markers a search over the
    joined text finds (private-use glyphs are blanked first, as :func:`cut_sections` does). A
    document without an ``ANNEX I`` line is read in full, because its fallback cut needs it.
    Corruption only on an unvisited leaflet page is no longer a whole-document failure: valid
    Annex I evidence survives, whereas the old full read could emit only ``pdf_unreadable``.
    """
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    pages: list[str] = []
    in_annex_i = False
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
        visible = _PRIVATE_USE.sub(" ", page_text)
        search_from = 0
        if not in_annex_i:
            start = _ANNEX_I.search(visible)
            if start is None:
                continue
            in_annex_i, search_from = True, start.end()
        if _NEXT_ANNEX.search(visible, search_from) is not None:
            break
    return "\n".join(pages)


def _read_one(path: str) -> tuple[str, str | None]:
    """``(annex_i_text, None)`` or ``("", "<ExcType>: <message>")`` for an unreadable PDF.

    Module-level (picklable) so a spawn worker can run it; only the expected unreadable-document
    failures are captured, anything else propagates and fails the task loudly.
    """
    try:
        return read_annex_i_text(Path(path)), None
    except (OSError, PdfReadError, ValueError) as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _read_texts(paths: list[Path], workers: int) -> list[tuple[str, str | None]]:
    """Read every PDF (see :func:`_read_one`), returning results in ``paths`` order.

    Parallel runs use a SPAWN pool (the Airflow worker is multi-threaded, so fork is unsafe) and
    dispatch the largest files first, one per task, so the 300-600 page documents do not land on
    the tail of the schedule.
    """
    if workers <= 1 or len(paths) <= 1:
        return [_read_one(str(path)) for path in paths]
    order = sorted(range(len(paths)), key=lambda index: (-_file_size(paths[index]), index))
    results: list[tuple[str, str | None]] = [("", None)] * len(paths)
    # Airflow's CLI is __main__ without a module spec; spawn must not execute that CLI again.
    from dakp_pipeline.assertions.ner_dispatch import _spawn_safe_main

    with _spawn_safe_main(), ProcessPoolExecutor(max_workers=min(workers, len(paths)), mp_context=mp.get_context("spawn")) as pool:
        for index, result in zip(order, pool.map(_read_one, [str(paths[index]) for index in order], chunksize=1), strict=True):
            results[index] = result
    return results


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0  # unreadable is reported by _read_one; order only needs a total key


def _workers(ctx: TaskContext) -> int:
    """Use run ``threads`` (else host cores), capped at 32 to bound PDF memory and imports."""
    value = ctx.params.get("threads")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return min(value, _MAX_PARSE_WORKERS)
    return min(os.cpu_count() or 1, _MAX_PARSE_WORKERS)


def cut_sections(text: str) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str]]]:
    """Cut the Annex I SmPC sections we mine out of one document's text.

    Returns ``{section_kind: (title, body)}`` plus ``(code, message)`` warnings. The body is
    whitespace-collapsed (the text layer carries NBSP, soft hyphens and layout newlines) and stops
    at the next numbered heading, so a missing intermediate section cannot leak into the next one.
    """
    warnings: list[tuple[str, str]] = []
    body, annex_warning = _annex_i_text(_PAGE_NUMBER_LINE.sub("", _PRIVATE_USE.sub(" ", text)))
    if annex_warning is not None:
        warnings.append(annex_warning)

    # (number, title, heading start, body start): a section's body runs from the end of its own
    # heading line to the START of the next heading, so no heading text ever leaks into a body.
    headings = [(match.group(1), _WS.sub(" ", match.group(2)).strip(), match.start(0), match.end(0)) for match in _HEADING.finditer(body)]
    if not headings:
        warnings.append(("no_headings", "no numbered QRD headings found in Annex I"))
        return {}, warnings

    sections: dict[str, tuple[str, str]] = {}
    for index, (number, title, _, start) in enumerate(headings):
        kind = _section_kind(number, title)
        if kind is None or kind in sections:
            if kind is not None:
                warnings.append(("duplicate_section", f"section {number} ({title}) appears more than once; kept the first"))
            continue
        end = headings[index + 1][2] if index + 1 < len(headings) else len(body)
        section_text = _WS.sub(" ", body[start:end]).strip()
        if not section_text:
            warnings.append(("empty_section", f"section {number} ({kind}) has no text"))
            continue
        sections[kind] = (title, section_text)

    for kind, (number, _) in _MINED_SECTIONS.items():
        if kind not in sections:
            warnings.append(("missing_section", f"section {number} ({kind}) was not found in Annex I"))
    return sections, warnings


def _section_kind(number: str, title: str) -> str | None:
    """The mined section kind for a heading, or ``None`` when it is not one we mine."""
    entry = _BY_NUMBER.get(number)
    if entry is None:
        return None
    kind, prefix = entry
    return kind if title.lower().startswith(prefix) else None


def _annex_i_text(text: str) -> tuple[str, tuple[str, str] | None]:
    """The Annex I (SmPC) slice of a product-information document, plus a warning when unmarked."""
    start_match = _ANNEX_I.search(text)
    if start_match is None:
        # Older/odd layouts sometimes omit the marker: fall back to the whole document but stop at
        # the first later annex, and say so, because the leaflet wording must never be mined.
        boundary = _NEXT_ANNEX.search(text)
        return (text[: boundary.start()] if boundary else text), ("no_annex_i", "no ANNEX I marker; cut at the first later annex instead")
    boundary = _NEXT_ANNEX.search(text, start_match.end())
    return text[start_match.end() : boundary.start() if boundary else len(text)], None


def _dedupe_superseded(rows: list[dict[str, str]], warnings: list[dict[str, str]]) -> list[dict[str, str]]:
    """One row per (product number, section): the newest document wins, the rest are reported."""
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["ema_product_number"], row["section_kind"])
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        superseded, kept = _newest_wins(current, row)
        best[key] = kept
        warnings.append(
            _warning(
                kept["ema_product_number"],
                "superseded_document",
                f"{kept['medicine_name']}: kept {kept['document_url']} over {superseded['document_url']}",
            )
        )
    return [best[key] for key in sorted(best, key=lambda item: (item[0], _SECTION_ORDER.index(item[1])))]


def _newest_wins(left: dict[str, str], right: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(superseded, kept)`` by ``last_updated_date`` then ``document_url`` (deterministic)."""
    left_key = (left["last_updated_date"], left["document_url"])
    right_key = (right["last_updated_date"], right["document_url"])
    return (left, right) if right_key >= left_key else (right, left)


def _warning(ema_product_number: str, code: str, message: str) -> dict[str, str]:
    return {"ema_product_number": ema_product_number, "code": code, "message": message, "count": "1"}


def _sorted_warnings(warnings: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(warnings, key=lambda row: (row["ema_product_number"], row["code"], row["message"]))


def _to_frame(rows: list[dict[str, str]], columns: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({column: [] for column in columns}, schema=dict.fromkeys(columns, pl.Utf8), orient="col")
    return pl.DataFrame(rows, schema=dict.fromkeys(columns, pl.Utf8), orient="row")


def _write_parquet(
    rows: list[dict[str, str]],
    columns: list[str],
    out: Path,
    store: ArtifactStore,
    operation: OperationBlock,
    fingerprint: str,
    warnings: int,
    inputs: list[str],
) -> ArtifactRef:
    frame = _to_frame(rows, columns)
    rows_written = schemas.write_parquet(frame, out)
    return store.register(
        out,
        media_type=schemas.PARQUET_MEDIA_TYPE,
        rows=rows_written,
        schema_fingerprint=fingerprint,
        inputs=inputs,
        operation=operation,
        table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint, warnings=warnings),
    )


extract = EmaSmpcExtractor().extract

__all__ = [
    "SMPC_SECTIONS_COLUMNS",
    "SMPC_WARNINGS_COLUMNS",
    "EmaSmpcExtractor",
    "cut_sections",
    "extract",
    "parse_documents",
    "read_annex_i_text",
    "read_pdf_text",
]
