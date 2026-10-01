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
"""

from __future__ import annotations

import re
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

SMPD_SECTIONS_COLUMNS: list[str] = [
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
SMPD_WARNINGS_COLUMNS: list[str] = ["ema_product_number", "code", "message", "count"]

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

        documents = {doc.stem: doc for doc in product_information_documents(load_documents(manifest_ref.uri))}
        rows, warnings = parse_documents(pdf_refs, documents)

        operation = OperationBlock(name=_EVENT)
        input_ids = [ref.blake3 for ref in inputs]
        interim_dir = wd.interim / "ema"
        sections_fp = schemas.schema_fingerprint(SMPD_SECTIONS_COLUMNS)
        warnings_fp = schemas.schema_fingerprint(SMPD_WARNINGS_COLUMNS)
        refs = [
            _write_parquet(
                rows, SMPD_SECTIONS_COLUMNS, interim_dir / "smpc_sections.parquet", store, operation, sections_fp, len(warnings), input_ids
            ),
            _write_parquet(
                warnings, SMPD_WARNINGS_COLUMNS, interim_dir / "smpc_warnings.parquet", store, operation, warnings_fp, len(warnings), input_ids
            ),
        ]
        stats(log, _EVENT, pdfs=len(pdf_refs), rows=len(rows), warnings=len(warnings), outputs=len(refs), sections=",".join(_SECTION_ORDER))
        return refs


def parse_documents(pdf_refs: list[ArtifactRef], documents: dict[str, EparDocument]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse every crawled PDF into section rows + warning rows (pure; deterministic order).

    ``documents`` maps the manifest's document stem to its record, so each PDF can be attributed to
    a medicine, a product number and a source URL. A PDF whose stem is not in the manifest is a
    warning, not a row: without the metadata the section could not be joined to an active substance
    or carry an approval id, and inventing one would be worse than reporting the gap.
    """
    rows: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for ref in pdf_refs:
        stem = ref.uri.stem
        doc = documents.get(stem)
        if doc is None:
            warnings.append(_warning("", "unknown_document", f"PDF {ref.uri.name} is not in the documents report selection"))
            continue
        try:
            text = read_pdf_text(ref.uri)
        except (OSError, PdfReadError, ValueError) as exc:
            warnings.append(_warning(doc.ema_product_number, "pdf_unreadable", f"{doc.medicine_name}: {type(exc).__name__}: {exc}"))
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

__all__ = ["SMPD_SECTIONS_COLUMNS", "SMPD_WARNINGS_COLUMNS", "EmaSmpcExtractor", "cut_sections", "extract", "parse_documents", "read_pdf_text"]
