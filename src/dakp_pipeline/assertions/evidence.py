"""Shared evidence helpers for the assertion-aggregation stage.

Pure, testable building blocks used by every assertion shaper:

* **NDA join-key normalization** — FAERS stores zero-stripped application numbers
  (``12345``) while DailyMed/Drugs@FDA store the padded FDA form (``012345``); every
  cross-source NDA join goes through :func:`normalize_nda`.
* **Provenance column assembly** — deduplicated, deterministically sorted, pipe-joined
  lists (the Translator list-encoding convention) for ``FDA_regulatory_approvals``,
  ``supporting_spl_sets``, ``supporting_spl_documents``, and ``supporting_spl_evidence`` —
  the ``dailymed:<spl_set_id>`` CURIEs of the backing SPL sets that Tablassert encodes as
  Biolink ``has_evidence`` (the legacy DAKP KG evidence form).
* **SPL-support joining** — index DailyMed SPL approvals/ingredients/sections and
  Drugs@FDA application→ingredient lookups so shapers can ask "which SPL sets support
  this approval?" without re-scanning frames.
* **FDA application-number expansion** — :class:`FDAApprovalIndex` turns the prefix-stripped
  number a source records (FAERS ``125514``) back into the FDA form every consumer expects
  (``BLA125514``), the ``FDA_regulatory_approvals`` edge values.
* **Table resolution + output writing** — find interim parquet tables among
  ``inputs`` and register the uncompressed assertion TSV.

Text-first by design: these helpers surface source text and only populate CURIEs where a
source already provides an id (DailyMed UNII). Canonical CURIE
mapping is a later milestone.
"""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_bytes
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock, read_manifest
from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.paths import Workdir

# LOINC section codes DAKP consumes for SPL support (mirrors extract.spl_xml.SECTION_CODE_NAMES).
INDICATION_LOINC = "34067-9"  # indications_and_usage
CONTRAINDICATION_LOINC = "34070-3"  # contraindications
BOXED_WARNING_LOINC = "34066-1"  # boxed_warning
#: Warnings sections feeding contraindication Pass 3: the combined warnings-and-precautions
#: section on modern labels plus the legacy pre-2009 split sections.
WARNINGS_LOINCS = frozenset({"43685-7", "34071-1", "42232-9"})  # warnings_and_precautions / warnings / precautions

_NDA_DIGITS_RE = re.compile(r"[^0-9]+")
#: An FDA application number as the sources write it: an optional letter prefix naming the
#: application TYPE (``NDA``/``ANDA``/``BLA``) immediately followed by the number. Ported from
#: the legacy ``split_prefix_number`` (``ref/legacy/bin/uselist2kg.py``), widened so a bare
#: number (FAERS) parses too.
_APPLICATION_NUMBER_RE = re.compile(r"([A-Za-z]*)\s*(\d+)")
#: NCI Thesaurus application-type codes as they appear in the SPL approval ``<code>`` element
#: (``codeSystem 2.16.840.1.113883.3.26.1.1``). DailyMed writes the human prefix into the
#: approval id itself, so this is only the fallback for a label whose id carries none.
_NCI_APPLICATION_TYPES = {
    "C73584": "ANDA",  # abbreviated new drug application
    "C73585": "BLA",  # biologics license application
    "C73594": "NDA",  # new drug application
    "C73605": "NDA",  # NDA authorized generic
}
_FAERS_QUARTER_RE = re.compile(r"^(?:(\d{4})|(\d{2}))Q([1-4])$", re.IGNORECASE)
_FAERS_FILENAME_RE = re.compile(r"faers_ascii_(\d{2}|\d{4})q([1-4])\.zip", re.IGNORECASE)
_PIPE_UNSAFE_RE = re.compile(r"[|\t\r\n]")
_PIPE_UNSAFE_RUN_RE = re.compile(r"[|\s]+")

#: Canonical FDA FAERS quarterly ZIP fallback. The FDA listing page is only a discovery
#: page; the immutable quarter files live under this download base.
FAERS_DOWNLOAD_BASE = "https://fis.fda.gov/content/Exports"


# --- NDA join-key normalization -------------------------------------------------


def normalize_nda(value: Any) -> str:
    """Normalize an FDA application number to a stable join key: digits, no leading zeros.

    FAERS reports ``12345`` while DailyMed/Drugs@FDA report ``012345``; both normalize to
    ``"12345"`` so cross-source NDA joins line up. Non-numeric input and empty values
    normalize to ``""`` (never a join key).
    """
    digits = _NDA_DIGITS_RE.sub("", "" if value is None else str(value))
    return digits.lstrip("0")


def application_type_prefix(approval_type: Any) -> str:
    """The letter prefix an SPL approval type column implies (``""`` when it implies none).

    DailyMed writes an NCI Thesaurus code (``C73584``) into the approval ``<code>`` element;
    the simplified fixture (and any source that curates the column) writes the letter prefix
    itself (``NDA``). Both are accepted; anything else contributes no prefix.
    """
    code = ("" if approval_type is None else str(approval_type)).strip().upper()
    if code in _NCI_APPLICATION_TYPES:
        return _NCI_APPLICATION_TYPES[code]
    return code if code.isalpha() else ""


def split_application_number(value: Any) -> tuple[str, str]:
    """Split an FDA application number into ``(type prefix, digits)``.

    ``"BLA125514"`` -> ``("BLA", "125514")``; ``"bla 0042"`` -> ``("BLA", "0042")``; a bare
    FAERS number ``"125514"`` -> ``("", "125514")``; anything with no digits -> ``("", "")``.
    Leading zeros are PRESERVED: they are part of the FDA display form (``NDA017977``), unlike
    the :func:`normalize_nda` join key.
    """
    match = _APPLICATION_NUMBER_RE.search("" if value is None else str(value))
    if match is None:
        return "", ""
    return match.group(1).upper(), match.group(2)


# --- FDA application-number display forms ---------------------------------------


@dataclass(frozen=True)
class FDAApprovalIndex:
    """Normalized application number -> the display forms ``<type><number>`` it is known by.

    The truncated-approval fix. FAERS records application numbers with the type prefix AND the
    leading zeros stripped (``125514``), which is not an identifier anyone can resolve: the FDA
    form is ``BLA125514``. Legacy DAKP rebuilt the prefix from a number -> prefix map
    (``ref/legacy/bin/uselist2kg.py``); this index is the same idea with Drugs@FDA as the
    authoritative source, keyed by :func:`normalize_nda` so FAERS (stripped), DailyMed (padded),
    and Drugs@FDA (padded) all hit the same entry.
    """

    displays: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def expand(self, value: Any) -> list[str]:
        """Return the FDA display forms for one raw application number.

        Falls back to the value's own ``<prefix><digits>`` when the number is in no FDA
        register — an unexpandable number is still real provenance, so it is emitted as it was
        recorded rather than dropped (legacy dropped the whole edge instead).
        """
        norm = normalize_nda(value)
        if not norm:
            return []
        known = self.displays.get(norm)
        if known:
            return list(known)
        prefix, digits = split_application_number(value)
        return [f"{prefix}{digits}"] if digits else []

    def expand_all(self, values: Iterable[Any]) -> list[str]:
        """Sorted unique display forms for many raw application numbers."""
        return merge_unique(display for value in values for display in self.expand(value))


def _index_display(displays: dict[str, list[str]], number: Any, prefix: str, *, authoritative: bool) -> None:
    """Record one ``<prefix><number>`` display form under its normalized join key.

    Drugs@FDA is authoritative: its entry replaces anything DailyMed contributed for that
    number. DailyMed entries accumulate, because a number genuinely can carry more than one
    prefix across labels (Biolink's own example: ranitidine's ``ANADA200536``/``ANDA200536``);
    legacy emitted every prefix it saw, and so does this.
    """
    norm = normalize_nda(number)
    _prefix, digits = split_application_number(number)
    if not norm or not digits or not prefix:
        return
    display = f"{prefix}{digits}"
    if authoritative:
        displays[norm] = [display]
    elif display not in displays.setdefault(norm, []):
        displays[norm].append(display)


def build_fda_approval_index(inputs: Iterable[ArtifactRef]) -> FDAApprovalIndex:
    """Index every FDA application number DAKP can see, with its ``<type><number>`` display form.

    Two registers, in precedence order:

    * **Drugs@FDA** (``applications.parquet``, then ``products.parquet``) — the authoritative
      FDA application register: ``appl_type`` + ``appl_no`` are separate, curated columns, and
      in a full production build it covers ~94% of the distinct application numbers FAERS
      reports with no number carrying two types.
    * **DailyMed SPL** (``spl_approvals.parquet``) — the label's own ``approval/id/@extension``,
      which already embeds the prefix (``ANDA089160``). Covers ~26% of FAERS numbers, and 25 of
      them carry conflicting prefixes (label typos: ``BN``, ``KGX``, ``NAD``), so it only fills
      gaps Drugs@FDA leaves. The SPL ``<code>`` element is an NCI Thesaurus code
      (``C73584``), NOT a letter prefix — :data:`_NCI_APPLICATION_TYPES` maps it for the rare
      label whose id has no prefix of its own.
    """
    displays: dict[str, list[str]] = {}
    approvals = find_table(inputs, "spl_approvals.parquet")
    if approvals is not None:
        for rec in approvals.iter_rows(named=True):
            raw = rec.get("approval_id") or rec.get("approval_code")
            prefix, _digits = split_application_number(raw)
            _index_display(displays, raw, prefix or application_type_prefix(rec.get("approval_type")), authoritative=False)
    for table in ("applications.parquet", "products.parquet"):
        frame = find_table(inputs, table)
        if frame is None or "appl_type" not in frame.columns:
            continue
        for rec in frame.iter_rows(named=True):
            number = rec.get("appl_no") or rec.get("appl_no_raw") or rec.get("appl_no_stripped")
            _index_display(displays, number, str(rec.get("appl_type") or "").strip().upper(), authoritative=True)
    index = FDAApprovalIndex({norm: tuple(sorted(set(forms))) for norm, forms in displays.items()})
    stats(logger, "fda_approval_index", application_numbers=len(index.displays))
    return index


# --- provenance column assembly -------------------------------------------------


def faers_quarter_url(quarter: Any, *, base_url: str = FAERS_DOWNLOAD_BASE) -> str:
    """Return the exact canonical FDA FAERS quarterly ZIP URL for ``quarter``.

    Accepted labels are ``24Q3``/``24q3`` and ``2024Q3``/``2024q3``. Two-digit years are
    interpreted as 20xx, matching the FAERS quarter labels produced by the extractor. The
    returned fallback is therefore ``.../faers_ascii_2024q3.zip``; callers should prefer the
    URL recorded in the input artifact manifest when one is available.
    """
    text = "" if quarter is None else str(quarter).strip()
    match = _FAERS_QUARTER_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid FAERS quarter {quarter!r}; expected YYQn or YYYYQn")
    year = int(match.group(1) or ("20" + match.group(2)))
    qtr = match.group(3)
    return f"{base_url.rstrip('/')}/faers_ascii_{year:04d}q{qtr}.zip"


def source_manifest_url(ref: ArtifactRef) -> str:
    """Return the recorded upstream URL for one artifact, or ``""`` when unavailable."""
    manifest_path = ref.manifest
    if manifest_path is None or not manifest_path.exists():
        return ""
    try:
        manifest = read_manifest(manifest_path)
    except Exception as exc:
        logger.warning("provenance: unreadable manifest {} ({})", manifest_path, exc)
        return ""
    return (manifest.source.url or "").strip()


def source_urls(inputs: Iterable[ArtifactRef]) -> list[str]:
    """Return sorted unique source URLs recorded by the input artifact manifests."""
    return merge_unique(source_manifest_url(ref) for ref in inputs)


def faers_quarter_urls(inputs: Iterable[ArtifactRef]) -> dict[str, str]:
    """Map canonical FAERS quarter labels to manifest URLs, with official URL fallback.

    Raw downloaded refs normally carry the exact FDA URL in ``SourceBlock.url``. Fixture refs
    and older manifests may not, so a filename/path quarter is resolved through
    :func:`faers_quarter_url` rather than producing a blank provenance value.
    """
    result: dict[str, str] = {}
    for ref in inputs:
        candidates = [source_manifest_url(ref), str(ref.uri)]
        match = next((m for value in candidates if (m := _FAERS_FILENAME_RE.search(value))), None)
        if match is None:
            continue
        year, qtr = match.group(1), match.group(2)
        quarter = f"{year[-2:]}Q{qtr}"
        url = source_manifest_url(ref) or faers_quarter_url(quarter)
        result[quarter] = url
    return dict(sorted(result.items()))


def faers_record_url(quarter: Any, quarter_urls: dict[str, str] | None = None) -> str:
    """Return the public FDA source URL for a FAERS record's quarter."""
    label = "" if quarter is None else str(quarter).strip().upper()
    if quarter_urls and label in quarter_urls:
        return quarter_urls[label]
    return faers_quarter_url(label)


def edge_evidence_pipe(*identifier_lists: Iterable[Any]) -> str:
    """Encode the identifier-only union used for final Biolink ``has_evidence``."""
    return sorted_pipe(value for values in identifier_lists for value in values)


def _validate_pipe_safe(value: str, label: str = "provenance value") -> str:
    """Reject values that cannot be represented losslessly in a pipe-encoded TSV cell."""
    if _PIPE_UNSAFE_RE.search(value):
        raise ValueError(f"{label} contains a pipe/tab/newline delimiter: {value!r}")
    return value


def pipe_safe_text(value: Any) -> str:
    """Collapse pipe/whitespace runs in free-form text to single spaces and strip it.

    Identifier provenance must stay pipe-safe verbatim (:func:`_validate_pipe_safe` rejects
    offenders), but free-form display text mined from label prose legitimately contains ``|``
    bullets and line breaks — sanitize it before it enters a pipe-encoded TSV cell.
    """
    return _PIPE_UNSAFE_RUN_RE.sub(" ", "" if value is None else str(value)).strip()


#: DailyMed label page URL every SPL set evidence value links to (``<base><spl_set_id>``).
DAILYMED_SET_URL_BASE = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="

#: CURIE prefix of the legacy DAKP KG SPL-set evidence form (``dailymed:<spl_set_id>``) — the
#: shape every ``has_evidence`` value is emitted in.
DAILYMED_SET_CURIE_PREFIX = "dailymed:"


def dailymed_set_url(set_id: Any) -> str:
    """Return the DailyMed label URL used for one SPL set evidence value (idempotent)."""
    value = "" if set_id is None else str(set_id).strip()
    if not value or value.startswith(DAILYMED_SET_URL_BASE):
        return value
    return f"{DAILYMED_SET_URL_BASE}{value}"


def dailymed_document_url(document_id: Any) -> str:
    """Return the DailyMed URL for one SPL document evidence value (``<set_id>#<loinc>``).

    The fragment keeps the LOINC section code so the section identity survives in the link
    (DailyMed ignores unknown fragments and loads the label page). Bare set ids without a
    ``#`` fragment get the plain set URL. Idempotent.
    """
    value = "" if document_id is None else str(document_id).strip()
    if not value or value.startswith(DAILYMED_SET_URL_BASE):
        return value
    set_id, _, fragment = value.partition("#")
    url = f"{DAILYMED_SET_URL_BASE}{set_id}"
    return f"{url}#{fragment}" if fragment else url


def dailymed_set_curie(value: Any) -> str:
    """Return the ``dailymed:<spl_set_id>`` CURIE for one SPL evidence value (idempotent).

    Accepts a bare set id, a document id (``<set_id>#<loinc>`` — the set part forms the CURIE,
    ``has_evidence`` is set-granular), or an already-prefixed CURIE / legacy label URL (both
    pass through to the same CURIE). Empty values stay empty.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if text.startswith(DAILYMED_SET_CURIE_PREFIX):
        return text
    if text.startswith(DAILYMED_SET_URL_BASE):
        text = text.removeprefix(DAILYMED_SET_URL_BASE)
    set_id = text.partition("#")[0]
    return f"{DAILYMED_SET_CURIE_PREFIX}{set_id}"


def merge_unique(*value_lists: Iterable[Any]) -> list[str]:
    """Flatten, stringify, drop empties, and return a deterministically sorted unique list."""
    seen: set[str] = set()
    for values in value_lists:
        for value in values:
            text = "" if value is None else str(value).strip()
            if text:
                seen.add(_validate_pipe_safe(text))
    return sorted(seen)


def sorted_pipe(values: Iterable[Any]) -> str:
    """Deduplicated, sorted, ``|``-joined encoding of ``values`` (Translator list convention)."""
    return "|".join(merge_unique(values))


def spl_evidence_pipe(sets: Iterable[Any], documents: Iterable[Any]) -> str:
    """Pipe-joined ``dailymed:<spl_set_id>`` CURIEs for every SPL set backing an assertion.

    This is the column Tablassert encodes as the Biolink ``has_evidence`` slot, in the legacy
    DAKP KG evidence form: sorted, deduplicated set-granular CURIEs (downstream
    translator-ingests treats them as ``publications``). Document ids (``<set_id>#<loinc>``)
    reduce to their set CURIE, so rows tracking section-level provenance contribute the same
    set evidence; the section granularity stays visible in the UN-annotated
    ``supporting_spl_documents`` debug column (one column reaches the edge because a section
    can declare ``has_evidence`` exactly once — annotation names must be unique per table).
    """
    return sorted_pipe([*(dailymed_set_curie(value) for value in sets), *(dailymed_set_curie(value) for value in documents)])


# --- input table resolution -----------------------------------------------------

_TableIndex = dict[str, list[ArtifactRef]]


def _table_index(inputs: Iterable[ArtifactRef]) -> _TableIndex:
    """Index refs by basename, preserving input order for duplicate names."""
    index: _TableIndex = {}
    for ref in inputs:
        index.setdefault(ref.uri.name, []).append(ref)
    return index


def _read_indexed_table(index: _TableIndex, filename: str) -> pl.DataFrame | None:
    """Read the first readable table matching ``filename`` from a prebuilt input index."""
    for ref in index.get(filename, []):
        try:
            return schemas.read_table(ref.uri)
        except Exception as exc:
            logger.warning("skipping unreadable input {} ({})", ref.uri, exc)
    return None


def find_table(inputs: Iterable[ArtifactRef], filename: str) -> pl.DataFrame | None:
    """Read the first parquet/tsv input whose ``uri.name`` equals ``filename``."""
    return _read_indexed_table(_table_index(inputs), filename)


def find_faers_cases(inputs: Iterable[ArtifactRef], columns: tuple[str, ...] | None = None) -> pl.DataFrame | None:
    """Resolve the FAERS case table: the global ``cases.parquet`` (preferred) or ``faers_cases.tsv``.

    The extractor emits a global ``cases.parquet`` first plus per-quarter ``cases.parquet``
    partitions; the global one (no ``quarter=`` in its path) is the deduped case join the
    shapers want. Falls back to the public ``faers_cases.tsv`` projection. Returns ``None``
    when no FAERS case table is present (in which case the approved-treats shaper degrades to
    its DailyMed fallback candidate path).

    ``columns`` optionally projects the read to a subset of columns — the production case
    table is tens of millions of rows wide, so shapers that need only a few columns
    (observed-uses: drugname/indication/primaryid; approved-treats: nda/nda_raw/indication/
    ingredient/drugname) skip the rest (a cheap schema peek decides which requested columns
    actually exist, so a primaryid-less table still loads).
    """
    refs = list(inputs)
    global_cases: ArtifactRef | None = None
    any_cases: ArtifactRef | None = None
    tsv_cases: ArtifactRef | None = None
    for ref in refs:
        if ref.uri.name == "cases.parquet":
            any_cases = any_cases or ref
            if "quarter=" not in str(ref.uri):
                global_cases = global_cases or ref
        elif ref.uri.name == "faers_cases.tsv":
            tsv_cases = tsv_cases or ref

    chosen = global_cases or any_cases or tsv_cases
    if chosen is None:
        logger.warning("faers_cases: no FAERS case table among the inputs")
        return None
    frame = _read_case_table(chosen.uri, columns)
    if "drugname" not in frame.columns or "indication" not in frame.columns:
        logger.warning("faers_cases: {} lacks drugname/indication columns; treating as absent", chosen.uri)
        return None
    stats(logger, "faers_cases", path=str(chosen.uri), rows=frame.height, projected_columns=",".join(frame.columns))
    return frame


def _read_case_table(path: Path, columns: tuple[str, ...] | None) -> pl.DataFrame:
    """Read the case table, projecting to ``columns`` when given (absent columns are skipped)."""
    if columns is None:
        return schemas.read_table(path)
    available = _schema_columns(path)
    keep = [c for c in columns if c in available]
    if not keep:
        return pl.DataFrame()
    if path.suffix == ".parquet":
        return pl.read_parquet(path, columns=keep)
    return pl.read_csv(path, separator="\t", columns=keep)


def _schema_columns(path: Path) -> set[str]:
    """Column names of a table WITHOUT loading its data (parquet metadata / CSV header scan)."""
    if path.suffix == ".parquet":
        return set(pl.read_parquet_schema(path))
    return set(pl.scan_csv(path, separator="\t").collect_schema().names())


# --- DailyMed SPL support index -------------------------------------------------


@dataclass(frozen=True)
class DailyMedEvidence:
    """Indexed DailyMed SPL evidence for assertion joining.

    All keys are normalized: ``approval_sets``/``approval_display`` are keyed by
    :func:`normalize_nda`; section maps are keyed by ``spl_set_id``.
    """

    approval_sets: dict[str, set[str]] = field(default_factory=dict)  # norm_nda -> {spl_set_id}
    approval_display: dict[str, str] = field(default_factory=dict)  # norm_nda -> display approval id (``<type><number>``, e.g. ``BLA103795``)
    approval_ids_by_set: dict[str, set[str]] = field(default_factory=dict)  # spl_set_id -> display approval ids
    set_ingredient: dict[str, tuple[str, str]] = field(default_factory=dict)  # spl_set_id -> first active (name, unii)
    active_ingredients_by_set: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> all active (name, unii)
    indication_docs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> [(doc_id, text)]
    contraindication_docs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> [(doc_id, text)]
    boxed_warning_docs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> [(doc_id, text)]
    warning_docs: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # set -> [(doc_id, text)] (warnings + precautions)

    def indication_support(self, norm_nda: str) -> tuple[list[str], list[str]]:
        """Sorted ``(spl_set_ids, spl_document_ids)`` supporting an approval via indication sections.

        A set supports the approval when it bears that approval AND has an indications-and-usage
        section. Empty lists when the approval has no supporting indication section.
        """
        sets = [s for s in self.approval_sets.get(norm_nda, set()) if self.indication_docs.get(s)]
        docs = [doc for s in sets for doc, _text in self.indication_docs[s]]
        return merge_unique(sets), merge_unique(docs)

    def approval_ids_for_sets(self, set_ids: Iterable[str]) -> list[str]:
        """Return sorted unique FDA application IDs attached to the supplied SPL sets."""
        return merge_unique(approval_id for set_id in set_ids for approval_id in self.approval_ids_by_set.get(set_id, set()))

    def contraindication_sets_for_drug(self, drug_text: str) -> list[str]:
        """Sorted SPL set ids whose active ingredient matches ``drug_text`` and that have a
        contraindication section (the "first-scope" contraindication-support link)."""
        drug = drug_text.strip().lower()
        if not drug:
            return []
        return merge_unique(
            set_id for set_id, (name, _unii) in self.set_ingredient.items() if name.strip().lower() == drug and self.contraindication_docs.get(set_id)
        )


def build_dailymed_evidence(inputs: Iterable[ArtifactRef]) -> DailyMedEvidence:
    """Index DailyMed SPL approvals, active ingredients, and consumed section types.

    Reads the normalized interim tables (``spl_approvals``/``spl_ingredients``/``spl_sections``)
    emitted by :mod:`dakp_pipeline.extract.spl_xml`, indexing the indication
    (:data:`INDICATION_LOINC`), contraindication (:data:`CONTRAINDICATION_LOINC`), boxed-warning
    (:data:`BOXED_WARNING_LOINC`), and warnings/precautions (:data:`WARNINGS_LOINCS`) sections.
    Missing tables yield empty indexes so the shapers degrade gracefully rather than failing.
    """
    evidence = DailyMedEvidence()
    index = _table_index(inputs)

    approvals = _read_indexed_table(index, "spl_approvals.parquet")
    if approvals is not None:
        for rec in approvals.iter_rows(named=True):
            norm = normalize_nda(rec.get("approval_id") or rec.get("approval_code"))
            set_id = str(rec.get("spl_set_id") or "").strip()
            if not norm or not set_id:
                continue
            evidence.approval_sets.setdefault(norm, set()).add(set_id)
            approval_id = str(rec.get("approval_id") or "").strip()
            approval_code = str(rec.get("approval_code") or "").strip()
            approval_type = str(rec.get("approval_type") or "").strip().upper()
            # Legacy-KG display form: application type + number (``BLA103795``). The SPL writes
            # the type INTO the id (``ANDA089160``), so the id is already the display form; the
            # ``approval_type`` column is the NCI Thesaurus code from the label's ``<code>``
            # element (``C73584``) and must never be concatenated onto the number. It only
            # supplies the prefix for the rare label whose id carries none.
            base = approval_id or approval_code
            prefix, digits = split_application_number(base)
            prefix = prefix or application_type_prefix(approval_type)
            display = f"{prefix}{digits}" if digits else (base or norm)
            evidence.approval_display.setdefault(norm, display)
            evidence.approval_ids_by_set.setdefault(set_id, set()).add(display)

    ingredients = _read_indexed_table(index, "spl_ingredients.parquet")
    if ingredients is not None:
        seen_ingredients: set[tuple[str, str, str]] = set()
        for rec in ingredients.iter_rows(named=True):
            if str(rec.get("role") or "").strip().lower() != "active":
                continue
            set_id = str(rec.get("spl_set_id") or "").strip()
            name = str(rec.get("ingredient_name") or "").strip()
            unii = str(rec.get("ingredient_unii") or "").strip()
            if not set_id or not name:
                continue
            key = (set_id, name.lower(), unii)
            if key in seen_ingredients:
                continue
            seen_ingredients.add(key)
            evidence.active_ingredients_by_set.setdefault(set_id, []).append((name, unii))
            evidence.set_ingredient.setdefault(set_id, (name, unii))
        for pairs in evidence.active_ingredients_by_set.values():
            pairs.sort()

    sections = _read_indexed_table(index, "spl_sections.parquet")
    if sections is not None:
        for rec in sections.iter_rows(named=True):
            set_id = str(rec.get("spl_set_id") or "").strip()
            doc_id = str(rec.get("spl_document_id") or "").strip()
            text = str(rec.get("clean_text") or rec.get("raw_text") or "").strip()
            loinc = str(rec.get("loinc_code") or "").strip()
            if not set_id:
                continue
            if loinc == INDICATION_LOINC:
                evidence.indication_docs.setdefault(set_id, []).append((doc_id, text))
            elif loinc == CONTRAINDICATION_LOINC:
                evidence.contraindication_docs.setdefault(set_id, []).append((doc_id, text))
            elif loinc == BOXED_WARNING_LOINC:
                evidence.boxed_warning_docs.setdefault(set_id, []).append((doc_id, text))
            elif loinc in WARNINGS_LOINCS:
                evidence.warning_docs.setdefault(set_id, []).append((doc_id, text))

    stats(
        logger,
        "dailymed_evidence",
        approvals=len(evidence.approval_sets),
        approval_set_links=len(evidence.approval_ids_by_set),
        sets_with_ingredients=len(evidence.set_ingredient),
        indication_sets=len(evidence.indication_docs),
        contraindication_sets=len(evidence.contraindication_docs),
        boxed_warning_sets=len(evidence.boxed_warning_docs),
        warning_sets=len(evidence.warning_docs),
    )
    return evidence


# --- DailyMed evidence cache (build once per run) -------------------------------

#: Operation name under which the serialized :class:`DailyMedEvidence` is registered.
EVIDENCE_OPERATION = "build_dailymed_evidence"
#: Bump when the evidence structure or builder logic changes (invalidates cached evidence).
EVIDENCE_BUILDER_VERSION = "v2"
#: Synthetic input id folding the builder version into the operation-index key.
_EVIDENCE_VERSION_INPUT = hash_bytes(f"dailymed_evidence_builder:{EVIDENCE_BUILDER_VERSION}".encode())

#: Interim tables ``build_dailymed_evidence`` consumes; their artifact ids key the cache.
_EVIDENCE_TABLES = ("spl_approvals.parquet", "spl_ingredients.parquet", "spl_sections.parquet")


def load_or_build_dailymed_evidence(inputs: Iterable[ArtifactRef], ctx: TaskContext | None = None) -> DailyMedEvidence:
    """The DailyMed evidence index for ``inputs``, served from the store when already built.

    ``build_dailymed_evidence`` scans ``spl_sections.parquet`` row-by-row and is needed by
    BOTH the approved-treats and contraindications shape tasks (separate Airflow tasks, so no
    in-memory sharing). The built structure is therefore persisted as a pickled store artifact
    keyed by the blake3 ids of the consumed DailyMed interim tables plus the builder version
    (:data:`EVIDENCE_BUILDER_VERSION`): the second shape task of a run deserializes instead of
    re-scanning. Without ``ctx`` (or without any consumed table among ``inputs``) this just
    builds — the graceful-degradation contract of the builder is unchanged.
    """
    refs = list(inputs)
    consumed = sorted(ref.blake3 for ref in refs if ref.uri.name in _EVIDENCE_TABLES)
    if ctx is None or not consumed:
        return build_dailymed_evidence(refs)
    key_inputs = [*consumed, _EVIDENCE_VERSION_INPUT]
    store = ArtifactStore(Workdir(ctx.workdir))
    cached = store.find_by_operation(EVIDENCE_OPERATION, key_inputs)
    if cached is not None:
        with cached[0].uri.open("rb") as handle:
            evidence: DailyMedEvidence = pickle.load(handle)
        stats(logger, "dailymed_evidence", cache_hit=True, blake3=cached[0].blake3)
        return evidence

    evidence = build_dailymed_evidence(refs)
    out = Workdir(ctx.workdir).store / "dailymed_evidence.pickle"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(evidence))
    store.register(out, media_type="application/octet-stream", inputs=key_inputs, operation=OperationBlock(name=EVIDENCE_OPERATION))
    stats(logger, "dailymed_evidence", cache_hit=False, path=str(out))
    return evidence


# --- shape-stage "already done" skip ---------------------------------------------


def shape_config_fingerprint(ctx: TaskContext) -> str:
    """A synthetic ``b3:<hex>`` input id capturing everything NON-artifact that shapes outputs.

    Folded into the shape tasks' operation-index keys so a skip only happens when the config
    is unchanged too: the lexical disease map content, the NER backend key material (model id
    + model content hash + backend config fingerprint via the Phase-2
    :func:`~dakp_pipeline.ner.mention_cache.ner_cache_material` helper when a production
    backend is injected), and the contraindication keyword override. The assertion-table
    schema fingerprints are folded in as well: a code change that adds, removes, or reorders
    output columns (e.g. a new evidence column) MUST bust the skip — Tablassert configs are
    generated from the CURRENT schema, so reusing a stale TSV shaped by an older one crashes
    the build on the missing column. Run limits
    (``quarter_limit`` etc.) are deliberately absent — they act on acquisition/extraction,
    whose OUTPUT ids are already among the keyed inputs.
    """
    from dakp_pipeline.ner.mention_cache import ner_cache_material  # lazy: pulls in the NER stack
    from dakp_pipeline.ner.ner import DiseaseNER

    material: dict[str, Any] = {
        "disease_map": ctx.params.get("disease_map") or {},
        "schemas": {table: schemas.schema_fingerprint(columns) for table, columns in schemas.ASSERTION_TABLES.items()},
    }
    ner = ctx.params.get("ner")
    if isinstance(ner, DiseaseNER):
        material["ner"] = ner_cache_material(ner)
    keywords = ctx.params.get("contraindication_keywords")
    if keywords is not None:
        material["contraindication_keywords"] = keywords.pattern if isinstance(keywords, re.Pattern) else str(keywords)
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hash_bytes(canonical.encode("utf-8"))


def shape_operation_inputs(refs: Iterable[ArtifactRef], ctx: TaskContext) -> list[str]:
    """The manifest input ids for a shape task: the input artifact ids + the config fingerprint."""
    return [ref.blake3 for ref in refs] + [shape_config_fingerprint(ctx)]


def cached_shape_outputs(operation: str, refs: Iterable[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef] | None:
    """Prior registered outputs of a shape task whose inputs+config are unchanged, else None.

    ``force`` bypasses the skip (registration still happens, overwriting the entry).
    """
    if ctx.params.get("force"):
        return None
    return ArtifactStore(Workdir(ctx.workdir)).find_by_operation(operation, shape_operation_inputs(list(refs), ctx))


# --- Drugs@FDA application -> ingredient index ----------------------------------


def build_drugsfda_ingredient_map(inputs: Iterable[ArtifactRef]) -> dict[str, set[str]]:
    """Map normalized FDA application number -> {active ingredient} from Drugs@FDA products.

    Confirms an NDA is a real FDA application and yields the approved ingredient(s). Keyed by
    :func:`normalize_nda` so it joins against FAERS (stripped) and DailyMed (padded) NDAs alike.
    """
    mapping: dict[str, set[str]] = {}
    products = find_table(inputs, "products.parquet")
    if products is None:
        return mapping
    for rec in products.iter_rows(named=True):
        norm = normalize_nda(rec.get("appl_no_stripped") or rec.get("appl_no"))
        ingredient = str(rec.get("active_ingredient") or "").strip()
        if norm and ingredient:
            mapping.setdefault(norm, set()).add(ingredient)
    stats(logger, "drugsfda_ingredient_map", ndas=len(mapping))
    return mapping


# --- assertion output registration ----------------------------------------------


def write_assertion_table(
    table: str, rows: list[dict[str, str]], inputs: list[ArtifactRef], ctx: TaskContext, *, operation: str
) -> list[ArtifactRef]:
    """Write ``rows`` as the uncompressed TSV for ``table`` and register it with provenance.

    Columns are fixed by :data:`dakp_pipeline.io.schemas.ASSERTION_TABLES`; row ordering is the
    caller's responsibility (shapers sort deterministically before calling). Returns ``[]`` only
    when the table contract is unknown (cannot happen for the three assertion tables).
    """
    columns = schemas.columns_for(table)
    frame = pl.DataFrame(rows, schema=columns) if rows else pl.DataFrame(schema=columns)
    out = Workdir(ctx.workdir).tabular / f"{table}.tsv"
    rows_written = schemas.write_tsv(frame, out)
    fingerprint = schemas.schema_fingerprint(columns)
    store = ArtifactStore(Workdir(ctx.workdir))
    # The manifest inputs carry the shape-stage config fingerprint as a synthetic final id so
    # the operation-index entry (maintained by ``register``) matches the shape task's skip
    # lookup (see ``shape_operation_inputs`` / ``cached_shape_outputs``).
    input_ids = shape_operation_inputs(inputs, ctx)
    ref = store.register(
        out,
        media_type=schemas.TSV_MEDIA_TYPE,
        rows=rows_written,
        schema_fingerprint=fingerprint,
        inputs=input_ids,
        operation=OperationBlock(name=operation),
        table=TableBlock(rows=rows_written, schema_fingerprint=fingerprint),
    )
    stats(logger, "assertion_table", table=table, rows=rows_written, path=str(out), blake3=ref.blake3, schema_fingerprint=fingerprint)
    return [ref]


__all__ = [
    "BOXED_WARNING_LOINC",
    "CONTRAINDICATION_LOINC",
    "DAILYMED_SET_CURIE_PREFIX",
    "DAILYMED_SET_URL_BASE",
    "EVIDENCE_BUILDER_VERSION",
    "EVIDENCE_OPERATION",
    "FAERS_DOWNLOAD_BASE",
    "INDICATION_LOINC",
    "WARNINGS_LOINCS",
    "DailyMedEvidence",
    "build_dailymed_evidence",
    "build_drugsfda_ingredient_map",
    "cached_shape_outputs",
    "dailymed_document_url",
    "dailymed_set_curie",
    "dailymed_set_url",
    "edge_evidence_pipe",
    "faers_quarter_url",
    "faers_quarter_urls",
    "faers_record_url",
    "find_faers_cases",
    "find_table",
    "load_or_build_dailymed_evidence",
    "merge_unique",
    "normalize_nda",
    "pipe_safe_text",
    "shape_config_fingerprint",
    "shape_operation_inputs",
    "sorted_pipe",
    "source_manifest_url",
    "source_urls",
    "spl_evidence_pipe",
    "write_assertion_table",
]
