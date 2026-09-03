"""FAERS ASCII extraction.

Parses each quarter's ``$``-delimited FAERS ASCII files into normalized parquet tables
(``DEMO``, ``DRUG``, ``INDI``, ``REAC``, ``RPSR``, ``DELETE``) and then builds a per-quarter
case-level join (:data:`_CASE_COLUMNS`) honoring:

* **DELETE filtering** — primaryids listed in the ``DELETE`` table are dropped from every
  joined table (ported from ``listCases.pl`` ``readDELETE``).
* **caseid dedup across quarters (most-recent-wins)** — when the same ``caseid`` appears in
  multiple quarters, only the most-recent quarter's rows survive; older rows move to the
  dedup audit table (ported from ``listCases.pl`` ``%prevSeenCase``).

Join semantics are ported from ``ref/legacy/FAERS/bin/listCases.pl`` (case rows are driven by
``INDI`` joined to ``DRUG`` on ``(primaryid, drug_seq == indi_drug_seq)``, with ``DEMO``
reporter metadata, ``RPSR`` source, and ``REAC`` reactions ``$``-joined per case).

Parsing robustness for real FDA ASCII:

* ``$`` delimiter with a trailing ``$`` on every line (produces a trailing empty column,
  dropped after read).
* CRLF line endings (Polars strips ``\\r``).
* UPPERCASE headers normalized to lowercase (modern FAERS uses uppercase; the legacy Perl
  lowercased the header line).
* ``primaryid`` / legacy ``isr`` column resolution (pre-2014 FAERS used ``isr``).

Outputs (under ``interim/faers/``):

* ``quarter=<Q>/<family>.parquet`` — faithful normalized tables, partitioned by quarter.
* ``quarter=<Q>/cases.parquet`` — per-quarter case join (DELETE-filtered, intra-quarter
  row-deduped).
* ``cases.parquet`` — global deduped case table (returned first so downstream shapers find it).
* ``faers_cases.tsv`` — uncompressed public contract (``FAERS_CASES_COLUMNS``) for the
  Tablassert source-section handoff.
* ``delete_audit.parquet`` / ``dedup_audit.parquet`` — DELETE and cross-quarter-dedup audits.

The global ``cases.parquet`` ref is returned **first** so the shapers' filename-based
:func:`dakp_pipeline.assertions.evidence.find_faers_cases` resolution (which prefers the
global table — no ``quarter=`` in its path) sees it ahead of the per-quarter partitions.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_bytes
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.paths import Workdir

_DELIMITER = "$"
_FAMILIES = ("DEMO", "DRUG", "INDI", "REAC", "RPSR", "DELETE")
_QUARTER_IN_NAME_RE = re.compile(r"(\d{2})q(\d)", re.IGNORECASE)

# Provenance columns prepended to every normalized table.
_PROVENANCE_COLS = ("quarter", "source_file", "source_record_id")

# Rich per-case schema (interim). The public faers_cases.tsv is a projection of a subset.
_CASE_COLUMNS = [
    "quarter",
    "primaryid",
    "caseid",
    "source",
    "occp_cod",
    "reporter_country",
    "drugname",
    "ingredient",
    "nda",
    "nda_raw",
    "role_cod",
    "drug_seq",
    "indi_drug_seq",
    "indication",
    "effects",
    "source_file",
    "source_record_id",
]

# Value columns used for intra-quarter exact-row dedup (legacy listCases.pl %seenRow).
_DEDUP_SUBSET = ["quarter", "primaryid", "caseid", "source", "occp_cod", "reporter_country", "drugname", "ingredient", "nda", "indication", "effects"]

# Deterministic sort key for stable parquet output (=> stable BLAKE3 across runs).
_CASE_SORT_KEY = ["primaryid", "drug_seq", "indication"]
_AUDIT_SORT_KEY = ["quarter", "primaryid"]

_DELETE_AUDIT_COLUMNS = ["quarter", "primaryid", "caseid", "source_file", "source_record_id"]
_DEDUP_AUDIT_COLUMNS = ["quarter", "primaryid", "caseid", "dedup_key", "winning_quarter", "source_file"]
_WARNINGS_COLUMNS = ["quarter", "family", "code", "message", "count"]


@dataclass(frozen=True)
class _FaersSource:
    """One logical FAERS ASCII file (a loose ``.txt`` or a zip member)."""

    quarter: str
    family: str
    content: bytes
    source_name: str
    source_b3: str


@dataclass
class _Warning:
    quarter: str
    family: str
    code: str
    message: str
    count: int = 1


@dataclass
class _Warnings:
    items: list[_Warning] = field(default_factory=list)

    def add(self, quarter: str, family: str, code: str, message: str, count: int = 1) -> None:
        self.items.append(_Warning(quarter, family, code, message, count))

    @property
    def total(self) -> int:
        return sum(w.count for w in self.items)

    def frame(self) -> pl.DataFrame:
        if not self.items:
            return pl.DataFrame(schema=_WARNINGS_COLUMNS)
        return pl.DataFrame([{k: getattr(w, k) for k in _WARNINGS_COLUMNS} for w in self.items], schema=_WARNINGS_COLUMNS).sort(
            _WARNINGS_COLUMNS[:-1]
        )


class FAERSASCIIExtractor:
    """Parse FAERS ASCII into normalized parquet tables + a per-quarter case join."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        warnings = _Warnings()
        by_quarter_family: dict[str, dict[str, pl.DataFrame]] = defaultdict(dict)
        for source in _iter_faers_sources(inputs):
            frame = _parse_source(source, warnings)
            if frame is not None and not frame.is_empty():
                by_quarter_family[source.quarter][source.family] = frame

        if not by_quarter_family:
            logger.warning("faers extract: no FAERS ASCII sources parsed")
            return []

        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        input_ids = [ref.blake3 for ref in inputs]

        normalized_refs: list[ArtifactRef] = []
        delete_frames: list[pl.DataFrame] = []
        per_quarter_cases: list[pl.DataFrame] = []

        # Process quarters most-recent-first so the global cases sort is intuitive, but the
        # dedup itself is order-independent (max quarter wins per dedup key).
        for quarter in sorted(by_quarter_family, reverse=True):
            families = by_quarter_family[quarter]
            for family, frame in families.items():
                normalized_refs.append(
                    self._write_table(
                        frame,
                        wd.interim / "faers" / f"quarter={quarter}" / f"{family.lower()}.parquet",
                        family.lower(),
                        input_ids,
                        warnings.total,
                        store,
                    )
                )
            delete_frame = families.get("DELETE")
            deleted_pids = _deleted_primaryids(delete_frame)
            if delete_frame is not None and not delete_frame.is_empty():
                delete_frames.append(delete_frame)
            cases = _build_quarter_cases(families, quarter, deleted_pids, warnings)
            normalized_refs.append(
                self._write_table(
                    cases,
                    wd.interim / "faers" / f"quarter={quarter}" / "cases.parquet",
                    "cases",
                    input_ids,
                    warnings.total,
                    store,
                    fingerprint=_CASE_COLUMNS,
                )
            )
            per_quarter_cases.append(cases)

        global_cases, dedup_audit = _reduce_cases(per_quarter_cases)
        delete_audit = _select_delete_audit(delete_frames)

        # Global deduped cases parquet (primary artifact; returned FIRST).
        cases_ref = self._write_table(
            global_cases,
            wd.interim / "faers" / "cases.parquet",
            "cases",
            input_ids,
            warnings.total,
            store,
            fingerprint=_CASE_COLUMNS,
            partitions=max(len(by_quarter_family), 1),
        )
        # Public uncompressed TSV (Tablassert source-section contract).
        tsv_frame = global_cases.select(schemas.FAERS_CASES_COLUMNS)
        schemas.write_tsv(tsv_frame, wd.interim / "faers" / "faers_cases.tsv")
        tsv_ref = store.register(
            wd.interim / "faers" / "faers_cases.tsv",
            media_type=schemas.TSV_MEDIA_TYPE,
            rows=global_cases.height,
            schema_fingerprint=schemas.schema_fingerprint(schemas.FAERS_CASES_COLUMNS),
            inputs=input_ids,
            operation=OperationBlock(name="emit_faers_cases_tsv"),
            table=TableBlock(
                rows=global_cases.height, schema_fingerprint=schemas.schema_fingerprint(schemas.FAERS_CASES_COLUMNS), warnings=warnings.total
            ),
        )
        # Audit tables.
        delete_ref = self._write_table(
            delete_audit,
            wd.interim / "faers" / "delete_audit.parquet",
            "delete_audit",
            input_ids,
            warnings.total,
            store,
            fingerprint=_DELETE_AUDIT_COLUMNS,
        )
        dedup_ref = self._write_table(
            dedup_audit,
            wd.interim / "faers" / "dedup_audit.parquet",
            "dedup_audit",
            input_ids,
            warnings.total,
            store,
            fingerprint=_DEDUP_AUDIT_COLUMNS,
        )
        warnings_ref = self._write_table(
            warnings.frame(), wd.interim / "faers" / "warnings.parquet", "warnings", input_ids, warnings.total, store, fingerprint=_WARNINGS_COLUMNS
        )

        stats(
            logger,
            "extract_faers",
            quarters=len(by_quarter_family),
            cases=global_cases.height,
            deleted=delete_audit.height,
            deduped=dedup_audit.height,
            warnings=warnings.total,
        )
        # cases_ref first so find_faers_cases resolves the global case table ahead of partitions.
        return [cases_ref, tsv_ref, delete_ref, dedup_ref, warnings_ref, *normalized_refs]

    def _write_table(
        self,
        frame: pl.DataFrame,
        path: Path,
        operation_name: str,
        input_ids: list[str],
        warnings_count: int,
        store: ArtifactStore,
        *,
        fingerprint: list[str] | None = None,
        partitions: int | None = None,
    ) -> ArtifactRef:
        """Write a parquet interim table and register it with a BLAKE3 manifest."""
        rows = schemas.write_parquet(frame, path)
        columns = fingerprint if fingerprint is not None else frame.columns
        schema_fp = schemas.schema_fingerprint(columns)
        return store.register(
            path,
            media_type=schemas.PARQUET_MEDIA_TYPE,
            rows=rows,
            schema_fingerprint=schema_fp,
            inputs=input_ids,
            operation=OperationBlock(name=f"extract_faers_{operation_name}"),
            table=TableBlock(rows=rows, partitions=partitions, schema_fingerprint=schema_fp, warnings=warnings_count or None),
        )


def _iter_faers_sources(refs: list[ArtifactRef]) -> Iterator[_FaersSource]:
    """Yield logical FAERS ASCII files from loose ``.txt`` refs or ``.zip`` members."""
    for ref in refs:
        uri = ref.uri
        suffix = uri.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(uri) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    family, quarter = _family_and_quarter(info.filename)
                    if family is None or quarter is None:
                        continue
                    content = zf.read(info)
                    yield _FaersSource(quarter, family, content, info.filename, hash_bytes(content))
        elif suffix == ".txt":
            family, quarter = _family_and_quarter(uri.name)
            if family is None or quarter is None:
                logger.debug("skipping non-FAERS .txt artifact", uri=str(uri))
                continue
            content = uri.read_bytes()
            yield _FaersSource(quarter, family, content, uri.name, hash_bytes(content))
        else:
            logger.debug("skipping non-FAERS artifact", uri=str(uri))


def _family_and_quarter(name: str) -> tuple[str | None, str | None]:
    """Derive ``(family, quarter)`` from a FAERS filename like ``ascii/DEMO24Q3.txt``."""
    base = Path(name).name.upper()
    family = next((fam for fam in _FAMILIES if base.startswith(fam)), None)
    match = _QUARTER_IN_NAME_RE.search(name)
    quarter = f"{match.group(1)}Q{match.group(2)}".upper() if match else None
    return family, quarter


def _parse_source(source: _FaersSource, warnings: _Warnings) -> pl.DataFrame | None:
    """Parse one ``$``-delimited ASCII source into a normalized, Utf8-typed frame.

    Adds ``quarter`` / ``source_file`` / ``source_record_id`` provenance. Returns ``None``
    (after recording a warning) if the file has no usable header or rows.
    """
    if not source.content.strip():
        warnings.add(source.quarter, source.family, "empty_file", "no bytes")
        return None
    try:
        frame = pl.read_csv(
            io.BytesIO(source.content),
            separator=_DELIMITER,
            infer_schema_length=0,  # all-Utf8: preserves NDA leading zeroes, id formatting
            truncate_ragged_lines=True,
            quote_char=None,
        )
    except pl.exceptions.PolarsError as exc:
        warnings.add(source.quarter, source.family, "parse_error", str(exc))
        return None
    if frame.is_empty():
        warnings.add(source.quarter, source.family, "empty_file", "no data rows")
        return None

    # Drop the trailing empty column produced by the trailing "$".
    empty_cols = [c for c in frame.columns if c.strip() == ""]
    if empty_cols:
        frame = frame.drop(empty_cols)
    # Lowercase + strip header names (modern FAERS uses UPPERCASE headers).
    frame = frame.rename({c: c.strip().lower() for c in frame.columns})
    # Trim whitespace from every field (FAERS fields often carry trailing spaces) and cast Utf8.
    for col in frame.columns:
        frame = frame.with_columns(pl.col(col).cast(pl.Utf8).str.strip_chars())

    # Resolve the case identifier column (primaryid, or legacy isr).
    if "isr" in frame.columns and "primaryid" not in frame.columns:
        frame = frame.rename({"isr": "primaryid"})
    elif "primaryid" not in frame.columns:
        warnings.add(source.quarter, source.family, "missing_primaryid", "no primaryid/isr column")
        return None

    frame = _ensure_cols(frame, ("primaryid",))
    frame = frame.with_columns(
        pl.col("primaryid").cast(pl.Utf8).fill_null("").alias("primaryid"),
        pl.lit(source.quarter).alias("quarter"),
        pl.lit(source.source_name).alias("source_file"),
        _source_record_id(frame, source.family, source.source_b3).alias("source_record_id"),
    )
    provenance = list(_PROVENANCE_COLS)
    return frame.select([*provenance, *(c for c in frame.columns if c not in provenance)])


def _ensure_cols(frame: pl.DataFrame, names: tuple[str, ...]) -> pl.DataFrame:
    """Add any missing columns as empty Utf8 (robust to family-specific column variants)."""
    existing = set(frame.columns)
    for name in names:
        if name not in existing:
            frame = frame.with_columns(pl.lit("").cast(pl.Utf8).alias(name))
    return frame


def _empty_frame(columns: tuple[str, ...]) -> pl.DataFrame:
    """An empty all-Utf8 frame with the given columns (for absent-family join fallbacks)."""
    return pl.DataFrame(schema=dict.fromkeys(columns, pl.Utf8))


def _col_or_blank(frame: pl.DataFrame, name: str) -> pl.Expr:
    return pl.col(name) if name in frame.columns else pl.lit("").cast(pl.Utf8)


def _source_record_id(frame: pl.DataFrame, family: str, source_b3: str) -> pl.Expr:
    """Deterministic per-row id: ``<source-hash-prefix>:<primaryid>[:<seq|pt>]``.

    Derived from source hash + source-local identifiers (not mutable row order), per the
    extraction requirements contract.
    """
    short = source_b3.split(":", 1)[1][:12]
    parts: list[pl.Expr] = [pl.lit(short), pl.col("primaryid")]
    if family == "DRUG":
        parts.append(_col_or_blank(frame, "drug_seq"))
    elif family == "INDI":
        parts.append(_col_or_blank(frame, "indi_drug_seq"))
        parts.append(_col_or_blank(frame, "indi_pt"))
    elif family == "REAC":
        parts.append(_col_or_blank(frame, "pt"))
    return pl.concat_str(parts, separator=":")


def _deleted_primaryids(delete_frame: pl.DataFrame | None) -> set[str]:
    """Primaryids marked deleted for a quarter (ported from listCases.pl readDELETE)."""
    if delete_frame is None or delete_frame.is_empty() or "primaryid" not in delete_frame.columns:
        return set()
    return {str(v).strip() for v in delete_frame.get_column("primaryid").to_list() if str(v).strip()}


def _build_quarter_cases(families: dict[str, pl.DataFrame], quarter: str, deleted_pids: set[str], warnings: _Warnings) -> pl.DataFrame:
    """Join one quarter's normalized tables into case rows (INDI-driven, DELETE-filtered)."""
    drug = families.get("DRUG")
    indi = families.get("INDI")
    if drug is None or indi is None or drug.is_empty() or indi.is_empty():
        # Case rows are indication-driven; no DRUG/INDI means no cases for this quarter.
        return pl.DataFrame(schema=_CASE_COLUMNS)

    drug = _ensure_cols(drug, ("primaryid", "drug_seq", "drugname", "role_cod", "nda_num", "prod_ai", "source_file"))
    indi = _ensure_cols(indi, ("primaryid", "indi_drug_seq", "indi_pt"))

    def not_deleted(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(~pl.col("primaryid").is_in(deleted_pids)) if deleted_pids else frame

    drug_lf = (
        not_deleted(drug)
        .select("primaryid", "drug_seq", "drugname", "role_cod", "nda_num", "prod_ai", "source_file")
        .with_columns(pl.col("drug_seq").cast(pl.Utf8).fill_null(""))
        # Synthetic join key so the right-side drug_seq column survives the join.
        .with_columns(pl.concat_str([pl.col("primaryid"), pl.col("drug_seq")], separator="|").alias("_casekey"))
        .lazy()
    )
    indi_lf = (
        not_deleted(indi)
        .select("primaryid", "indi_drug_seq", "indi_pt")
        .with_columns(pl.col("indi_drug_seq").cast(pl.Utf8).fill_null(""))
        .with_columns(pl.concat_str([pl.col("primaryid"), pl.col("indi_drug_seq")], separator="|").alias("_casekey"))
        .lazy()
    )

    joined = indi_lf.join(drug_lf, on="_casekey", how="inner").drop("_casekey")

    # Always left-join DEMO/RPSR/REAC (with empty-frame fallbacks when a family is
    # absent) so the downstream with_columns always finds rpsr_cod/caseid/effects.
    demo = families.get("DEMO")
    demo_df = (
        not_deleted(demo) if (demo is not None and not demo.is_empty()) else _empty_frame(("primaryid", "caseid", "occp_cod", "reporter_country"))
    )
    demo_df = _ensure_cols(demo_df, ("primaryid", "caseid", "occp_cod", "reporter_country"))
    joined = joined.join(demo_df.select("primaryid", "caseid", "occp_cod", "reporter_country").lazy(), on="primaryid", how="left")

    rpsr = families.get("RPSR")
    rpsr_df = not_deleted(rpsr) if (rpsr is not None and not rpsr.is_empty()) else _empty_frame(("primaryid", "rpsr_cod"))
    rpsr_df = _ensure_cols(rpsr_df, ("primaryid", "rpsr_cod"))
    joined = joined.join(rpsr_df.select("primaryid", "rpsr_cod").lazy(), on="primaryid", how="left")

    reac = families.get("REAC")
    if reac is not None and not reac.is_empty():
        reac = _ensure_cols(reac, ("primaryid", "pt"))
        effects = (
            not_deleted(reac)
            .filter(pl.col("pt").fill_null("") != "")
            .group_by("primaryid")
            .agg(pl.col("pt").unique(maintain_order=False).sort().str.join(_DELIMITER).alias("effects"))
            .lazy()
        )
        joined = joined.join(effects, on="primaryid", how="left")
    else:
        joined = joined.with_columns(pl.lit(None).cast(pl.Utf8).alias("effects"))

    cases = joined.with_columns(
        pl.lit(quarter).alias("quarter"),
        pl.col("rpsr_cod").fill_null("").alias("source"),
        pl.col("caseid").fill_null("").alias("caseid"),
        pl.col("occp_cod").fill_null("").alias("occp_cod"),
        pl.col("reporter_country").fill_null("").alias("reporter_country"),
        pl.col("role_cod").fill_null("").alias("role_cod"),
        pl.col("drug_seq").fill_null("").alias("drug_seq"),
        pl.col("prod_ai").fill_null("").alias("ingredient"),
        pl.col("drugname").fill_null("").alias("drugname"),
        pl.col("nda_num").fill_null("").alias("nda_raw"),
        pl.col("indi_pt").fill_null("").alias("indication"),
        pl.col("effects").fill_null("").alias("effects"),
        # nda normalized: digits only, leading zeroes stripped (joins Drugs@FDA ApplNo).
        pl.col("nda_num").fill_null("").str.replace_all(r"\D", "").str.strip_chars_start("0").alias("nda"),
        pl.concat_str([pl.lit(quarter), pl.col("primaryid"), pl.col("drug_seq"), pl.col("indi_pt")], separator=":").alias("source_record_id"),
    )

    cases = cases.select(_CASE_COLUMNS).collect()
    if cases.is_empty():
        return cases
    # Deterministic order before intra-quarter exact-row dedup (legacy %seenRow).
    cases = cases.sort(_CASE_SORT_KEY).unique(subset=_DEDUP_SUBSET, keep="first", maintain_order=True)
    if deleted_pids:
        dropped = drug.filter(pl.col("primaryid").is_in(deleted_pids)).height
        if dropped:
            warnings.add(quarter, "DRUG", "deleted_rows_dropped", f"{dropped} drug rows dropped (DELETE)", dropped)
    return cases


def _reduce_cases(per_quarter: list[pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Concat per-quarter cases and apply cross-quarter caseid dedup (most-recent-wins)."""
    frames = [f for f in per_quarter if not f.is_empty()]
    if not frames:
        empty_audit = pl.DataFrame(schema=_DEDUP_AUDIT_COLUMNS)
        return pl.DataFrame(schema=_CASE_COLUMNS), empty_audit
    all_cases = pl.concat(frames, how="vertical_relaxed")
    # dedup key: caseid when present, else primaryid (matches listCases.pl caseid fallback).
    all_cases = all_cases.with_columns(
        pl.when(pl.col("caseid").cast(pl.Utf8).fill_null("") != "").then(pl.col("caseid")).otherwise(pl.col("primaryid")).alias("_dedup_key")
    )
    winning = all_cases.group_by("_dedup_key").agg(pl.col("quarter").max().alias("_winning_quarter"))
    merged = all_cases.join(winning, on="_dedup_key", how="left")
    kept = merged.filter(pl.col("quarter") == pl.col("_winning_quarter"))
    superseded = merged.filter(pl.col("quarter") != pl.col("_winning_quarter"))
    dedup_audit = (
        superseded.select(
            "quarter",
            "primaryid",
            "caseid",
            pl.col("_dedup_key").alias("dedup_key"),
            pl.col("_winning_quarter").alias("winning_quarter"),
            "source_file",
        )
        if not superseded.is_empty()
        else pl.DataFrame(schema=_DEDUP_AUDIT_COLUMNS)
    )
    kept = kept.drop(["_dedup_key", "_winning_quarter"])
    # kept is never empty here: each dedup-key group keeps its max-quarter row(s). Sorting an
    # empty frame is a no-op, so this is unconditional (no dead empty-guard branch).
    kept = kept.sort(_CASE_SORT_KEY)
    if not dedup_audit.is_empty():
        dedup_audit = dedup_audit.sort(_AUDIT_SORT_KEY)
    return kept.select(_CASE_COLUMNS), dedup_audit


def _select_delete_audit(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(schema=_DELETE_AUDIT_COLUMNS)
    selected = [_ensure_cols(f, tuple(_DELETE_AUDIT_COLUMNS)).select(_DELETE_AUDIT_COLUMNS) for f in frames]
    audit = pl.concat(selected, how="vertical_relaxed")
    return audit.sort(_AUDIT_SORT_KEY) if not audit.is_empty() else audit


extract = FAERSASCIIExtractor().extract

__all__ = ["FAERSASCIIExtractor", "extract"]
