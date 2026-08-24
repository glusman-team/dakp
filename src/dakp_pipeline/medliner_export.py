"""MEDliNER training-data export: the ``dakp.medliner.export.v1`` bundle.

Hands DAKP's annotation corpus to MEDliNER as a self-describing, versioned, deterministic
bundle — a single directory with exactly three files:

* ``manifest.json`` — schema ``dakp.medliner.export.v1``: schema strings, ``generated_at``,
  per-file blake3 hashes + counts, task/family counts, and the blake3 ids of the consumed
  interim tables.
* ``candidates.ndjson`` — candidate annotation texts in MEDliNER's raw-candidate shape
  (``CandidateText`` field names only, so rows round-trip through MEDliNER without loss),
  deduped on ``(task, normalized text)`` and deterministically sorted so identical inputs
  produce byte-identical output regardless of input row order.
* ``ner_gold.json`` — byte-identical copy of the committed NER gold benchmark
  (``tests/eval/ner_gold.json``), the eval artifact for the trained model.

Candidate derivation (the three MEDliNER corpus families): DailyMed contraindication
sections (LOINC ``34070-3``), DailyMed indications-and-usage sections (LOINC ``34067-9``),
and FAERS observed-use ``indication`` strings from the global ``cases.parquet``.

The stage entry point :func:`export` is Airflow-free and offline-safe (pure table reads +
JSON/JSONL writes; no network, no GLiNER/torch). A missing ``spl_documents.parquet`` or
FAERS case table is a loud error naming the file; a missing/corrupt gold benchmark fails
before ANY bundle file is written. A table that yields zero exportable rows is legal and
produces a valid bundle with 0 counts.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.assertions.evidence import (
    CONTRAINDICATION_LOINC,
    INDICATION_LOINC,
    dailymed_document_url,
    faers_record_url,
    find_faers_cases,
    find_table,
)
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir

#: Bundle schema version; bump on any layout/field change.
SCHEMA_VERSION = "dakp.medliner.export.v1"
#: The MEDliNER row contract the candidate rows honor (``CandidateText`` field names).
CANDIDATE_SCHEMA = "medliner.candidates.v1"
#: The ``schema_version`` the NER gold benchmark must carry to be exported.
GOLD_SCHEMA_VERSION = "dakp.ner.gold.v1"
#: Bundle directory name under ``Workdir.store``.
OUT_DIRNAME = "medliner-export"

MANIFEST_FILENAME = "manifest.json"
CANDIDATES_FILENAME = "candidates.ndjson"
GOLD_FILENAME = "ner_gold.json"

_JSON_MEDIA_TYPE = "application/json"
_NDJSON_MEDIA_TYPE = "application/x-ndjson"

_OPERATION = "export_medliner"
_DAILYMED_TABLE = "spl_documents.parquet"
_FAERS_CASES_TABLE = "cases.parquet"
#: Column projection for the case-table read: all the export needs is the indication string,
#: the primaryid provenance, and the quarter for the FDA source URL.
_FAERS_CASES_COLUMNS: tuple[str, ...] = ("quarter", "primaryid", "drugname", "indication")
_FAERS_QUARTER_PARTITION = re.compile(r"^quarter=(?:\d{2}|\d{4})Q[1-4]$", re.IGNORECASE)


def _is_faers_partition_ref(ref: ArtifactRef) -> bool:
    """Return whether a cases table is under an extractor quarter partition."""
    return bool(_FAERS_QUARTER_PARTITION.fullmatch(ref.uri.parent.name))


_TASK_BY_LOINC: dict[str, str] = {CONTRAINDICATION_LOINC: "contraindication", INDICATION_LOINC: "indication"}
_TASKS: tuple[str, ...] = ("contraindication", "indication")
_FAMILIES: tuple[str, ...] = ("dailymed", "faers")


# --- inputs -------------------------------------------------------------------------


def gold_path() -> Path:
    """The committed NER gold benchmark, resolved from the repo root.

    Same repo-root convention as ``cli._REPO_ROOT`` (``Path(__file__).resolve().parents[2]``).
    Raises ``FileNotFoundError`` when the file is absent — the exporter never runs without
    its eval gold (no silent fallback).
    """
    path = Path(__file__).resolve().parents[2] / "tests" / "eval" / GOLD_FILENAME
    if not path.exists():
        msg = f"medliner_export: NER gold benchmark is missing: {path}"
        raise FileNotFoundError(msg)
    return path


def _normalized(text: str) -> str:
    """Dedupe normalization: lowercase + whitespace collapse (identical to MEDliNER's)."""
    return " ".join(text.split()).lower()


def select_dailymed_rows(table: pl.DataFrame) -> list[dict[str, str]]:
    """DailyMed candidate rows: contraindication (34070-3) + indications-and-usage (34067-9).

    Every ``spl_documents.parquet`` row with an export LOINC and non-blank ``section_text``
    becomes one row; other LOINC sections are skipped. Text is exported verbatim except for
    leading/trailing strip; ``source_uri`` is the DailyMed label URL carrying the LOINC
    fragment (``dailymed_document_url`` form).
    """
    rows: list[dict[str, str]] = []
    for rec in table.iter_rows(named=True):
        loinc = str(rec.get("loinc_code") or "").strip()
        task = _TASK_BY_LOINC.get(loinc)
        if task is None:
            continue
        text = str(rec.get("section_text") or "").strip()
        if not text:
            continue
        document_id = str(rec.get("spl_document_id") or "").strip()
        rows.append(
            {
                "text": text,
                "task": task,
                "source_family": "dailymed",
                "source_document_id": document_id,
                "section": loinc,
                "source_uri": dailymed_document_url(document_id),
            }
        )
    return rows


def select_faers_rows(table: pl.DataFrame) -> list[dict[str, str]]:
    """FAERS candidate rows: one indication row per case row with non-blank ``indication``.

    Duplicate indication strings collapse later in :func:`dedupe_sort` (deterministic
    ``primaryid`` winner); ``source_uri`` is the FDA FAERS quarter URL
    (``faers_record_url`` form).
    """
    rows: list[dict[str, str]] = []
    for rec in table.iter_rows(named=True):
        text = str(rec.get("indication") or "").strip()
        if not text:
            continue
        record_id = str(rec.get("primaryid") or "").strip()
        rows.append(
            {
                "text": text,
                "task": "indication",
                "source_family": "faers",
                "source_record_id": record_id,
                "source_uri": faers_record_url(rec.get("quarter")),
            }
        )
    return rows


# --- dedupe + determinism -------------------------------------------------------------


def _sort_key(row: Mapping[str, str]) -> tuple[str, ...]:
    """Deterministic row order (export contract R5).

    The first four fields ARE the R5 order ``(task, normalized text, source_document_id or
    "", source_record_id or "")``; verbatim text and URI are tie-breakers so even degenerate
    duplicate sets order identically for any input row order (they cannot reorder surviving
    rows, which all differ on ``(task, normalized text)`` after dedupe).
    """
    text = row.get("text", "")
    return (
        row.get("task", ""),
        _normalized(text),
        row.get("source_document_id") or "",
        row.get("source_record_id") or "",
        text,
        row.get("source_uri") or "",
    )


def dedupe_sort(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Dedupe on ``(task, normalized text)`` and sort deterministically (export contract R5).

    The retained row is the first in :func:`_sort_key` order (its verbatim ``text`` and
    provenance survive); output lines follow the same sorted order, so identical input sets
    yield byte-identical ``candidates.ndjson`` regardless of input row order.
    """
    winners: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=_sort_key):
        key = (row.get("task", ""), _normalized(row.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        winners.append(dict(row))
    return winners


# --- gold validation -----------------------------------------------------------------


def _load_gold(path: Path) -> dict[str, Any]:
    """Parse and validate a gold benchmark; loud errors, never a silent fallback.

    ``FileNotFoundError`` when absent; ``ValueError`` when the content is not readable JSON
    or its ``schema_version`` is not ``dakp.ner.gold.v1``.
    """
    if not path.exists():
        msg = f"medliner_export: NER gold benchmark is missing: {path}"
        raise FileNotFoundError(msg)
    try:
        gold = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"medliner_export: NER gold benchmark is not readable JSON ({path}): {exc}"
        raise ValueError(msg) from exc
    if not isinstance(gold, dict):
        msg = f"medliner_export: NER gold benchmark {path} must be a JSON object"
        raise ValueError(msg)
    schema_version = gold.get("schema_version")
    if schema_version != GOLD_SCHEMA_VERSION:
        msg = f"medliner_export: NER gold benchmark {path} has schema_version {schema_version!r}, expected {GOLD_SCHEMA_VERSION!r}"
        raise ValueError(msg)
    annotation_policy = gold.get("annotation_policy")
    if not isinstance(annotation_policy, str) or not annotation_policy.strip():
        msg = f"medliner_export: NER gold benchmark {path} is missing required annotation_policy"
        raise ValueError(msg)
    cases = gold.get("cases")
    if not isinstance(cases, list) or not cases:
        msg = f"medliner_export: NER gold benchmark {path} must contain a non-empty cases list"
        raise ValueError(msg)
    return gold


# --- bundle assembly -------------------------------------------------------------------


def build_manifest(candidates_path: Path, gold_path: Path, input_refs: Iterable[ArtifactRef]) -> dict[str, Any]:
    """Assemble ``manifest.json`` from the payload files as written (export contract R2).

    Blake3 hashes are computed over the bundle's own files, so R7 self-consistency holds by
    construction; ``generated_at`` is the only non-deterministic field. Both ``task_counts``
    keys and both ``family_counts`` keys are always present.
    """
    rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = _load_gold(gold_path)
    gold_cases = gold.get("cases")
    task_tally = Counter(str(row.get("task")) for row in rows)
    family_tally = Counter(str(row.get("source_family")) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_schema": CANDIDATE_SCHEMA,
        "benchmark_schema": GOLD_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "files": {
            CANDIDATES_FILENAME: {"blake3": hash_file(candidates_path), "rows": len(rows)},
            GOLD_FILENAME: {"blake3": hash_file(gold_path), "cases": len(gold_cases) if isinstance(gold_cases, list) else 0},
        },
        "task_counts": {task: task_tally.get(task, 0) for task in _TASKS},
        "family_counts": {family: family_tally.get(family, 0) for family in _FAMILIES},
        "inputs": sorted({ref.blake3 for ref in input_refs}),
    }


def write_bundle(out_dir: Path, candidate_rows: Iterable[Mapping[str, str]], gold_src: Path, input_refs: Iterable[ArtifactRef]) -> dict[str, Path]:
    """Write the three-file bundle into ``out_dir``; returns the paths keyed by filename.

    Validates the gold benchmark BEFORE any file is written (export contract R6), then writes
    the deduped+sorted ``candidates.ndjson``, a byte-identical copy of the gold, and the
    manifest. Existing files are overwritten cleanly (idempotent re-run). Zero candidate rows
    is legal and yields an empty ``candidates.ndjson`` with 0 counts.
    """
    _load_gold(gold_src)
    rows = dedupe_sort(candidate_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        CANDIDATES_FILENAME: out_dir / CANDIDATES_FILENAME,
        GOLD_FILENAME: out_dir / GOLD_FILENAME,
        MANIFEST_FILENAME: out_dir / MANIFEST_FILENAME,
    }
    paths[CANDIDATES_FILENAME].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    shutil.copyfile(gold_src, paths[GOLD_FILENAME])
    manifest = build_manifest(paths[CANDIDATES_FILENAME], paths[GOLD_FILENAME], list(input_refs))
    paths[MANIFEST_FILENAME].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


# --- stage entry point ------------------------------------------------------------------


def export(inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Export the MEDliNER training-data bundle (Transformer-shaped entry point).

    Locates ``spl_documents.parquet`` and the FAERS global ``cases.parquet`` among
    ``inputs`` — a missing or unusable table is a loud ``RuntimeError`` naming the file
    (export contract R8). Derives the candidate rows, writes the bundle under
    ``Workdir(ctx.workdir).store / "medliner-export"``, registers the three files with the
    artifact store (provenance inputs = the consumed tables' blake3 ids), and returns
    ``[manifest, candidates, gold]`` refs.
    """
    refs = list(inputs)
    with step(logger, _OPERATION):
        dailymed_ref = next((ref for ref in refs if ref.uri.name == _DAILYMED_TABLE), None)
        if dailymed_ref is None:
            msg = f"medliner_export: missing DailyMed interim table {_DAILYMED_TABLE} among the input refs"
            raise RuntimeError(msg)
        faers_ref = next((ref for ref in refs if ref.uri.name == _FAERS_CASES_TABLE and not _is_faers_partition_ref(ref)), None)
        if faers_ref is None:
            msg = f"medliner_export: missing FAERS case table {_FAERS_CASES_TABLE} among the input refs"
            raise RuntimeError(msg)
        dailymed_table = find_table([dailymed_ref], _DAILYMED_TABLE)
        if dailymed_table is None:
            msg = f"medliner_export: unreadable DailyMed interim table {dailymed_ref.uri}"
            raise RuntimeError(msg)
        faers_table = find_faers_cases([faers_ref], columns=_FAERS_CASES_COLUMNS)
        if faers_table is None:
            msg = f"medliner_export: unusable FAERS case table {faers_ref.uri} (needs drugname/indication columns)"
            raise RuntimeError(msg)
        candidate_rows = [*select_dailymed_rows(dailymed_table), *select_faers_rows(faers_table)]
        out_dir = Workdir(ctx.workdir).store / OUT_DIRNAME
        paths = write_bundle(out_dir, candidate_rows, gold_path(), [dailymed_ref, faers_ref])
        candidates_path = paths[CANDIDATES_FILENAME]
        rows_written = len(candidates_path.read_text(encoding="utf-8").splitlines())
        store = ArtifactStore(Workdir(ctx.workdir))
        operation = OperationBlock(name=_OPERATION)
        input_ids = sorted({dailymed_ref.blake3, faers_ref.blake3})
        manifest_ref = store.register(paths[MANIFEST_FILENAME], media_type=_JSON_MEDIA_TYPE, inputs=input_ids, operation=operation)
        candidates_ref = store.register(candidates_path, media_type=_NDJSON_MEDIA_TYPE, rows=rows_written, inputs=input_ids, operation=operation)
        gold_ref = store.register(paths[GOLD_FILENAME], media_type=_JSON_MEDIA_TYPE, inputs=input_ids, operation=operation)
        stats(logger, _OPERATION, out_dir=str(out_dir), candidates=rows_written, manifest_blake3=manifest_ref.blake3)
    return [manifest_ref, candidates_ref, gold_ref]


__all__ = [
    "CANDIDATES_FILENAME",
    "CANDIDATE_SCHEMA",
    "GOLD_FILENAME",
    "GOLD_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "OUT_DIRNAME",
    "SCHEMA_VERSION",
    "build_manifest",
    "dedupe_sort",
    "export",
    "gold_path",
    "select_dailymed_rows",
    "select_faers_rows",
    "write_bundle",
]
