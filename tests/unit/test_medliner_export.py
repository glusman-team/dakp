"""Unit tests for the MEDliNER training-data export stage (:mod:`dakp_pipeline.medliner_export`).

Happy-path coverage of the ``dakp.medliner.export.v1`` contract, run against REAL fixture
extracts per the ``tests/unit/conftest.py`` convention: LOINC section selection, FAERS
indication selection, the dedupe winner + deterministic ordering rules, the three-file bundle
layout with a verbatim gold copy, manifest self-consistency, and the Transformer-shaped
:func:`export` entry point registering three artifact refs. Error/edge branches live in
``test_medliner_export_edge.py``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl

from dakp_pipeline.assertions.evidence import dailymed_document_url, faers_record_url
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.medliner_export import (
    CANDIDATE_SCHEMA,
    CANDIDATES_FILENAME,
    GOLD_FILENAME,
    GOLD_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    OUT_DIRNAME,
    SCHEMA_VERSION,
    dedupe_sort,
    export,
    gold_path,
    select_dailymed_rows,
    select_faers_rows,
    write_bundle,
)
from dakp_pipeline.paths import Workdir

_EXPORT_LOINCS = ("34070-3", "34067-9")


def _bundle_dir(ctx: TaskContext) -> Path:
    return Workdir(ctx.workdir).store / OUT_DIRNAME


def _input_ref(directory: Path, name: str) -> ArtifactRef:
    """A stand-in interim-table ref whose blake3 id is real (content matters, bytes don't)."""
    path = directory / name
    path.write_bytes(b"payload:" + name.encode())
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _sample_rows() -> list[dict[str, str]]:
    """One DailyMed contraindication row + two FAERS indication rows (contract R3 shapes)."""
    return [
        {
            "text": "Contraindicated in patients with active liver disease.",
            "task": "contraindication",
            "source_family": "dailymed",
            "source_document_id": "set-1#34070-3",
            "section": "34070-3",
            "source_uri": dailymed_document_url("set-1#34070-3"),
        },
        {"text": "headache", "task": "indication", "source_family": "faers", "source_record_id": "1002", "source_uri": faers_record_url("24Q3")},
        {
            "text": "hypercholesterolemia",
            "task": "indication",
            "source_family": "faers",
            "source_record_id": "1001",
            "source_uri": faers_record_url("24Q3"),
        },
    ]


# --- helper-level tests -----------------------------------------------------------------


def test_gold_path_resolves_the_committed_benchmark() -> None:
    """R6: the exporter's gold source is the committed benchmark carrying the gold schema."""
    path = gold_path()
    assert path.name == GOLD_FILENAME
    gold = json.loads(path.read_text(encoding="utf-8"))
    assert gold["schema_version"] == GOLD_SCHEMA_VERSION
    assert gold["cases"]  # the benchmark has annotated cases to count


def test_select_dailymed_rows_maps_only_the_export_loincs(dailymed_refs: list[ArtifactRef]) -> None:
    """R4: only 34070-3/34067-9 sections with non-blank text become CandidateText rows."""
    docs = next(ref for ref in dailymed_refs if ref.uri.name == "spl_documents.parquet")
    table = pl.read_parquet(docs.uri)
    rows = select_dailymed_rows(table)
    expected = table.filter(pl.col("loinc_code").is_in(list(_EXPORT_LOINCS)) & (pl.col("section_text").str.strip_chars() != ""))
    assert len(rows) == expected.height > 0
    assert {row["task"] for row in rows} == {"contraindication", "indication"}
    for row in rows:
        assert row["source_family"] == "dailymed"
        assert row["section"] == ("34070-3" if row["task"] == "contraindication" else "34067-9")
        assert row["source_uri"] == dailymed_document_url(row["source_document_id"])
    first_contra = table.filter(pl.col("loinc_code") == "34070-3").row(0, named=True)
    assert {
        "text": str(first_contra["section_text"]).strip(),
        "task": "contraindication",
        "source_family": "dailymed",
        "source_document_id": first_contra["spl_document_id"],
        "section": "34070-3",
        "source_uri": dailymed_document_url(first_contra["spl_document_id"]),
    } in rows


def test_select_faers_rows_maps_case_indications(faers_refs: list[ArtifactRef]) -> None:
    """R4: every case row's non-blank indication becomes an indication row with primaryid + quarter URL."""
    cases = next(ref for ref in faers_refs if ref.uri.name == "cases.parquet" and "quarter=" not in str(ref.uri))
    table = pl.read_parquet(cases.uri)
    rows = select_faers_rows(table)
    assert len(rows) == table.filter(pl.col("indication").str.strip_chars() != "").height > 0
    for row, rec in zip(rows, table.iter_rows(named=True), strict=True):
        assert row == {
            "text": str(rec["indication"]).strip(),
            "task": "indication",
            "source_family": "faers",
            "source_record_id": rec["primaryid"],
            "source_uri": faers_record_url(rec["quarter"]),
        }


def test_dedupe_sort_keeps_the_deterministic_winner_for_any_input_order() -> None:
    """R5: duplicates on (task, norm text) keep the smallest source_document_id, verbatim."""
    doc_b = {"text": "Headache.", "task": "indication", "source_family": "dailymed", "source_document_id": "doc-b", "source_uri": "u-b"}
    doc_a = {"text": "headache.", "task": "indication", "source_family": "dailymed", "source_document_id": "doc-a", "source_uri": "u-a"}
    assert dedupe_sort([doc_b, doc_a]) == [doc_a]
    assert dedupe_sort([doc_a, doc_b]) == [doc_a]  # input order never changes the winner


def test_dedupe_sort_normalizes_case_and_whitespace_but_keeps_verbatim_text() -> None:
    """R5: the dedupe key is MEDliNER's normalization; the survivor keeps its verbatim text."""
    noisy = {"text": "Active  LIVER\n disease", "task": "contraindication", "source_document_id": "doc-1"}
    clean = {"text": "active liver disease", "task": "contraindication", "source_document_id": "doc-2"}
    other_task = {"text": "active liver disease", "task": "indication", "source_document_id": "doc-3"}
    assert dedupe_sort([noisy, clean, other_task]) == [noisy, other_task]  # same text, different task: both survive


def test_dedupe_sort_orders_lines_deterministically() -> None:
    """R5: output follows (task, normalized text, doc id, record id) — tasks first, then text."""
    zeta = {"text": "zeta text", "task": "indication", "source_record_id": "2"}
    alpha = {"text": "alpha text", "task": "indication", "source_record_id": "1"}
    contra = {"text": "contra text", "task": "contraindication", "source_document_id": "d"}
    assert dedupe_sort([zeta, alpha, contra]) == [contra, alpha, zeta]


# --- bundle-level tests -------------------------------------------------------------------


def test_write_bundle_writes_exactly_three_files_with_a_verbatim_gold(tmp_path: Path) -> None:
    """R1/R6: the bundle holds exactly the three contract files; gold is a byte-identical copy."""
    src = gold_path()
    paths = write_bundle(tmp_path / "out", _sample_rows(), src, [_input_ref(tmp_path, "spl_documents.parquet")])
    out_dir = tmp_path / "out"
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(paths)
    assert set(paths) == {MANIFEST_FILENAME, CANDIDATES_FILENAME, GOLD_FILENAME}
    assert (out_dir / GOLD_FILENAME).read_bytes() == src.read_bytes()
    text = (out_dir / CANDIDATES_FILENAME).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert len(text.splitlines()) == 3


def test_write_bundle_is_byte_identical_for_shuffled_input_order(tmp_path: Path) -> None:
    """R5: shuffling the input rows must not change one byte of candidates.ndjson."""
    rows = [
        *_sample_rows(),
        {"text": "HEADACHE", "task": "indication", "source_family": "faers", "source_record_id": "9999", "source_uri": faers_record_url("24Q2")},
        {"text": "headache", "task": "indication", "source_family": "faers", "source_record_id": "1002", "source_uri": faers_record_url("24Q3")},
    ]
    shuffled = random.Random(7).sample(rows, len(rows))
    write_bundle(tmp_path / "a", rows, gold_path(), [])
    write_bundle(tmp_path / "b", shuffled, gold_path(), [])
    assert (tmp_path / "a" / CANDIDATES_FILENAME).read_bytes() == (tmp_path / "b" / CANDIDATES_FILENAME).read_bytes()
    # the manifest too, except generated_at (the ONLY non-deterministic field)
    manifest_a = json.loads((tmp_path / "a" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest_b = json.loads((tmp_path / "b" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest_a.pop("generated_at")
    assert manifest_b.pop("generated_at")
    assert manifest_a == manifest_b


def test_write_bundle_manifest_matches_the_contract_and_self_verifies(tmp_path: Path) -> None:
    """R2/R7: schema strings, counts, sorted input ids, and blake3 hashes that re-verify."""
    dm_ref = _input_ref(tmp_path, "spl_documents.parquet")
    faers_ref = _input_ref(tmp_path, "cases.parquet")
    paths = write_bundle(tmp_path / "out", _sample_rows(), gold_path(), [faers_ref, dm_ref])
    manifest = json.loads(paths[MANIFEST_FILENAME].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["candidate_schema"] == CANDIDATE_SCHEMA
    assert manifest["benchmark_schema"] == GOLD_SCHEMA_VERSION
    assert manifest["generated_at"]
    assert manifest["files"][CANDIDATES_FILENAME] == {"blake3": hash_file(paths[CANDIDATES_FILENAME]), "rows": 3}
    gold_cases = json.loads(gold_path().read_text(encoding="utf-8"))["cases"]
    assert manifest["files"][GOLD_FILENAME] == {"blake3": hash_file(paths[GOLD_FILENAME]), "cases": len(gold_cases)}
    assert manifest["task_counts"] == {"contraindication": 1, "indication": 2}
    assert manifest["family_counts"] == {"dailymed": 1, "faers": 2}
    assert manifest["inputs"] == sorted([dm_ref.blake3, faers_ref.blake3])  # sorted regardless of arg order


def test_write_bundle_rows_only_use_candidatetext_fields(tmp_path: Path) -> None:
    """R3: emitted rows carry only CandidateText field names, with the per-family presence rules."""
    paths = write_bundle(tmp_path / "out", _sample_rows(), gold_path(), [])
    allowed = {"text", "task", "source_family", "source_document_id", "source_record_id", "section", "source_uri", "source_hash"}
    rows = [json.loads(line) for line in paths[CANDIDATES_FILENAME].read_text(encoding="utf-8").splitlines()]
    for row in rows:
        assert set(row) <= allowed
        assert row["text"].strip()
        assert row["task"] in {"indication", "contraindication"}
    dailymed = [row for row in rows if row["source_family"] == "dailymed"]
    faers = [row for row in rows if row["source_family"] == "faers"]
    assert dailymed
    assert faers
    assert all({"source_document_id", "section", "source_uri"} <= set(r) and "source_record_id" not in r for r in dailymed)
    assert all({"source_record_id", "source_uri"} <= set(r) and "source_document_id" not in r and "section" not in r for r in faers)


# --- entry point -----------------------------------------------------------------------------


def test_export_writes_the_bundle_and_registers_three_refs(ctx: TaskContext, dailymed_refs: list[ArtifactRef], faers_refs: list[ArtifactRef]) -> None:
    """US-001/R1: export() writes the bundle under the store and registers all three files."""
    refs = export([*dailymed_refs, *faers_refs], ctx)
    out_dir = _bundle_dir(ctx)
    assert [ref.uri for ref in refs] == [out_dir / MANIFEST_FILENAME, out_dir / CANDIDATES_FILENAME, out_dir / GOLD_FILENAME]
    assert all(ref.manifest is not None and ref.manifest.exists() for ref in refs)

    docs = next(ref for ref in dailymed_refs if ref.uri.name == "spl_documents.parquet")
    global_cases = next(ref for ref in faers_refs if ref.uri.name == "cases.parquet" and "quarter=" not in str(ref.uri))
    manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    # exactly the two consumed tables (ids are content hashes; a single-quarter fixture's
    # global and per-quarter cases.parquet can share one hash, so membership is by id, not path)
    assert manifest["inputs"] == sorted([docs.blake3, global_cases.blake3])
    assert len(manifest["inputs"]) == 2

    # R7: the recorded hashes reproduce against the written files
    assert manifest["files"][CANDIDATES_FILENAME]["blake3"] == hash_file(out_dir / CANDIDATES_FILENAME)
    assert manifest["files"][GOLD_FILENAME]["blake3"] == hash_file(out_dir / GOLD_FILENAME) == hash_file(gold_path())
    lines = (out_dir / CANDIDATES_FILENAME).read_text(encoding="utf-8").splitlines()
    assert manifest["files"][CANDIDATES_FILENAME]["rows"] == len(lines)
    assert sum(manifest["task_counts"].values()) == len(lines)
    assert refs[1].rows == len(lines) > 0
    # the artifact-store manifest carries the consumed table ids as provenance inputs
    assert refs[0].manifest is not None
    assert json.loads(refs[0].manifest.read_text(encoding="utf-8"))["inputs"] == sorted([docs.blake3, global_cases.blake3])


def test_export_is_idempotent_across_reruns(ctx: TaskContext, dailymed_refs: list[ArtifactRef], faers_refs: list[ArtifactRef]) -> None:
    """Edge spec: re-running (even with shuffled ref order) overwrites cleanly, identical bytes."""
    first = export([*dailymed_refs, *faers_refs], ctx)
    candidates_bytes = first[1].uri.read_bytes()
    second = export([*faers_refs, *dailymed_refs], ctx)
    assert second[1].uri.read_bytes() == candidates_bytes
    assert second[1].blake3 == first[1].blake3
    assert sorted(p.name for p in _bundle_dir(ctx).iterdir()) == sorted([CANDIDATES_FILENAME, GOLD_FILENAME, MANIFEST_FILENAME])
