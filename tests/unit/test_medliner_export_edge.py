"""Error + edge-branch tests for :mod:`dakp_pipeline.medliner_export` (export-contract R6/R8).

Negative coverage: loud failures for missing/unusable input tables, missing/corrupt/wrong-schema
gold (always BEFORE any bundle file is written), plus the legal-but-degenerate paths: zero
exportable rows, null/blank cells, JSON-escapable text, and manifest derivation edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import dakp_pipeline.medliner_export as medliner_export
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.medliner_export import (
    CANDIDATES_FILENAME,
    GOLD_FILENAME,
    GOLD_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    OUT_DIRNAME,
    build_manifest,
    export,
    read_dailymed_sections,
    reduce_faers_frame,
    select_dailymed_rows,
    select_faers_rows,
    write_bundle,
)
from dakp_pipeline.paths import Workdir


def _table_ref(path: Path, table: pl.DataFrame) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/vnd.apache.parquet")


# --- R8: missing / unusable input tables ----------------------------------------------------


def test_export_without_spl_documents_raises_naming_the_file(ctx: TaskContext, faers_refs: list[ArtifactRef]) -> None:
    """R8: a missing DailyMed table is a loud error naming it; no bundle is started."""
    with pytest.raises(RuntimeError, match=r"spl_documents\.parquet"):
        export(faers_refs, ctx)
    assert not (Workdir(ctx.workdir).store / OUT_DIRNAME).exists()


def test_export_without_faers_cases_raises_naming_the_file(ctx: TaskContext, dailymed_refs: list[ArtifactRef]) -> None:
    """R8: a missing FAERS case table is a loud error naming it; no bundle is started."""
    with pytest.raises(RuntimeError, match=r"cases\.parquet"):
        export(dailymed_refs, ctx)
    assert not (Workdir(ctx.workdir).store / OUT_DIRNAME).exists()


def test_export_resolves_global_cases_by_immediate_partition_parent(ctx: TaskContext, tmp_path: Path) -> None:
    """WHY: an unrelated ancestor named ``quarter=`` must not hide the global case table."""
    dailymed = _table_ref(
        tmp_path / "spl_documents.parquet",
        pl.DataFrame({"spl_document_id": ["set-1#34067-9"], "loinc_code": ["34067-9"], "section_text": ["daily text"]}),
    )
    partition = _table_ref(
        tmp_path / "faers" / "quarter=24Q3" / "cases.parquet",
        pl.DataFrame({"quarter": ["24Q3"], "primaryid": ["partition"], "drugname": ["Drug"], "indication": ["partition text"]}),
    )
    global_cases = _table_ref(
        tmp_path / "quarter=unrelated-ancestor" / "global" / "cases.parquet",
        pl.DataFrame({"quarter": ["24Q3"], "primaryid": ["global"], "drugname": ["Drug"], "indication": ["global text"]}),
    )

    export([dailymed, partition, global_cases], ctx)

    rows = [json.loads(line) for line in (Workdir(ctx.workdir).store / OUT_DIRNAME / CANDIDATES_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert {row["text"] for row in rows} == {"daily text", "global text"}
    manifest = json.loads((Workdir(ctx.workdir).store / OUT_DIRNAME / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["inputs"] == sorted([dailymed.blake3, global_cases.blake3])


def test_export_with_an_unreadable_spl_documents_raises(ctx: TaskContext, faers_refs: list[ArtifactRef], tmp_path: Path) -> None:
    """R8: a corrupt DailyMed table fails loudly instead of producing a partial bundle."""
    bad = tmp_path / "spl_documents.parquet"
    bad.write_bytes(b"this is not parquet")
    ref = ArtifactRef(uri=bad, blake3=hash_file(bad), media_type="application/vnd.apache.parquet")
    with pytest.raises(RuntimeError, match="unreadable DailyMed"):
        export([ref, *faers_refs], ctx)


def test_export_with_a_columnless_faers_cases_raises(ctx: TaskContext, dailymed_refs: list[ArtifactRef], tmp_path: Path) -> None:
    """R8: a FAERS case table lacking drugname/indication is unusable and fails loudly."""
    bad = tmp_path / "cases.parquet"
    _table_ref(bad, pl.DataFrame({"quarter": ["24Q3"], "primaryid": ["1"]}))
    ref = ArtifactRef(uri=bad, blake3=hash_file(bad), media_type="application/vnd.apache.parquet")
    with pytest.raises(RuntimeError, match="unusable FAERS"):
        export([*dailymed_refs, ref], ctx)


def test_export_zero_exportable_rows_writes_a_valid_empty_bundle(ctx: TaskContext, tmp_path: Path) -> None:
    """R8: tables that exist but yield nothing are LEGAL: a valid bundle with 0 counts."""
    dailymed = _table_ref(
        tmp_path / "spl_documents.parquet",
        pl.DataFrame({"spl_document_id": ["set-x#99999-9"], "loinc_code": ["99999-9"], "section_text": ["a section nobody exports"]}),
    )
    cases = _table_ref(
        tmp_path / "cases.parquet", pl.DataFrame({"quarter": ["24Q3"], "primaryid": ["1"], "drugname": ["DrugX"], "indication": ["   "]})
    )
    refs = export([dailymed, cases], ctx)
    out_dir = Workdir(ctx.workdir).store / OUT_DIRNAME
    assert (out_dir / CANDIDATES_FILENAME).read_text(encoding="utf-8") == ""
    assert (out_dir / GOLD_FILENAME).read_bytes()  # the gold still ships
    manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["files"][CANDIDATES_FILENAME]["rows"] == 0
    assert manifest["task_counts"] == {"contraindication": 0, "indication": 0}
    assert manifest["family_counts"] == {"dailymed": 0, "faers": 0}
    assert refs[1].rows == 0


# --- R6: gold validation (always before any file is written) ---------------------------------


def test_gold_path_raises_when_the_benchmark_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gold resolver fails loudly when the committed benchmark is absent."""
    monkeypatch.setattr(medliner_export, "__file__", "/nonexistent/src/dakp_pipeline/medliner_export.py")
    with pytest.raises(FileNotFoundError, match=GOLD_FILENAME):
        medliner_export.gold_path()


def test_write_bundle_refuses_a_missing_gold_before_writing_anything(tmp_path: Path) -> None:
    """R6: a missing gold file raises before ANY bundle file exists."""
    out_dir = tmp_path / "bundle"
    with pytest.raises(FileNotFoundError, match=GOLD_FILENAME):
        write_bundle(out_dir, [{"text": "x", "task": "indication"}], tmp_path / "missing_ner_gold.json", [])
    assert not out_dir.exists()


def test_write_bundle_refuses_corrupt_gold(tmp_path: Path) -> None:
    """R6: gold that is not readable JSON is a loud ValueError; nothing is written."""
    bad = tmp_path / GOLD_FILENAME
    bad.write_text("{definitely not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not readable JSON"):
        write_bundle(tmp_path / "out", [], bad, [])
    assert not (tmp_path / "out").exists()


def test_write_bundle_refuses_an_unreadable_gold_path(tmp_path: Path) -> None:
    """R6: OS-level read failures (a directory where a file is expected) surface as ValueError."""
    bad = tmp_path / GOLD_FILENAME
    bad.mkdir()
    with pytest.raises(ValueError, match="not readable JSON"):
        write_bundle(tmp_path / "out", [], bad, [])
    assert not (tmp_path / "out").exists()


def test_write_bundle_refuses_a_wrong_schema_gold(tmp_path: Path) -> None:
    """R6: a gold carrying a different schema_version is rejected, never silently copied."""
    bad = tmp_path / GOLD_FILENAME
    bad.write_text(json.dumps({"schema_version": "dakp.ner.gold.v0", "cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        write_bundle(tmp_path / "out", [], bad, [])


def test_write_bundle_refuses_a_non_object_gold(tmp_path: Path) -> None:
    """WHY: R6 requires object metadata so malformed gold cannot produce a bundle."""
    bad = tmp_path / GOLD_FILENAME
    bad.write_text("[1, 2]", encoding="utf-8")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="JSON object"):
        write_bundle(out_dir, [], bad, [])
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("gold", "message"),
    [
        ({"schema_version": GOLD_SCHEMA_VERSION, "cases": [{}]}, "annotation_policy"),
        ({"schema_version": GOLD_SCHEMA_VERSION, "annotation_policy": "policy"}, "non-empty cases"),
        ({"schema_version": GOLD_SCHEMA_VERSION, "annotation_policy": "policy", "cases": []}, "non-empty cases"),
    ],
)
def test_write_bundle_refuses_incomplete_gold_metadata_before_output(tmp_path: Path, gold: dict[str, object], message: str) -> None:
    """WHY: required benchmark metadata prevents MEDliNER-rejectable partial bundles."""
    bad = tmp_path / GOLD_FILENAME
    bad.write_text(json.dumps(gold), encoding="utf-8")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        write_bundle(out_dir, [], bad, [])
    assert not out_dir.exists()


# --- selector robustness ---------------------------------------------------------------------


def test_select_dailymed_rows_skips_blank_and_null_cells() -> None:
    """Blank/null section text and unknown/null LOINCs are skipped; null doc ids degrade to ''."""
    table = pl.DataFrame(
        {
            "spl_document_id": ["set-1#34070-3", "set-2#34070-3", "set-3#34070-3", None],
            "loinc_code": ["34070-3", None, "34070-3", "34067-9"],
            "section_text": ["   ", "text here", None, "indication text"],
        }
    )
    rows = select_dailymed_rows(table)
    assert rows == [
        {
            "text": "indication text",
            "task": "indication",
            "source_family": "dailymed",
            "source_document_id": "",
            "section": "34067-9",
            "source_uri": "",
        }
    ]


def test_select_faers_rows_skips_blank_and_null_cells() -> None:
    """Blank/null indications are skipped; a null primaryid degrades to an empty record id."""
    table = pl.DataFrame({"indication": [None, "  ", "ok"], "primaryid": ["1", "2", None], "quarter": ["24Q3", "24Q3", "24Q3"]})
    rows = select_faers_rows(table)
    assert len(rows) == 1
    assert rows[0]["text"] == "ok"
    assert rows[0]["source_record_id"] == ""
    assert rows[0]["source_uri"].endswith("faers_ascii_2024q3.zip")


def test_candidates_roundtrip_json_escapable_text(tmp_path: Path) -> None:
    """Edge spec: quotes, newlines, and non-ASCII survive via json.dumps(ensure_ascii=False)."""
    tricky = 'He said "stop"\ncafé — 肝炎'
    rows = [{"text": tricky, "task": "indication", "source_family": "faers", "source_record_id": "1", "source_uri": "u"}]
    paths = write_bundle(tmp_path / "out", rows, medliner_export.gold_path(), [])
    raw = paths[CANDIDATES_FILENAME].read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 1
    assert "café" in raw  # non-ASCII written verbatim, not \u-escaped
    assert json.loads(raw.splitlines()[0])["text"] == tricky


def test_build_manifest_ignores_blank_lines_and_counts_valid_gold(tmp_path: Path) -> None:
    """WHY: manifest counts must remain correct for valid gold and blank JSONL lines."""
    candidates = tmp_path / CANDIDATES_FILENAME
    candidates.write_text('{"text": "a", "task": "indication", "source_family": "faers"}\n\n', encoding="utf-8")
    gold = tmp_path / GOLD_FILENAME
    gold.write_text(json.dumps({"schema_version": GOLD_SCHEMA_VERSION, "annotation_policy": "policy", "cases": [{}]}), encoding="utf-8")
    manifest = build_manifest(candidates, gold, [])
    assert manifest["files"][CANDIDATES_FILENAME]["rows"] == 1
    assert manifest["files"][GOLD_FILENAME]["cases"] == 1
    assert manifest["task_counts"] == {"contraindication": 0, "indication": 1}
    assert manifest["family_counts"] == {"dailymed": 0, "faers": 1}
    assert manifest["inputs"] == []


# --- fast paths: same bundle, less work ------------------------------------------------------


def test_dailymed_read_projects_and_pushes_the_loinc_filter_into_the_reader(tmp_path: Path) -> None:
    """The table holds one row per SPL section of EVERY type; only two LOINCs are exported."""
    path = tmp_path / "spl_documents.parquet"
    pl.DataFrame(
        {
            "spl_document_id": ["set-1#34070-3", "set-2#42229-5", "set-3#34067-9"],
            "loinc_code": ["34070-3", "42229-5", "34067-9"],
            "section_text": ["contra text", "boilerplate nobody exports", "indication text"],
            "xml_path": ["a.xml", "b.xml", "c.xml"],
        }
    ).write_parquet(path)
    table = read_dailymed_sections(path)
    assert table is not None
    assert table.columns == list(medliner_export._DAILYMED_COLUMNS)  # the unread columns never load
    assert table.height == 2
    assert select_dailymed_rows(table) == select_dailymed_rows(pl.read_parquet(path))


def test_dailymed_read_falls_back_to_a_full_read_on_an_unexpected_schema(tmp_path: Path) -> None:
    """An older/renamed schema still exports rather than silently yielding nothing."""
    path = tmp_path / "spl_documents.parquet"
    pl.DataFrame({"loinc_code": ["34067-9"], "section_text": ["indication text"]}).write_parquet(path)
    table = read_dailymed_sections(path)
    assert table is not None
    assert table.height == 1
    assert select_dailymed_rows(table)[0]["source_document_id"] == ""


def test_dailymed_read_reports_an_unreadable_table_as_none(tmp_path: Path) -> None:
    path = tmp_path / "spl_documents.parquet"
    path.write_bytes(b"this is not parquet")
    assert read_dailymed_sections(path) is None


def test_faers_reduction_keeps_exactly_the_rows_dedupe_sort_would_have_kept() -> None:
    """The reduction must be winner-identical, not merely smaller.

    Case variants and whitespace variants collapse to the same ``_normalized`` key, duplicate
    primaryids force the tie-break, and blank/null indications must still be dropped.
    """
    table = pl.DataFrame(
        {
            "indication": ["ASTHMA", "asthma  ", "  Asthma", "ASTHMA", None, "   ", "pain", "pain"],
            "primaryid": ["5", "2", "8", "3", "9", "9", "7", "1"],
            "quarter": ["24Q3", "24Q3", "24Q1", "24Q3", "24Q3", "24Q3", "24Q4", "24Q2"],
        }
    )
    reference = medliner_export.dedupe_sort(select_faers_rows(table))
    reduced = medliner_export.dedupe_sort(select_faers_rows(reduce_faers_frame(table)))
    assert reduced == reference
    assert reduce_faers_frame(table).height < table.height


def test_faers_reduction_is_winner_identical_under_shuffled_input_order() -> None:
    """Differential check over random orderings: the bundle cannot depend on row order."""
    rows = {
        "indication": ["asthma", "ASTHMA", "asthma ", "pain", "PAIN", "nausea", "", "nausea "],
        "primaryid": ["4", "1", "9", "6", "2", "7", "3", "5"],
        "quarter": ["24Q1", "24Q2", "24Q3", "24Q4", "24Q1", "24Q2", "24Q3", "24Q4"],
    }
    table = pl.DataFrame(rows)
    reference = medliner_export.dedupe_sort(select_faers_rows(table))
    for seed in range(5):
        shuffled = table.sample(fraction=1.0, shuffle=True, seed=seed)
        assert medliner_export.dedupe_sort(select_faers_rows(reduce_faers_frame(shuffled))) == reference


def test_faers_reduction_passes_a_table_without_indications_straight_through() -> None:
    table = pl.DataFrame({"quarter": ["24Q3"], "primaryid": ["1"]})
    assert reduce_faers_frame(table).equals(table)
    assert select_faers_rows(table) == []


def test_faers_reduction_tolerates_a_table_without_primaryid() -> None:
    """A primaryid-less table still reduces; the record id degrades to '' as it always did."""
    table = pl.DataFrame({"indication": ["asthma", "asthma", "pain"], "quarter": ["24Q3"] * 3})
    reduced = reduce_faers_frame(table)
    assert reduced.height == 2
    assert [row["source_record_id"] for row in select_faers_rows(reduced)] == ["", ""]


def test_a_table_without_a_quarter_column_still_fails_loudly() -> None:
    """Unchanged behavior: there is no source URL to record without a quarter."""
    table = pl.DataFrame({"indication": ["asthma"], "primaryid": ["1"]})
    with pytest.raises(ValueError, match="invalid FAERS quarter"):
        select_faers_rows(reduce_faers_frame(table))


def test_a_malformed_quarter_still_fails_loudly() -> None:
    """Resolving one URL per distinct quarter must not turn a bad quarter into a silent pass."""
    table = pl.DataFrame({"indication": ["asthma"], "primaryid": ["1"], "quarter": ["not-a-quarter"]})
    with pytest.raises(ValueError, match="invalid FAERS quarter"):
        select_faers_rows(table)


def test_a_malformed_quarter_on_a_blank_indication_row_is_still_ignored() -> None:
    """A row the export never emits must not be able to fail the export."""
    table = pl.DataFrame({"indication": ["  ", "asthma"], "primaryid": ["1", "2"], "quarter": ["not-a-quarter", "24Q3"]})
    rows = select_faers_rows(table)
    assert [row["text"] for row in rows] == ["asthma"]


# --- already-done skip ------------------------------------------------------------------------


def test_export_skips_the_whole_stage_when_its_inputs_are_unchanged(ctx: TaskContext, tmp_path: Path) -> None:
    """Every other expensive stage skips on unchanged inputs; this one used to redo everything."""
    dailymed = _table_ref(
        tmp_path / "spl_documents.parquet",
        pl.DataFrame({"spl_document_id": ["set-1#34067-9"], "loinc_code": ["34067-9"], "section_text": ["daily text"]}),
    )
    cases = _table_ref(
        tmp_path / "cases.parquet", pl.DataFrame({"quarter": ["24Q3"], "primaryid": ["1"], "drugname": ["Drug"], "indication": ["case text"]})
    )
    first = export([dailymed, cases], ctx)
    bundle = Workdir(ctx.workdir).store / OUT_DIRNAME / CANDIDATES_FILENAME
    stamp = bundle.stat().st_mtime_ns

    second = export([dailymed, cases], ctx)

    assert [ref.blake3 for ref in second] == [ref.blake3 for ref in first]
    assert bundle.stat().st_mtime_ns == stamp  # nothing was rewritten


def test_force_bypasses_the_skip_and_rebuilds(ctx: TaskContext, tmp_path: Path) -> None:
    dailymed = _table_ref(
        tmp_path / "spl_documents.parquet",
        pl.DataFrame({"spl_document_id": ["set-1#34067-9"], "loinc_code": ["34067-9"], "section_text": ["daily text"]}),
    )
    cases = _table_ref(
        tmp_path / "cases.parquet", pl.DataFrame({"quarter": ["24Q3"], "primaryid": ["1"], "drugname": ["Drug"], "indication": ["case text"]})
    )
    export([dailymed, cases], ctx)
    bundle = Workdir(ctx.workdir).store / OUT_DIRNAME / CANDIDATES_FILENAME
    bundle.write_text("", encoding="utf-8")  # a rebuild must restore this

    forced = TaskContext(workdir=ctx.workdir, fixture_root=ctx.fixture_root, params={**dict(ctx.params), "force": True})
    export([dailymed, cases], forced)

    assert "case text" in bundle.read_text(encoding="utf-8")


def test_build_manifest_reads_the_payload_back_when_the_caller_passes_nothing(tmp_path: Path) -> None:
    """External callers get the on-disk derivation; the stage passes what it just wrote."""
    paths = write_bundle(tmp_path / "out", _sample_faers_rows(), medliner_export.gold_path(), [])
    from_disk = build_manifest(paths[CANDIDATES_FILENAME], paths[GOLD_FILENAME], [])
    from_memory = build_manifest(paths[CANDIDATES_FILENAME], paths[GOLD_FILENAME], [], rows=_sample_faers_rows(), gold=None)
    assert from_disk["files"] == from_memory["files"]
    assert from_disk["task_counts"] == from_memory["task_counts"]
    assert from_disk["family_counts"] == from_memory["family_counts"]


def _sample_faers_rows() -> list[dict[str, str]]:
    return [
        {"text": "asthma", "task": "indication", "source_family": "faers", "source_record_id": "1", "source_uri": "u"},
        {"text": "pain", "task": "indication", "source_family": "faers", "source_record_id": "2", "source_uri": "u"},
    ]
