"""REAL Go<->Python parity: the Go ``dakp-worker`` subcommands are drop-in extractors.

For each source this builds the Go binary (skipping cleanly if Go is absent), runs BOTH the
Go subcommand and the Python extractor on the *same* pipeline fixture, and asserts the
produced uncompressed TSV tables are **byte-for-byte identical**. That is the proof the Go
workers can replace the Python extractors: same input bytes in, same Tablassert-facing TSV
bytes out. A second set of tests drives the same parity through the extractors' opt-in
delegation path (``use_go_workers=True``), proving ``extract()`` shells out to Go correctly.

Tables the Python path only emits as parquet are rendered to TSV with
``pl.DataFrame.write_csv(separator="\\t")`` — exactly how the Go golden fixtures in
``go/internal/*/testdata/golden/`` were produced — before comparing.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.extract import drugsfda_products, faers_ascii, spl_xml
from dakp_pipeline.io.content_hash import hash_file, hash_tree
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.workers import go_runner
from dakp_pipeline.workers.go_runner import GoRunner

pytestmark = pytest.mark.skipif(not go_runner.go_available(), reason="Go toolchain not installed (Go worker parity needs `go`)")

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


@pytest.fixture(scope="session")
def runner() -> GoRunner:
    """A GoRunner with the dakp-worker binary built once per session (cached by source hash)."""
    built = GoRunner()
    built.ensure_binary()
    return built


@pytest.fixture
def delegate_runner(runner: GoRunner, monkeypatch: pytest.MonkeyPatch) -> GoRunner:
    """Route the extractors' ``get_runner()`` calls to the session runner (no rebuild)."""
    monkeypatch.setattr(go_runner, "get_runner", lambda: runner)
    return runner


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _ctx(workdir: Path, *, use_go: bool = False) -> TaskContext:
    Workdir(workdir).create()
    return TaskContext(profile="mock", workdir=workdir, fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params={"use_go_workers": use_go})


def _tsv_bytes(frame: pl.DataFrame) -> bytes:
    """Render a frame to uncompressed TSV bytes (how the Go goldens were produced).

    polars ``write_csv`` returns a ``str`` when given no target, so write into a ``BytesIO``
    to get the exact on-disk bytes (identical to ``write_csv(path, separator="\\t")``).
    """
    buffer = io.BytesIO()
    frame.write_csv(buffer, separator="\t")
    return buffer.getvalue()


def _by_name(refs: list[ArtifactRef], filename: str) -> ArtifactRef:
    for ref in refs:
        if ref.uri.name == filename:
            return ref
    msg = f"no output named {filename!r}; got {[r.uri.name for r in refs]}"
    raise AssertionError(msg)


# --- hash subcommand parity ------------------------------------------------------


def test_hash_subcommand_matches_python(runner: GoRunner, tmp_path: Path) -> None:
    fixture = _FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz"
    file_result = runner.run("hash", [fixture])
    assert file_result.artifact_id == hash_file(fixture)

    # A directory tree-hashes identically to the Python reference (go/README.md parity lock).
    tree_dir = tmp_path / "tree"
    (tree_dir / "nested").mkdir(parents=True)
    (tree_dir / "a.txt").write_text("alpha")
    (tree_dir / "nested" / "b.txt").write_text("beta")
    tree_result = runner.run("hash", ["-mode=tree", tree_dir])
    assert tree_result.artifact_id == hash_tree(tree_dir)


# --- DailyMed parity -------------------------------------------------------------


def test_dailymed_parity(runner: GoRunner, tmp_path: Path) -> None:
    fixture_dir = _FIXTURE_ROOT / "dailymed"
    go_out = tmp_path / "go"
    go_out.mkdir()
    runner.run_table("dailymed", fixture_dir, go_out)

    refs = spl_xml.extract([_ref(fixture_dir / "dailymed_spl.xml.gz")], _ctx(tmp_path / "py"))

    # Every one of the five normalized tables is byte-for-byte identical (Go TSV vs the
    # Python parquet rendered to TSV).
    tables = {
        "spl_documents": "spl_documents.parquet",
        "spl_sets": "spl_sets.parquet",
        "spl_approvals": "spl_approvals.parquet",
        "spl_ingredients": "spl_ingredients.parquet",
        "spl_sections": "spl_sections.parquet",
    }
    for tsv_stem, parquet_name in tables.items():
        go_bytes = (go_out / f"{tsv_stem}.tsv").read_bytes()
        py_bytes = _tsv_bytes(pl.read_parquet(_by_name(refs, parquet_name).uri))
        assert go_bytes == py_bytes, f"{tsv_stem}.tsv differs between Go and Python"

    # The public section TSV the Python path writes for Tablassert matches Go's spl_sections.tsv.
    assert _by_name(refs, "dailymed_spl_sections.tsv").uri.read_bytes() == (go_out / "spl_sections.tsv").read_bytes()


# --- FAERS parity ----------------------------------------------------------------


def test_faers_parity(runner: GoRunner, tmp_path: Path) -> None:
    fixture_dir = _FIXTURE_ROOT / "faers"
    go_out = tmp_path / "go"
    go_out.mkdir()
    runner.run_table("faers", fixture_dir, go_out)

    txt_refs = [_ref(path) for path in sorted(fixture_dir.glob("*.txt"))]
    refs = faers_ascii.extract(txt_refs, _ctx(tmp_path / "py"))

    # The public case table (the Tablassert source-section contract) is byte-identical.
    assert _by_name(refs, "faers_cases.tsv").uri.read_bytes() == (go_out / "faers_cases.tsv").read_bytes()

    # Audits: Go writes TSV, Python writes parquet — compare on the same TSV rendering.
    for stem, parquet_name in (("delete_audit", "delete_audit.parquet"), ("dedup_audit", "dedup_audit.parquet")):
        go_bytes = (go_out / f"{stem}.tsv").read_bytes()
        py_bytes = _tsv_bytes(pl.read_parquet(_by_name(refs, parquet_name).uri))
        assert go_bytes == py_bytes, f"{stem}.tsv differs between Go and Python"


# --- Drugs@FDA parity ------------------------------------------------------------


def test_drugsfda_parity(runner: GoRunner, tmp_path: Path) -> None:
    fixture_dir = _FIXTURE_ROOT / "drugsfda"
    go_out = tmp_path / "go"
    go_out.mkdir()
    runner.run_table("drugsfda", fixture_dir, go_out)

    names = ("drugsfda_products.tsv", "drugsfda_applications.tsv", "drugsfda_submissions.tsv")
    refs = drugsfda_products.extract([_ref(fixture_dir / name) for name in names], _ctx(tmp_path / "py"))

    # The public products TSV (Tablassert handoff) is byte-identical.
    assert _by_name(refs, "drugsfda_products.tsv").uri.read_bytes() == (go_out / "drugsfda_products.tsv").read_bytes()

    # applications/submissions/lookups: Go TSV vs Python parquet rendered to TSV.
    for stem, parquet_name in (
        ("drugsfda_applications", "applications.parquet"),
        ("drugsfda_submissions", "submissions.parquet"),
        ("drugsfda_lookups", "lookups.parquet"),
    ):
        go_bytes = (go_out / f"{stem}.tsv").read_bytes()
        py_bytes = _tsv_bytes(pl.read_parquet(_by_name(refs, parquet_name).uri))
        assert go_bytes == py_bytes, f"{stem}.tsv differs between Go and Python"


# --- delegation through extract() (use_go_workers=True) --------------------------


def test_dailymed_delegation_via_extract(delegate_runner: GoRunner, tmp_path: Path) -> None:
    fixture = _FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz"
    py_refs = spl_xml.extract([_ref(fixture)], _ctx(tmp_path / "py"))
    go_refs = spl_xml.extract([_ref(fixture)], _ctx(tmp_path / "go", use_go=True))

    # Same ref names/order, and the public section TSV is byte-identical.
    assert [r.uri.name for r in go_refs] == [r.uri.name for r in py_refs]
    assert _by_name(go_refs, "dailymed_spl_sections.tsv").uri.read_bytes() == _by_name(py_refs, "dailymed_spl_sections.tsv").uri.read_bytes()
    # The Go-delegated parquet tables carry identical data.
    for name in ("spl_documents.parquet", "spl_sections.parquet", "spl_ingredients.parquet"):
        assert pl.read_parquet(_by_name(go_refs, name).uri).equals(pl.read_parquet(_by_name(py_refs, name).uri))


def test_faers_delegation_via_extract(delegate_runner: GoRunner, tmp_path: Path) -> None:
    fixture_dir = _FIXTURE_ROOT / "faers"
    txt_refs = [_ref(path) for path in sorted(fixture_dir.glob("*.txt"))]
    py_refs = faers_ascii.extract(txt_refs, _ctx(tmp_path / "py"))
    go_refs = faers_ascii.extract(txt_refs, _ctx(tmp_path / "go", use_go=True))

    # The global cases parquet is still returned first (downstream find_faers_cases resolves it).
    assert go_refs[0].uri.name == "cases.parquet"
    # The public case TSV is byte-identical to the Python path.
    assert _by_name(go_refs, "faers_cases.tsv").uri.read_bytes() == _by_name(py_refs, "faers_cases.tsv").uri.read_bytes()
    # The reconstructed cases parquet agrees with Python on the public columns.
    from dakp_pipeline.io import schemas

    go_cases = pl.read_parquet(_by_name(go_refs, "cases.parquet").uri).select(schemas.FAERS_CASES_COLUMNS)
    py_cases = pl.read_parquet(_by_name(py_refs, "cases.parquet").uri).select(schemas.FAERS_CASES_COLUMNS)
    assert go_cases.equals(py_cases)


def test_drugsfda_delegation_via_extract(delegate_runner: GoRunner, tmp_path: Path) -> None:
    fixture_dir = _FIXTURE_ROOT / "drugsfda"
    names = ("drugsfda_products.tsv", "drugsfda_applications.tsv", "drugsfda_submissions.tsv")
    inputs = [_ref(fixture_dir / name) for name in names]
    py_refs = drugsfda_products.extract(inputs, _ctx(tmp_path / "py"))
    go_refs = drugsfda_products.extract(inputs, _ctx(tmp_path / "go", use_go=True))

    assert [r.uri.name for r in go_refs] == [r.uri.name for r in py_refs]
    assert _by_name(go_refs, "drugsfda_products.tsv").uri.read_bytes() == _by_name(py_refs, "drugsfda_products.tsv").uri.read_bytes()
    for name in ("products.parquet", "applications.parquet", "submissions.parquet", "lookups.parquet"):
        assert pl.read_parquet(_by_name(go_refs, name).uri).equals(pl.read_parquet(_by_name(py_refs, name).uri))
