"""Unit tests for the local release-layout writer (Milestone 8).

Builds a small "mock build output" workdir (assertion TSVs + Tablassert configs + a build
summary) and asserts ``write_release`` lays out ``release/`` with the tables, configs,
build_summary.json, a provenance ``manifest.json`` (version, generated_at, git commit, input
``b3:`` hashes, row counts, schema fingerprints), and a ``VERSION`` file — all locally, with
no publication. Also covers faithful byte copies, version/commit overrides, graceful handling
of a missing build summary, git-commit auto-detection, determinism of the manifest shape and
content hashes, and a run against genuine ``run_pipeline`` output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from dakp_pipeline import __version__
from dakp_pipeline.io import schemas
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.paths import Workdir
from dakp_pipeline.pipeline import run_pipeline
from dakp_pipeline.release import RELEASE_SCHEMA, detect_git_commit, write_release

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
TABLES = ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions")
CONFIG_NAMES = {"graph", "approved_treats", "faers_applied_to_treat", "contraindications"}
# rows per assertion table in the mock build output (approved gets two rows to exercise counting).
_MOCK_ROWS = {"approved_treats_assertions": 2, "faers_applied_to_treat_assertions": 1, "contraindication_assertions": 1}


def _write_mock_table(wd: Workdir, name: str, rows: int) -> Path:
    columns = schemas.columns_for(name)
    path = wd.tabular / f"{name}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(columns)]
    lines.extend("\t".join(f"v{col}" for col in columns) for _ in range(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_mock_workdir(root: Path, *, with_build_summary: bool = True) -> Workdir:
    """Assemble a minimal but faithful "mock build output" workdir for the release writer."""
    wd = Workdir(root)
    wd.create()
    for name in TABLES:
        _write_mock_table(wd, name, _MOCK_ROWS[name])
    configs_dir = wd.root / "tables"
    configs_dir.mkdir(parents=True, exist_ok=True)
    for config_name in CONFIG_NAMES:
        (configs_dir / f"{config_name}.yaml").write_text(f"name: {config_name}\n", encoding="utf-8")
    if with_build_summary:
        (wd.reports / "build_summary.json").write_text(json.dumps({"schema_version": "dakp.build_summary.v1", "profile": "mock"}), encoding="utf-8")
    return wd


# --- release layout ---------------------------------------------------------------


def test_write_release_assembles_layout(tmp_path: Path) -> None:
    wd = _build_mock_workdir(tmp_path / "work")
    release = write_release(wd)

    assert release.root == wd.root / "release"
    assert release.manifest_path == release.root / "manifest.json"
    assert release.version_path == release.root / "VERSION"
    assert release.manifest_path.exists()
    assert release.version_path.exists()
    assert (release.root / "build_summary.json").exists()

    # The three assertion tables land under release/tables/.
    assert {p.name for p in release.tables} == {f"{name}.tsv" for name in TABLES}
    assert all(p.parent == release.root / "tables" for p in release.tables)
    # The generated Tablassert configs land under release/configs/.
    assert {p.stem for p in release.configs} == CONFIG_NAMES
    assert all(p.parent == release.root / "configs" for p in release.configs)

    assert release.version_path.read_text(encoding="utf-8") == f"{__version__}\n"


def test_release_manifest_has_full_provenance(tmp_path: Path) -> None:
    wd = _build_mock_workdir(tmp_path / "work")
    manifest = write_release(wd, git_commit="0" * 40).manifest

    assert manifest["schema_version"] == RELEASE_SCHEMA
    assert manifest["version"] == __version__
    assert isinstance(manifest["generated_at"], str)
    assert manifest["generated_at"]
    assert manifest["git_commit"] == "0" * 40
    assert manifest["profile"] == "mock"

    # Input b3: hashes + row counts + schema fingerprints for every assertion table.
    table_entries = {entry["name"]: entry for entry in manifest["inputs"]["tables"]}
    assert set(table_entries) == set(TABLES)
    for name, entry in table_entries.items():
        assert entry["blake3"].startswith("b3:")
        assert entry["rows"] == _MOCK_ROWS[name]
        assert entry["schema_fingerprint"].startswith("b3:")
        # The fingerprint reflects the table's column contract (the mock writes the header).
        assert entry["schema_fingerprint"] == schemas.schema_fingerprint(schemas.columns_for(name))
        assert entry["path"] == f"tables/{name}.tsv"

    assert manifest["row_counts"] == _MOCK_ROWS
    assert set(manifest["schema_fingerprints"]) == set(TABLES)

    # Configs + build summary carry b3: hashes too.
    assert {entry["name"] for entry in manifest["inputs"]["configs"]} == CONFIG_NAMES
    assert all(entry["blake3"].startswith("b3:") for entry in manifest["inputs"]["configs"])
    assert manifest["inputs"]["build_summary"]["blake3"].startswith("b3:")


def test_release_records_no_absolute_paths(tmp_path: Path) -> None:
    wd = _build_mock_workdir(tmp_path / "work")
    manifest = write_release(wd).manifest

    recorded = [entry["path"] for entry in manifest["inputs"]["tables"]]
    recorded.extend(entry["path"] for entry in manifest["inputs"]["configs"])
    recorded.append(manifest["inputs"]["build_summary"]["path"])
    assert all(not path.startswith("/") for path in recorded)


def test_release_copies_table_bytes_faithfully(tmp_path: Path) -> None:
    wd = _build_mock_workdir(tmp_path / "work")
    release = write_release(wd)

    for copied in release.tables:
        source = wd.tabular / copied.name
        assert copied.read_bytes() == source.read_bytes()
        assert hash_file(copied) == hash_file(source)


# --- overrides + robustness -------------------------------------------------------


def test_release_version_and_commit_overrides(tmp_path: Path) -> None:
    wd = _build_mock_workdir(tmp_path / "work")
    release = write_release(wd, version="9.9.9", git_commit="deadbeef" * 5)

    assert release.manifest["version"] == "9.9.9"
    assert release.manifest["git_commit"] == "deadbeef" * 5
    assert release.version_path.read_text(encoding="utf-8") == "9.9.9\n"


def test_release_handles_missing_build_summary(tmp_path: Path) -> None:
    wd = _build_mock_workdir(tmp_path / "work", with_build_summary=False)
    release = write_release(wd)

    assert release.manifest["inputs"]["build_summary"] is None
    assert release.manifest["profile"] is None
    assert not (release.root / "build_summary.json").exists()
    # Tables + configs are still released.
    assert len(release.tables) == len(TABLES)


def test_detect_git_commit_returns_hash_or_none() -> None:
    commit = detect_git_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


# --- determinism ------------------------------------------------------------------


def test_release_manifest_shape_and_hashes_are_deterministic(tmp_path: Path) -> None:
    first = write_release(_build_mock_workdir(tmp_path / "a"), version="1.2.3", git_commit="f" * 40).manifest
    second = write_release(_build_mock_workdir(tmp_path / "b"), version="1.2.3", git_commit="f" * 40).manifest

    # Identical shape: same keys, same table order, same per-entry fields.
    assert list(first) == list(second)
    assert [e["name"] for e in first["inputs"]["tables"]] == [e["name"] for e in second["inputs"]["tables"]]
    # Identical content hashes for identical inputs (content-addressing is deterministic).
    first_hashes = {e["name"]: e["blake3"] for e in first["inputs"]["tables"]}
    second_hashes = {e["name"]: e["blake3"] for e in second["inputs"]["tables"]}
    assert first_hashes == second_hashes
    assert first["row_counts"] == second["row_counts"]
    assert first["schema_fingerprints"] == second["schema_fingerprints"]


# --- against genuine pipeline output ----------------------------------------------


def test_release_against_real_pipeline_output(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    run_pipeline(profile="mock", fixture_root=FIXTURE_ROOT, workdir=workdir, run_airflow=False)
    release = write_release(Workdir(workdir), git_commit="1" * 40)

    assert release.manifest["profile"] == "mock"
    assert set(release.manifest["row_counts"]) == set(TABLES)
    assert all(rows > 0 for rows in release.manifest["row_counts"].values())
    assert {p.stem for p in release.configs} == CONFIG_NAMES
    assert (release.root / "build_summary.json").exists()
