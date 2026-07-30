"""Local release-layout writer (Milestone 8) — assembles a provenance-complete ``release/``.

This is the *local* half of "release artifacts": it lays out a self-contained release
directory under the workdir with everything a consumer needs plus full provenance. It
deliberately does **NOT** publish or upload anywhere (publication was explicitly de-scoped);
it only writes local files.

Given a populated workdir (e.g. after :func:`dakp_pipeline.pipeline.run_pipeline`), it
assembles::

    <workdir>/release/
      VERSION                         # the release version string
      manifest.json                   # provenance: version, generated_at, git commit,
                                      #   input b3: hashes, row counts, schema fingerprints
      build_summary.json              # copied from data/reports/ (if present)
      tables/<assertion_table>.tsv    # the assertion tables (copied from data/tabular/)
      configs/<...>.yaml              # the generated Tablassert configs (from tables/)

Provenance is re-derived from the files themselves (BLAKE3 via
:func:`dakp_pipeline.io.content_hash.hash_file`, row counts + schema fingerprints from the
TSV headers), so the manifest is correct whether or not the caller still holds the original
:class:`~dakp_pipeline.io.contracts.ArtifactRef` handles. All paths recorded in the manifest
are release-relative (no absolute paths).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dakp_pipeline import __version__
from dakp_pipeline.io import schemas
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.logging_setup import bind
from dakp_pipeline.paths import Workdir

RELEASE_SCHEMA = "dakp.release.v1"
RELEASE_DIRNAME = "release"
_TABLES_DIRNAME = "tables"
_CONFIGS_DIRNAME = "configs"
_BUILD_SUMMARY_NAME = "build_summary.json"
_GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class ReleaseResult:
    """Handle to an assembled release directory."""

    root: Path
    manifest_path: Path
    version_path: Path
    manifest: dict[str, Any]
    tables: list[Path] = field(default_factory=list)
    configs: list[Path] = field(default_factory=list)


def _tsv_columns_and_rows(path: Path) -> tuple[list[str], int]:
    """Read a TSV's header columns and data-row count without loading the whole frame.

    Cheap even for ``prod``-sized tables: reads the header line, then counts the
    remaining lines. An empty file yields ``([], 0)``; a header-only file yields ``(cols, 0)``.
    """
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        columns = header.split("\t") if header else []
        rows = sum(1 for _ in handle)
    return columns, rows


def detect_git_commit() -> str | None:
    """Best-effort ``git rev-parse HEAD`` for the DAKP checkout, or ``None`` if unavailable.

    Runs in the repository containing this package (never the data workdir, which is not a
    repo). Any failure (no git, not a repo, timeout) degrades gracefully to ``None`` so a
    release can still be written without provenance-git.
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False, timeout=_GIT_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def write_release(workdir: Workdir | Path | str, *, version: str | None = None, git_commit: str | None = None) -> ReleaseResult:
    """Assemble a local ``release/`` directory under ``workdir`` with full provenance.

    Args:
        workdir: A populated pipeline workdir (or its root path). Assertion tables are read
            from ``data/tabular/``, Tablassert configs from ``tables/``, and the build summary
            from ``data/reports/build_summary.json``.
        version: Release version; defaults to the DAKP package version.
        git_commit: Git commit to record; defaults to auto-detection (``None`` if unavailable).

    Returns:
        A :class:`ReleaseResult` pointing at the assembled release directory and manifest.
    """
    wd = workdir if isinstance(workdir, Workdir) else Workdir(Path(workdir))
    resolved_version = version if version is not None else __version__
    resolved_commit = git_commit if git_commit is not None else detect_git_commit()
    log = bind(task_id="write_release", version=resolved_version)

    release_root = wd.root / RELEASE_DIRNAME
    tables_dir = release_root / _TABLES_DIRNAME
    configs_dir = release_root / _CONFIGS_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    table_entries, table_paths = _copy_tables(wd, tables_dir)
    config_entries, config_paths = _copy_configs(wd, configs_dir)
    build_summary_entry, build_profile = _copy_build_summary(wd, release_root)

    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "version": resolved_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": resolved_commit,
        "profile": build_profile,
        "inputs": {"tables": table_entries, "configs": config_entries, "build_summary": build_summary_entry},
        "row_counts": {entry["name"]: entry["rows"] for entry in table_entries},
        "schema_fingerprints": {entry["name"]: entry["schema_fingerprint"] for entry in table_entries},
    }

    manifest_path = release_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    version_path = release_root / "VERSION"
    version_path.write_text(f"{resolved_version}\n", encoding="utf-8")

    log.info("release assembled", tables=len(table_paths), configs=len(config_paths), release=str(release_root))
    return ReleaseResult(
        root=release_root, manifest_path=manifest_path, version_path=version_path, manifest=manifest, tables=table_paths, configs=config_paths
    )


# --- per-section assembly ---------------------------------------------------------


def _copy_tables(wd: Workdir, tables_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Copy the assertion TSV tables into the release and record their provenance."""
    entries: list[dict[str, Any]] = []
    paths: list[Path] = []
    for name in schemas.ASSERTION_TABLES:  # canonical order; only tables that exist are released
        src = wd.tabular / f"{name}.tsv"
        if not src.exists():
            continue
        dest = tables_dir / src.name
        shutil.copy2(src, dest)
        columns, rows = _tsv_columns_and_rows(src)
        entries.append(
            {
                "name": name,
                "path": f"{_TABLES_DIRNAME}/{src.name}",
                "blake3": hash_file(dest),
                "rows": rows,
                "schema_fingerprint": schemas.schema_fingerprint(columns),
            }
        )
        paths.append(dest)
    return entries, paths


def _copy_configs(wd: Workdir, configs_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Copy the generated Tablassert configs (``tables/*.yaml``) into the release."""
    entries: list[dict[str, Any]] = []
    paths: list[Path] = []
    configs_src = wd.root / _TABLES_DIRNAME
    if not configs_src.exists():
        return entries, paths
    for src in sorted(configs_src.glob("*.yaml"), key=lambda p: p.name):
        dest = configs_dir / src.name
        shutil.copy2(src, dest)
        entries.append({"name": src.stem, "path": f"{_CONFIGS_DIRNAME}/{src.name}", "blake3": hash_file(dest)})
        paths.append(dest)
    return entries, paths


def _copy_build_summary(wd: Workdir, release_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Copy ``data/reports/build_summary.json`` into the release; return (entry, profile)."""
    src = wd.reports / _BUILD_SUMMARY_NAME
    if not src.exists():
        return None, None
    dest = release_root / _BUILD_SUMMARY_NAME
    shutil.copy2(src, dest)
    profile = None
    try:
        profile = json.loads(src.read_text(encoding="utf-8")).get("profile")
    except (OSError, json.JSONDecodeError):
        profile = None
    return {"path": _BUILD_SUMMARY_NAME, "blake3": hash_file(dest)}, profile


__all__ = ["RELEASE_SCHEMA", "ReleaseResult", "detect_git_commit", "write_release"]
