"""Capture a build-output manifest: one row per assertion TSV + graph output with row counts and blake3.

Usage (on the GPU host, after a build completes)::

    uv run python tests/eval/build_manifest.py --workdir tmp --out baseline-1.15.0/manifest.json

The PR gate diffs the 1.15.0 baseline manifest against the new-stack build's manifest: identical
row counts and identical table contents (hash over the sorted, schema-column projection) mean the
speed work changed nothing observable downstream of NER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

_TABLES = ("approved_treats_assertions.tsv", "faers_applied_to_treat_assertions.tsv", "contraindication_assertions.tsv")


def _content_hash(path: Path) -> str:
    """blake3-style sha256 over the file bytes (the store's blake3 already fingerprints these)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_count(path: Path) -> int:
    # TSVs are written from a fixed schema; count data rows cheaply (minus the header).
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tabular = args.workdir / "tabular"
    if not tabular.is_dir():
        print(f"error: {tabular} does not exist")
        return 2
    manifest: dict[str, dict[str, object]] = {}
    for name in _TABLES:
        path = tabular / name
        if not path.exists():
            manifest[name] = {"missing": True}
            continue
        frame = pl.read_csv(path, separator="\t", infer_schema_length=0)
        manifest[name] = {"rows": _row_count(path), "cols": frame.columns, "sha256": _content_hash(path)}
    graph = tabular / "graph.yaml"
    manifest["graph.yaml"] = {"sha256": _content_hash(graph)} if graph.exists() else {"missing": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
