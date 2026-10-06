#!/usr/bin/env python3
"""Select this CI runner's slice of ``tests/unit``.

The unit phase is split across several runners to keep the workflow inside its wall-clock
budget. The split is balanced by MEASURED per-file pytest seconds
(``.github/ci/unit-test-durations.json``) with a greedy longest-processing-time pack, because
neither file size nor test count predicts duration here: one file
(``tests/unit/test_tablassert_configs.py``) is about a third of the suite's CPU, so a
size-balanced split left one shard running 1.8x longer than its sibling.

Guarantees (asserted in ``tests/unit/test_ci_unit_shard_selector.py``):

- every ``git ls-files tests/unit/test_*.py`` path lands in exactly one shard, so no test can
  be dropped by the split; a file absent from the durations data is assigned the median
  measured duration, which keeps new tests running without anyone editing CI first;
- the selection is a pure function of the tracked file list, the durations data, and the shard
  index: no clock, no randomness, no runner-local state, so every shard agrees on the
  partition and a rerun reproduces it byte for byte;
- durations entries for deleted files are ignored.

Usage: ``select_unit_shard.py <shard> [shard-count]`` (shard is 1-based) prints one path per
line. The workflow pipes it into ``pytest``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DURATIONS_PATH = Path(__file__).resolve().with_name("unit-test-durations.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_GLOB = "tests/unit/test_*.py"


def tracked_unit_files() -> list[str]:
    """The tracked unit test files, in git's index order (stable across runners)."""
    out = subprocess.run(["git", "ls-files", UNIT_GLOB], capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    return [line for line in out.splitlines() if line]


def measured_seconds() -> dict[str, float]:
    """Per-file measured pytest seconds; entries for files that no longer exist are dropped."""
    data = json.loads(DURATIONS_PATH.read_text(encoding="utf-8"))
    seconds = data.get("seconds")
    if not isinstance(seconds, dict):
        raise ValueError(f"{DURATIONS_PATH.name} has no 'seconds' mapping")
    return {str(path): float(value) for path, value in seconds.items()}


def shard_files(files: list[str], shard: int, shard_count: int, seconds: dict[str, float]) -> list[str]:
    """The LPT bin for ``shard`` (1-based): heaviest file first into the lightest bin so far.

    Ties break on the lower bin index and then on path order, so the pack is deterministic.
    """
    if not 1 <= shard <= shard_count:
        raise ValueError(f"shard {shard} is outside 1..{shard_count}")
    known = [seconds[path] for path in files if path in seconds]
    fallback = sorted(known)[len(known) // 2] if known else 0.0
    bins: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for path in sorted(files, key=lambda p: (-seconds.get(p, fallback), p)):
        lightest = min(range(shard_count), key=lambda i: (loads[i], i))
        bins[lightest].append(path)
        loads[lightest] += seconds.get(path, fallback)
    return sorted(bins[shard - 1])


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    shard = int(argv[0])
    shard_count = int(argv[1]) if len(argv) == 2 else 3
    for path in shard_files(tracked_unit_files(), shard, shard_count, measured_seconds()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
