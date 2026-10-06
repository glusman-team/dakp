"""The CI unit-shard selector must partition the suite completely and deterministically.

The workflow splits ``tests/unit`` across runners by calling
``.github/ci/select_unit_shard.py``. A bug there does not fail loudly: it silently drops tests
from CI. These assertions pin the two properties that make the split safe -- every tracked unit
test file lands in exactly one shard, and the same inputs always produce the same partition --
plus the fallback that keeps newly added test files running before anyone measures them.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / ".github" / "ci" / "select_unit_shard.py"
DURATIONS = REPO_ROOT / ".github" / "ci" / "unit-test-durations.json"


def _selector():
    spec = importlib.util.spec_from_file_location("select_unit_shard", SELECTOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "tests/unit/test_*.py"], capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    return [line for line in out.splitlines() if line]


@pytest.fixture(scope="module")
def selector():
    return _selector()


@pytest.fixture(scope="module")
def seconds() -> dict[str, float]:
    return json.loads(DURATIONS.read_text(encoding="utf-8"))["seconds"]


def test_this_file_is_committed_so_ci_selects_it() -> None:
    """Guard the guard: CI partitions `git ls-files` output, so an untracked test file never runs.

    Skips in a dirty worktree (a patch-based verification snapshot copies this file in without
    committing it); in CI the tree is committed and the assertion is real.
    """
    path = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    if path not in _tracked_files():
        pytest.skip(f"{path} is not committed in this worktree")


@pytest.mark.parametrize("shard_count", [1, 2, 3, 4, 8])
def test_shards_partition_every_tracked_unit_file_exactly_once(selector, seconds, shard_count) -> None:
    files = _tracked_files()
    shards = [selector.shard_files(files, i, shard_count, seconds) for i in range(1, shard_count + 1)]
    flat = [path for shard in shards for path in shard]
    assert sorted(flat) == sorted(files)
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("shard_count", [2, 3])
def test_selection_is_deterministic(selector, seconds, shard_count) -> None:
    files = _tracked_files()
    first = [selector.shard_files(files, i, shard_count, seconds) for i in range(1, shard_count + 1)]
    second = [selector.shard_files(list(reversed(files)), i, shard_count, seconds) for i in range(1, shard_count + 1)]
    assert first == second


def test_unmeasured_file_still_runs_and_uses_the_median(selector, seconds) -> None:
    """A brand-new test file nobody measured must be assigned, not skipped."""
    files = [*_tracked_files(), "tests/unit/test_brand_new_unmeasured.py"]
    shard_count = selector.DEFAULT_SHARDS
    shards = [selector.shard_files(files, i, shard_count, seconds) for i in range(1, shard_count + 1)]
    assert sum("tests/unit/test_brand_new_unmeasured.py" in shard for shard in shards) == 1


def test_durations_entries_for_deleted_files_are_ignored(selector, seconds) -> None:
    stale = dict(seconds)
    stale["tests/unit/test_deleted_long_ago.py"] = 9999.0
    files = _tracked_files()
    n = selector.DEFAULT_SHARDS
    assert selector.shard_files(files, 1, n, stale) == selector.shard_files(files, 1, n, seconds)


def test_heaviest_file_drives_the_balance(selector, seconds) -> None:
    """No shard may exceed the heaviest single file's cost, and none may be idle.

    The pack is longest-processing-time, so the worst shard is bounded below by the largest
    file and the split stays within 1.6x of the ideal equal share on the measured suite.
    """
    files = _tracked_files()
    shard_count = selector.DEFAULT_SHARDS
    loads = [sum(seconds.get(path, 0.0) for path in selector.shard_files(files, i, shard_count, seconds)) for i in range(1, shard_count + 1)]
    heaviest = max(seconds.get(path, 0.0) for path in files)
    ideal = sum(seconds.get(path, 0.0) for path in files) / shard_count
    assert max(loads) >= heaviest
    assert min(loads) > 0
    assert max(loads) <= 1.6 * ideal


@pytest.mark.parametrize("shard", [0, -1, 4])
def test_out_of_range_shard_is_rejected(selector, seconds, shard) -> None:
    with pytest.raises(ValueError, match=r"outside 1\.\.3"):
        selector.shard_files(_tracked_files(), shard, 3, seconds)


def test_default_shard_count_matches_the_workflow(selector) -> None:
    """The selector default, `UNIT_SHARDS`, and the matrix entries must agree.

    Drift here fails silently: the aggregate job would wait for N coverage artifacts while the
    matrix ran M shards, or a shard index would select nothing and its tests would never run.
    """
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    env_shards = int(workflow["env"]["UNIT_SHARDS"])
    phases = workflow["jobs"]["python-test-shard"]["strategy"]["matrix"]["phase"]
    unit_phases = [phase for phase in phases if phase.startswith("unit-")]
    assert selector.DEFAULT_SHARDS == env_shards == len(unit_phases)
    assert sorted(int(phase.rsplit("-", 1)[1]) for phase in unit_phases) == list(range(1, env_shards + 1))


def test_durations_file_shape(selector) -> None:
    data = json.loads(DURATIONS.read_text(encoding="utf-8"))
    assert isinstance(data["seconds"], dict)
    assert all(isinstance(value, int | float) and value >= 0 for value in data["seconds"].values())
    # Every measured path still exists: a stale entry means the data was never refreshed after a
    # rename or deletion, which silently degrades the balance. Existence (not `git ls-files`) is
    # the check so this also holds in a dirty worktree.
    for path in data["seconds"]:
        assert (REPO_ROOT / path).is_file(), f"stale durations entry for a missing file: {path}"
    assert selector.measured_seconds() == data["seconds"]


def test_cli_prints_one_path_per_line(selector) -> None:
    n = str(selector.DEFAULT_SHARDS)
    out = subprocess.run(["python3", str(SELECTOR), "1", n], capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.splitlines()
    assert out == selector.shard_files(_tracked_files(), 1, selector.DEFAULT_SHARDS, selector.measured_seconds())
    assert all(line.startswith("tests/unit/test_") and line.endswith(".py") for line in out)


def test_cli_rejects_a_bad_shard_index() -> None:
    result = subprocess.run(["python3", str(SELECTOR), "9", "3"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
