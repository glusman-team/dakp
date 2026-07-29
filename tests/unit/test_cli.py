from __future__ import annotations

import subprocess
import sys

import pytest

from dakp_pipeline.cli import build_parser, main


def test_help_lists_run_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run", "--help"])
    assert exc_info.value.code == 0


def test_module_dakp_help_exits_zero() -> None:
    # Top-level help must list the `run` subcommand (no airflow import needed).
    proc = subprocess.run([sys.executable, "-m", "dakp_pipeline.cli", "--help"], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "run" in proc.stdout

    # `run --help` exposes the profile/workdir flags.
    proc_run = subprocess.run([sys.executable, "-m", "dakp_pipeline.cli", "run", "--help"], check=False, capture_output=True, text=True)
    assert proc_run.returncode == 0, proc_run.stderr
    assert "--profile" in proc_run.stdout
    assert "--workdir" in proc_run.stdout


def test_mock_profile_requires_fixture_root() -> None:
    # The fixture-root requirement is enforced in main() (not argparse), as SystemExit(2).
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--profile", "mock"])
    assert exc_info.value.code == 2


# End-to-end run_pipeline behavior (with real fixtures + monkeypatching) is covered by
# tests/integration/test_mock_pipeline.py.
