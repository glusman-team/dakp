"""Command-line entrypoint: ``dakp run --profile mock ...``.

Uses stdlib :mod:`argparse` (not cyclopts) to keep the base dependency set minimal, per
the task brief. The ``run`` subcommand delegates to :func:`dakp_pipeline.pipeline.run_pipeline`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dakp_pipeline import __version__
from dakp_pipeline.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dakp",
        description=(
            "Drug Approvals Knowledge Provider pipeline. Builds treatment and "
            "contraindication assertion tables from DailyMed, Drugs@FDA, FAERS, and MEDI."
        ),
    )
    parser.add_argument("--version", action="version", version=f"dakp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    run = sub.add_parser("run", help="Run the pipeline (mock profile needs no network).")
    run.add_argument("--profile", default="mock", choices=("mock", "sample", "prod"), help="Execution profile (default: mock).")
    run.add_argument("--fixture-root", type=Path, default=None, help="Directory of mock source fixtures (required for the mock profile).")
    run.add_argument(
        "--workdir", type=Path, default=Path("data"), help="Pipeline workdir root; all artifacts are written under here (default: ./data)."
    )
    run.add_argument(
        "--run-airflow", action="store_true", help="Execute via the Airflow DAG instead of the pure-Python runner (requires airflow extra)."
    )
    run.add_argument(
        "--quarter-limit", type=int, default=None, help="Cap FAERS quarters processed (overrides the profile; e.g. 1 for a bounded smoke run)."
    )
    run.add_argument(
        "--release-limit",
        type=int,
        default=None,
        help="Cap DailyMed full releases processed (overrides the profile; e.g. 1 for a bounded smoke run).",
    )
    run.add_argument("--force", action="store_true", help="Ignore cached artifacts and rerun every stage (overrides the profile).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.profile == "mock" and args.fixture_root is None:
            parser.error("--fixture-root is required for the mock profile")
        # Scope/force overrides forwarded to the profile via run_pipeline's params merge.
        params: dict[str, object] = {}
        if args.quarter_limit is not None:
            params["quarter_limit"] = args.quarter_limit
        if args.release_limit is not None:
            params["release_limit"] = args.release_limit
        if args.force:
            params["force"] = True
        result = run_pipeline(
            profile=args.profile, fixture_root=args.fixture_root, workdir=args.workdir, run_airflow=args.run_airflow, params=params or None
        )
        summary = result.build_summary
        print(f"Pipeline complete: workdir={result.workdir.root}")
        if summary is not None:
            print(f"Build summary: {summary}")
        return 0

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
