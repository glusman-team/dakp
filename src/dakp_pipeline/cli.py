"""``dakp`` command-line runner — one command runs the whole Airflow-native pipeline.

Replaces the retired ``scripts/dakp_up.sh`` / ``scripts/dakp_down.sh`` shell orchestrators and the
``Makefile`` / direnv ``.envrc`` setup with a single self-contained cyclopts CLI (``uv run dakp …``).
Everything the run needs is built in: locations (workdir, Airflow home, fixture root) and scope are
hardcoded constants derived from the repo root, and the CLI reads **no environment variables**. The
only inputs are the ``profile`` positional plus a few short-aliased flags (``--fullmap/-f``,
``--port/-p``, ``--log-level/-l``, ``--detach/-d``).

Commands::

    uv run dakp up [profile]   # build+pack the Go bundle, start Airflow, run dakp_build, wait
    uv run dakp down           # stop the local Airflow started by `up`
    uv run dakp clean          # remove caches, coverage data, tmp/, and the Go worker binary

``up`` is a faithful Python port of ``dakp_up.sh``: preflight-verifies the Airflow install
(self-heals a corrupt venv), builds + packs the native Go bundle, starts Airflow standalone with the
ExecutableCoordinator configured, sets the ``dakp_config`` Variable + task pools, triggers the
``dakp_build`` DAG, and waits for it to finish (unless ``--detach``). The fullmap redb is never
downloaded — ``--fullmap <path>`` points at a prebuilt redb (required for the real ``prod`` handoff).

Every side effect goes through a small module-level function (``run_subprocess``, ``api_up``,
``dag_registered``, ``run_state``, ``start_standalone``, ``sleep``, …) so the tests monkeypatch the
boundary and exercise every branch with no real Airflow / Go / network — the same convention as
:mod:`dakp_pipeline.tablassert.run`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Annotated, Any, Literal

from cyclopts import App, Parameter

from dakp_pipeline.config import load_profile

#: Repository root (``src/dakp_pipeline/cli.py`` -> two levels up).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# --- hardcoded run locations + defaults (not flags, not env vars) ----------------
_DEFAULT_WORKDIR = _REPO_ROOT / "tmp" / "airflow-run" / "data"
_DEFAULT_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "pipeline"
_DEFAULT_AIRFLOW_HOME = _REPO_ROOT / "tmp" / "airflow-home"
_DEFAULT_PORT = 8090  # 8080 is commonly taken (e.g. by the aoe daemon on dev hosts)
_DEFAULT_LOG_LEVEL = "INFO"

# Poll budgets (matches the retired bash orchestrator).
_API_WAIT_ROUNDS = 90
_API_WAIT_SECONDS = 2
_DAG_WAIT_ROUNDS = 45
_DAG_WAIT_SECONDS = 2
_RUN_WAIT_ROUNDS = 300
_RUN_WAIT_SECONDS = 3

ProfileName = Literal["mock", "sample", "prod"]


# --- side-effect boundary (monkeypatch points for tests) --------------------------


def run_subprocess(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``command`` capturing stdout/stderr, never raising on non-zero exit (monkeypatch point)."""
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def airflow_importable() -> bool:
    """True when ``import airflow`` would succeed here (the preflight health probe)."""
    return importlib.util.find_spec("airflow") is not None


def api_up(base_url: str) -> bool:
    """True when the Airflow API answers ``GET /api/v2/version`` (stdlib urllib; no ``requests``)."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/v2/version", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def dag_registered(db_path: Path, dag_id: str) -> bool:
    """True when ``dag_id`` is present in Airflow's sqlite metadata db (DAG parsed + registered)."""
    try:
        con = sqlite3.connect(str(db_path))
        try:
            count = con.execute("SELECT count(*) FROM dag WHERE dag_id = ?", (dag_id,)).fetchone()[0]
            return bool(count)
        finally:
            con.close()
    except Exception:
        return False


def run_state(db_path: Path, dag_id: str) -> str:
    """State of the most recent ``dag_run`` for ``dag_id`` (``success``/``failed``/… or ``none``)."""
    try:
        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute("SELECT state FROM dag_run WHERE dag_id = ? ORDER BY run_after DESC LIMIT 1", (dag_id,)).fetchone()
            return str(row[0]) if row and row[0] else "none"
        finally:
            con.close()
    except Exception:
        return "none"


def start_standalone(log_path: Path, env: dict[str, str]) -> int:
    """Launch ``airflow standalone`` in the background (stdout/stderr -> ``log_path``); return its pid."""
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(["uv", "run", "airflow", "standalone"], stdout=log, stderr=subprocess.STDOUT, env=env)
    return int(proc.pid)


def sleep(seconds: float) -> None:
    """Thin ``time.sleep`` wrapper so poll loops are instant under test (monkeypatch point)."""
    time.sleep(seconds)


def pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` exists (signal 0 probe)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate(pid: int) -> None:
    """Send SIGTERM to ``pid`` (best-effort; a race-lost process is already gone)."""
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)


def _tail(path: Path, lines: int = 40) -> None:
    """Print the last ``lines`` of ``path`` (for diagnosing a failed Airflow startup)."""
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(text[-lines:]))


# --- orchestration (faithful port of scripts/dakp_up.sh) --------------------------


def _airflow_env(airflow_home: Path, bundle_dir: Path, port: int) -> dict[str, str]:
    """The Airflow process environment: ``AIRFLOW__<SECTION>__<KEY>`` config merged over os.environ."""
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env.update(
        {
            "AIRFLOW_HOME": str(airflow_home),
            "AIRFLOW__CORE__DAGS_FOLDER": str(_REPO_ROOT / "src" / "dakp_pipeline" / "dags"),
            "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
            "AIRFLOW__CORE__EXECUTOR": "LocalExecutor",
            "AIRFLOW__API__PORT": str(port),
            "AIRFLOW__API__HOST": "127.0.0.1",
            # MANDATORY: without this the task supervisor's Execution-API client defaults to localhost:8080.
            "AIRFLOW__CORE__EXECUTION_API_SERVER_URL": f"{base_url}/execution/",
            "AIRFLOW__SDK__COORDINATORS": json.dumps(
                {"go": {"classpath": "airflow.sdk.coordinators.executable.ExecutableCoordinator", "kwargs": {"executables_root": [str(bundle_dir)]}}}
            ),
            "AIRFLOW__SDK__QUEUE_TO_COORDINATOR": json.dumps({"golang": "go"}),
        }
    )
    return env


def _preflight() -> int:
    """Verify the Airflow install is intact, self-healing a corrupt venv. Returns 0 when healthy.

    A corrupted venv (bit-rot on a RAID-backed disk or a tainted wheel in the uv cache) makes
    ``import airflow`` die with a cryptic yaml ReaderError. Reinstall the owning package
    (``apache-airflow-core``, which ships the file — NOT the ``apache-airflow`` meta-package) instead
    of crashing; if a corrupt uv cache re-supplies the bad bytes, drop the cached copy and re-fetch.
    """
    if airflow_importable():
        return 0
    print(">>> [preflight] Airflow install is broken; reinstalling apache-airflow-core")
    run_subprocess(["uv", "sync", "--reinstall-package", "apache-airflow-core"])
    if not airflow_importable():
        run_subprocess(["uv", "cache", "clean", "apache-airflow-core"])
        run_subprocess(["uv", "sync", "--reinstall-package", "apache-airflow-core"])
    if not airflow_importable():
        print("!!! Airflow still fails to import after reinstall; inspect .venv (try: uv sync --reinstall)")
        return 1
    return 0


def run_up(*, profile: str, fullmap: str | None, port: int, log_level: str, detach: bool) -> int:
    """Run the pipeline end-to-end on ``profile`` via a local Airflow. Returns a process exit code."""
    # Fast-fail: the real (prod) Tablassert handoff needs a fullmap; DAKP never downloads one.
    if load_profile(profile).run_tablassert and not fullmap:
        print(f"!!! profile {profile!r} runs the real Tablassert handoff, which needs a fullmap redb: pass --fullmap <path>")
        return 2

    workdir = _DEFAULT_WORKDIR
    fixture_root = _DEFAULT_FIXTURE_ROOT
    airflow_home = _DEFAULT_AIRFLOW_HOME
    bundle_dir = airflow_home / "executable-bundles"
    base_url = f"http://127.0.0.1:{port}"
    log_path = airflow_home / "standalone.log"
    pidfile = airflow_home / "standalone.pid"
    db_path = airflow_home / "airflow.db"
    airflow_home.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    # --- 0. preflight: verify the Airflow install (self-heal if corrupt) ---------
    # Runs BEFORE any airflow import so a corrupt venv is healed rather than crashing on import.
    preflight = _preflight()
    if preflight != 0:
        return preflight

    # Imported here (not at module load) so `dakp --help` / `down` / `clean` stay Airflow-free, and
    # only after the preflight so a just-healed install imports cleanly.
    from dakp_pipeline.dags.dakp_build import CONFIG_VARIABLE, DAG_ID, DOWNLOAD_POOL, EXTRACT_POOL

    env = _airflow_env(airflow_home, bundle_dir, port)

    # --- 1. build + pack the Go bundle into executables_root --------------------
    print(">>> [1/6] building + packing the native Go bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pack = run_subprocess(
        ["go", "tool", "airflow-go-pack", "--output", str(bundle_dir / "dakp-bundle"), "./cmd/dakp-bundle"], cwd=_REPO_ROOT / "go", env=env
    )
    if pack.returncode != 0:
        print("!!! bundle pack failed")
        if pack.stderr:
            print(pack.stderr.strip()[-2000:])
        return 1

    # --- 2. start Airflow standalone (reuse if already up) ----------------------
    if api_up(base_url):
        print(f">>> [2/6] Airflow already running on :{port} (reusing)")
    else:
        print(f">>> [2/6] starting Airflow standalone on :{port} (logs: {log_path})")
        pidfile.write_text(str(start_standalone(log_path, env)), encoding="utf-8")
        for _ in range(_API_WAIT_ROUNDS):
            if api_up(base_url):
                break
            sleep(_API_WAIT_SECONDS)
        if not api_up(base_url):
            print(f"!!! API server did not come up; tail of {log_path}:")
            _tail(log_path)
            return 1

    # --- 3. wait for the DAG to register, then unpause + provision pools --------
    print(">>> [3/6] waiting for DAG registration")
    for _ in range(_DAG_WAIT_ROUNDS):
        if dag_registered(db_path, DAG_ID):
            break
        sleep(_DAG_WAIT_SECONDS)
    if not dag_registered(db_path, DAG_ID):
        print(f"!!! DAG {DAG_ID} never registered; check {log_path}")
        return 1
    run_subprocess(["uv", "run", "airflow", "dags", "unpause", DAG_ID], env=env)
    # The concurrency-bounding pools are not auto-created; tasks on a missing pool never schedule.
    run_subprocess(["uv", "run", "airflow", "pools", "set", DOWNLOAD_POOL, "4", "Concurrent source downloads (network I/O)"], env=env)
    run_subprocess(["uv", "run", "airflow", "pools", "set", EXTRACT_POOL, "4", "Concurrent raw->interim extracts (Go bundle)"], env=env)

    # --- 4. set the per-run config Variable (shared by Python tasks + Go bundle) -
    # null quarter/release limits => profile default (prod = unbounded full build).
    config: dict[str, Any] = {
        "workdir": str(workdir),
        "profile": profile,
        "fixture_root": str(fixture_root),
        "quarter_limit": None,
        "release_limit": None,
        "log_level": log_level,
        "fullmap": fullmap,
    }
    print(f">>> [4/6] setting {CONFIG_VARIABLE} Variable (profile={profile} workdir={workdir})")
    run_subprocess(["uv", "run", "airflow", "variables", "set", CONFIG_VARIABLE, json.dumps(config)], env=env)

    # --- 5. trigger the DAG run -------------------------------------------------
    print(f">>> [5/6] triggering {DAG_ID}")
    trigger = run_subprocess(["uv", "run", "airflow", "dags", "trigger", DAG_ID], env=env)
    if trigger.returncode != 0:
        print("!!! trigger failed")
        if trigger.stderr:
            print(trigger.stderr.strip()[-2000:])
        return 1

    if detach:
        print(f">>> triggered {DAG_ID} (detached) — watch {base_url} ; logs: {log_path}")
        return 0

    # --- 6. wait for completion -------------------------------------------------
    print(">>> [6/6] waiting for the run to finish")
    final = ""
    for i in range(1, _RUN_WAIT_ROUNDS + 1):
        state = run_state(db_path, DAG_ID)
        print(f"    [{i}] run_state={state}")
        if state == "success":
            final = "success"
            break
        if state == "failed":
            final = "failed"
            break
        sleep(_RUN_WAIT_SECONDS)

    print()
    summary = workdir / "data" / "reports" / "build_summary.json"
    if final == "success":
        print(f">>> SUCCESS — build summary: {summary}")
        if summary.exists():
            print(summary.read_text(encoding="utf-8"))
        return 0
    print(f"!!! DAG run did not succeed (final={final or 'timeout'}). Inspect: {log_path} and {airflow_home / 'logs'}/")
    return 1


# --- down (faithful port of scripts/dakp_down.sh) ---------------------------------


def run_down() -> int:
    """Stop the local Airflow standalone started by :func:`run_up`. Returns a process exit code."""
    pidfile = _DEFAULT_AIRFLOW_HOME / "standalone.pid"
    if pidfile.exists():
        pid_text = pidfile.read_text(encoding="utf-8").strip()
        if pid_text.isdigit() and pid_alive(int(pid_text)):
            print(f">>> stopping Airflow standalone (pid {pid_text})")
            terminate(int(pid_text))
            sleep(5)
        pidfile.unlink(missing_ok=True)
    # Reap lingering standalone components. Patterns match only Airflow processes (the aoe daemon
    # that owns 8080 is "aoe", never "airflow").
    for pattern in ("airflow standalone", "airflow api_server", "airflow scheduler", "airflow triggerer", "airflow worker", "airflow serve-logs"):
        run_subprocess(["pkill", "-TERM", "-f", pattern])
    sleep(2)
    run_subprocess(["pkill", "-9", "-f", "airflow"])  # catch-all for reparented children
    sleep(1)
    print(">>> Airflow stopped")
    return 0


# --- clean (port of `make clean`) -------------------------------------------------


def run_clean() -> int:
    """Remove caches, coverage data, ``tmp/``, stray ``__pycache__`` dirs, and the Go worker binary."""
    for name in (".pytest_cache", ".ruff_cache", ".coverage", "htmlcov", "tmp"):
        target = _REPO_ROOT / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink(missing_ok=True)
    for pycache in _REPO_ROOT.rglob("__pycache__"):
        if ".venv" not in pycache.parts:
            shutil.rmtree(pycache, ignore_errors=True)
    (_REPO_ROOT / "go" / "dakp-worker").unlink(missing_ok=True)
    print(">>> cleaned caches, coverage data, tmp/, and the Go worker binary")
    return 0


# --- cyclopts app (console-script entry point: `dakp = dakp_pipeline.cli:app`) -----

app = App(name="dakp", help="DAKP pipeline runner — one command runs the whole Airflow-native pipeline.")


@app.command
def up(
    profile: Annotated[ProfileName, Parameter(name="profile")] = "mock",
    *,
    fullmap: Annotated[str | None, Parameter(name=["--fullmap", "-f"])] = None,
    port: Annotated[int, Parameter(name=["--port", "-p"])] = _DEFAULT_PORT,
    log_level: Annotated[str, Parameter(name=["--log-level", "-l"])] = _DEFAULT_LOG_LEVEL,
    detach: Annotated[bool, Parameter(name=["--detach", "-d"])] = False,
) -> None:
    """Run the pipeline end-to-end on PROFILE via a local Airflow (build Go bundle, trigger, wait).

    PROFILE selects sources + concurrency + the Tablassert handoff mode: ``mock`` (fixtures; no
    network; the default), ``sample`` (real sources, bounded), ``prod`` (real full build; needs
    ``--fullmap``). ``--detach`` triggers and returns immediately instead of waiting.
    """
    raise SystemExit(run_up(profile=profile, fullmap=fullmap, port=port, log_level=log_level, detach=detach))


@app.command
def down() -> None:
    """Stop the local Airflow started by ``up``."""
    raise SystemExit(run_down())


@app.command
def clean() -> None:
    """Remove caches, coverage data, ``tmp/``, and the Go worker binary."""
    raise SystemExit(run_clean())


__all__ = [
    "airflow_importable",
    "api_up",
    "app",
    "dag_registered",
    "pid_alive",
    "run_clean",
    "run_down",
    "run_state",
    "run_subprocess",
    "run_up",
    "sleep",
    "start_standalone",
    "terminate",
]
