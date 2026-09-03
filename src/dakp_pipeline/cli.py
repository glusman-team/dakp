"""``dakp`` command-line runner — one command runs the whole Airflow-native pipeline.

Replaces the retired ``scripts/dakp_up.sh`` / ``scripts/dakp_down.sh`` shell orchestrators and the
``Makefile`` / direnv ``.envrc`` setup with a single self-contained cyclopts CLI (``uv run dakp …``).
Everything the run needs is built in: locations (workdir, Airflow home, fixture root) and scope are
hardcoded constants derived from the repo root, and the CLI reads **no environment variables**. The
only inputs are a few short-aliased flags (``--fullmap/-f``, ``--port/-p``, ``--log-level/-l``,
``--detach/-d``).

Commands::

    uv run dakp up             # build+pack the Go bundle, start Airflow, run dakp_pipeline, wait
    uv run dakp down           # stop the local Airflow started by `up`
    uv run dakp clean          # stop a live NER cache server, then remove caches, coverage data, tmp/, and the Go worker binary
    uv run dakp export-medliner --out <dir>   # export the MEDliNER training-data bundle (--fixtures = offline path)

``up`` is a faithful Python port of ``dakp_up.sh``: preflight-verifies the Airflow install
(self-heals a corrupt venv), builds + packs the native Go bundle, starts Airflow standalone with the
ExecutableCoordinator configured, sets the ``dakp_config`` Variable + task pools, triggers the
``dakp_pipeline`` DAG, and waits for it to finish (unless ``--detach``). The fullmap redb is never
downloaded — ``--fullmap <path>`` points at a prebuilt redb and triggers the real Tablassert
handoff (without it the handoff is deferred, never an error).

Every side effect goes through a small module-level function (``run_subprocess``, ``api_up``,
``dag_registered``, ``run_state``, ``start_standalone``, ``sleep``, …) so the tests monkeypatch the
boundary and exercise every branch with no real Airflow / Go / network — the same convention as
:mod:`dakp_pipeline.tablassert`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import App, Parameter

if TYPE_CHECKING:
    from dakp_pipeline.io.contracts import ArtifactRef, TaskContext

#: Repository root (``src/dakp_pipeline/cli.py`` -> two levels up).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# --- hardcoded run locations + defaults (not flags, not env vars) ----------------
_DEFAULT_WORKDIR = _REPO_ROOT / "tmp"
_DEFAULT_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "pipeline"
_DEFAULT_AIRFLOW_HOME = _REPO_ROOT / "tmp" / "airflow"
_DEFAULT_PORT = 8090  # 8080 is commonly taken (e.g. by the aoe daemon on dev hosts)
_DEFAULT_LOG_LEVEL = "INFO"
_SMALL_SCOPE = 1  # `--small` bounds scope to ~1 FAERS quarter + 1 DailyMed release (the ONE surviving integer knob)
# A stored DailyMed release younger than this (days) is reused without re-download; DailyMed
# replaces its fixed-name full-release ZIPs in place, so without the gate every new release
# re-downloads the whole snapshot (~tens of GB). <= 0 disables the gate (always re-check).
_DAILYMED_MAX_AGE_DAYS = 7

# Poll budgets (matches the retired bash orchestrator).
_API_WAIT_ROUNDS = 90
_API_WAIT_SECONDS = 2
_DAG_WAIT_ROUNDS = 45
_DAG_WAIT_SECONDS = 2
_RUN_WAIT_ROUNDS = 1200  # 60 min @ 3s: a release's GLiNER contraindication mining + KG build can take ~20+ min
_RUN_WAIT_SECONDS = 3

#: Heal steps cap uv's cache-lock wait: a stray long-running ``uv run`` child (e.g. an orphaned
#: ``airflow standalone``) holds uv's cache lock for its whole lifetime, and uv's 300s default
#: would stall ``dakp up`` silently while nothing is healed.
_UV_HEAL_LOCK_TIMEOUT_SECONDS = 60

# --- side-effect boundary (monkeypatch points for tests) --------------------------


def run_subprocess(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``command`` capturing stdout/stderr, never raising on non-zero exit (monkeypatch point)."""
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def airflow_importable() -> bool:
    """True when ``import airflow`` really succeeds here (the preflight health probe).

    Runs a real import in a subprocess: ``find_spec`` only locates the package and passes even when
    shipped data files are corrupt (a bit-rotted ``config_templates/config.yml`` kills
    ``import airflow`` deep inside YAML parsing). Subprocess so a broken install never poisons this
    process and a just-healed install is re-tested against fresh bytes.
    """
    return run_subprocess([sys.executable, "-c", "import airflow"]).returncode == 0


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
            # The Go extractors read their upstream acquisition XComs via the Execution API; under
            # real-data load the task-SDK default 5s timeout died with ReadTimeout (extract_faers
            # died even at 300s while the API server was busy). acquire_dailymed now pushes ONE
            # refs-file ref (its tens of thousands of SPL member refs live in a single store
            # JSON), but the API server still slows under full-build load — keep the headroom.
            "AIRFLOW__WORKERS__EXECUTION_API_TIMEOUT": "1000",
            "AIRFLOW__SDK__COORDINATORS": json.dumps(
                {"go": {"classpath": "airflow.sdk.coordinators.executable.ExecutableCoordinator", "kwargs": {"executables_root": [str(bundle_dir)]}}}
            ),
            "AIRFLOW__SDK__QUEUE_TO_COORDINATOR": json.dumps({"golang": "go"}),
            # --- Airflow UI/task-log settings ---------------------------------------
            # Identify this orchestrator so it is distinguishable from any other local Airflow...
            "AIRFLOW__API__INSTANCE_NAME": "DAKP",
            # ...and wrap task logs by default: the native Go workers log one wide slog JSON
            # record per line, and without wrap the task-log view horizontal-scrolls instead.
            "AIRFLOW__API__DEFAULT_WRAP": "True",
        }
    )
    return env


def _run_heal(command: list[str]) -> None:
    """Run one preflight self-heal step with a bounded cache-lock wait; print stderr on failure.

    Never raises: a failed heal step (e.g. uv's cache lock held by a stray ``uv run`` child) must
    surface its stderr and still let the rest of the heal ladder run.
    """
    result = run_subprocess(command, env={**os.environ, "UV_LOCK_TIMEOUT": str(_UV_HEAL_LOCK_TIMEOUT_SECONDS)})
    if result.returncode == 0:
        return
    print(f"!!! heal step failed (rc={result.returncode}): {' '.join(command)}")
    if result.stderr:
        print(result.stderr.strip()[-2000:])


def _preflight() -> int:
    """Verify the Airflow install is intact, self-healing a corrupt venv. Returns 0 when healthy.

    A corrupted venv (bit-rot on a RAID-backed disk or a tainted wheel in the uv cache) makes
    ``import airflow`` die with a cryptic yaml ReaderError. Reinstall the owning package
    (``apache-airflow-core``, which ships the file — NOT the ``apache-airflow`` meta-package) instead
    of crashing; if a corrupt uv cache re-supplies the bad bytes, drop the cached copy and re-fetch.
    Heal-step failures print their stderr (a common one: uv's cache lock held by a stray
    ``uv run airflow standalone`` — ``dakp down`` releases it); if nothing heals, the bail prints
    the actual ``import airflow`` error.
    """
    if airflow_importable():
        return 0
    print(">>> [preflight] Airflow install is broken; reinstalling apache-airflow-core")
    _run_heal(["uv", "sync", "--reinstall-package", "apache-airflow-core"])
    if not airflow_importable():
        _run_heal(["uv", "cache", "clean", "apache-airflow-core"])
        _run_heal(["uv", "sync", "--reinstall-package", "apache-airflow-core"])
    if not airflow_importable():
        probe = run_subprocess([sys.executable, "-c", "import airflow"])
        print("!!! Airflow still fails to import after reinstall; import error:")
        if probe.stderr:
            print(probe.stderr.strip()[-2000:])
        print("!!! inspect .venv (try: uv sync --reinstall); if the error mentions a uv lock, run `dakp down` — a stray Airflow holds it")
        return 1
    return 0


def run_up(*, fullmap: str | None, port: int, log_level: str, detach: bool, small: bool = False) -> int:
    """Run the pipeline end-to-end via a local Airflow. Returns a process exit code.

    Always runs real acquisition; ``fullmap`` only decides the Tablassert handoff mode (a path
    triggers the real handoff, absent => deferred — never a hard failure). ``small`` bounds the
    acquisition SCOPE to a tiny real subset (~1 FAERS quarter + 1 DailyMed release) — same pipeline,
    less data; it changes nothing else (threads stay ``os.cpu_count()``).
    """
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
    from dakp_pipeline.dags.dakp_build import CONFIG_VARIABLE, DAG_ID, DOWNLOAD_POOL, EXTRACT_POOL, NER_MINING_POOL

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
    # The NER mention-cache server lives at <workdir>/bin/dakp-nercache (the client's default
    # lookup). Optional: a failed build only disables mention caching, never the run.
    nercache_bin = workdir / "bin" / "dakp-nercache"
    nercache_bin.parent.mkdir(parents=True, exist_ok=True)
    nercache = run_subprocess(["go", "build", "-o", str(nercache_bin), "./cmd/dakp-nercache"], cwd=_REPO_ROOT / "go", env=env)
    if nercache.returncode != 0:
        print("!!! dakp-nercache build failed (NER mention caching disabled; non-fatal)")
        if nercache.stderr:
            print(nercache.stderr.strip()[-2000:])

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
    run_subprocess(["uv", "run", "airflow", "pools", "set", NER_MINING_POOL, "1", "Serializes GLiNER mining across shape tasks"], env=env)

    # --- 4. set the per-run config Variable (shared by Python tasks + Go bundle) -
    # `--small` bounds the acquisition scope (quarter/release limit = _SMALL_SCOPE); otherwise null
    # limits => unbounded full build. threads = all cores either way (Go all-cores contract).
    # Source max-age settings keep fixed-name snapshots fresh without repeating unchanged downloads.
    # See sources/dailymed.py and sources/drugsfda.py.
    scope_limit = _SMALL_SCOPE if small else None
    config: dict[str, Any] = {
        "workdir": str(workdir),
        "fixture_root": str(fixture_root),
        "threads": os.cpu_count(),
        "quarter_limit": scope_limit,
        "release_limit": scope_limit,
        "dailymed_max_age_days": _DAILYMED_MAX_AGE_DAYS,
        "drugsfda_max_age_days": _DAILYMED_MAX_AGE_DAYS,
        "force": False,
        # Release mode: `tablassert build-kg --release` emits the slim, significant-only graph
        # (drops biolink:not_significant edges before resolution). Always on for DAKP builds.
        "release": True,
        # QC mode: `tablassert build-kg --qc` runs the SapBERT embedding audit (12.0) plus
        # 13.0's stage-7 fail-the-build assertions on the final NDJSON (empty-or-null-values,
        # unnamed/unidentified nodes, incomplete-edges). Always on for DAKP builds.
        "qc": True,
        # `tablassert build-kg --no-original` (16.1): omit the verbatim `original_*` source-cell
        # copies from final edges. Edge identity no longer depends on them (uuid_fields is the
        # resolved statement only), so they are dead weight in the published graph. Always on.
        "no_original": True,
        # `tablassert build-kg --threads 70`: worker count for the parallel fullmap reads behind
        # entity resolution. Fixed at 70 for the wenceslaus build host (leaves headroom for the
        # Airflow workers + Go extractors; Tablassert's auto would claim every core).
        "tablassert_threads": 70,
        "log_level": log_level,
        "fullmap": fullmap,
    }
    print(f">>> [4/6] setting {CONFIG_VARIABLE} Variable (workdir={workdir})")
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
    from dakp_pipeline.paths import Workdir

    summary = Workdir(workdir).reports / "build_summary.json"
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
    """Remove caches, coverage data, ``tmp/``, stray ``__pycache__`` dirs, and the Go worker binary.

    A live ``dakp-nercache`` server (``<workdir>/cache/ner/server.json`` pid probe) is SIGTERMed
    first — Pebble holds an exclusive directory lock, so deleting under a running server would
    corrupt nothing but the server would keep serving the deleted store. If the server survives
    the SIGTERM, the clean is refused.
    """
    server_file = _DEFAULT_WORKDIR / "cache" / "ner" / "server.json"
    if server_file.exists():
        try:
            pid = int(json.loads(server_file.read_text(encoding="utf-8")).get("pid", 0))
        except (OSError, ValueError, AttributeError):
            pid = 0
        if pid and pid_alive(pid):
            print(f">>> stopping live dakp-nercache server (pid {pid})")
            terminate(pid)
            sleep(2)
            if pid_alive(pid):
                print(f"!!! dakp-nercache pid {pid} survived SIGTERM; refusing to clean while it serves {_DEFAULT_WORKDIR / 'cache' / 'ner'}")
                return 1
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


# --- export-medliner (MEDliNER training-data bundle; offline-capable) -----------

#: The interim tables the MEDliNER export consumes, relative to ``Workdir.interim``.
_EXPORT_INTERIM_TABLES: tuple[str, ...] = ("dailymed/spl_documents.parquet", "faers/cases.parquet")
#: The DailyMed fixture the offline ``--fixtures`` path extracts (the FAERS side takes every
#: ``faers/*.txt`` family, exactly the harness fixture set).
_FIXTURE_DAILYMED_SPL = "dailymed/dailymed_spl.xml.gz"


def export_interim_refs(workdir: Path) -> list[ArtifactRef]:
    """Refs for the already-extracted interim tables the MEDliNER export consumes.

    Raises ``FileNotFoundError`` naming EVERY missing table when the workdir is not materialized:
    this command never runs acquisition or extraction, so it never triggers a download.
    """
    from dakp_pipeline.io.content_hash import hash_file
    from dakp_pipeline.io.contracts import ArtifactRef
    from dakp_pipeline.io.downloads import infer_media_type
    from dakp_pipeline.paths import Workdir

    interim = Workdir(workdir).interim
    missing = [name for name in _EXPORT_INTERIM_TABLES if not (interim / name).exists()]
    if missing:
        msg = f"export-medliner: missing interim table(s) under {interim}: {', '.join(missing)} \u2014 run `dakp up` first (this command never downloads)"
        raise FileNotFoundError(msg)
    return [
        ArtifactRef(uri=interim / name, blake3=hash_file(interim / name), media_type=infer_media_type(interim / name))
        for name in _EXPORT_INTERIM_TABLES
    ]


def extract_fixture_sources(ctx: TaskContext, fixture_root: Path) -> list[ArtifactRef]:
    """Run the pure-Python reference extractors over the committed fixture pipeline (offline).

    The same fixture plumbing the integration harness uses: the single DailyMed SPL fixture plus
    every FAERS ``.txt`` family under ``fixture_root/faers`` (no Drugs@FDA — the export consumes
    only the DailyMed + FAERS extracts).
    """
    from dakp_pipeline.extract import faers_ascii, spl_xml

    dailymed_raw = [ctx.fixture(_FIXTURE_DAILYMED_SPL)]
    faers_raw = [ctx.fixture(f"faers/{path.name}") for path in sorted((fixture_root / "faers").glob("*.txt"))]
    return [*spl_xml.extract(dailymed_raw, ctx), *faers_ascii.extract(faers_raw, ctx)]


def copy_export_bundle(src_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Copy the three MEDliNER bundle files into ``out_dir``, overwriting ONLY those files."""
    from dakp_pipeline import medliner_export

    out_dir.mkdir(parents=True, exist_ok=True)
    names = (medliner_export.MANIFEST_FILENAME, medliner_export.CANDIDATES_FILENAME, medliner_export.GOLD_FILENAME)
    return {name: Path(shutil.copyfile(src_dir / name, out_dir / name)) for name in names}


def run_export_medliner(*, out: str | None = None, workdir: str | None = None, fixtures: bool = False) -> int:
    """Export the MEDliNER training-data bundle from the DailyMed + FAERS extracts.

    Returns a process exit code. Default mode reads the already-extracted interim tables from a
    materialized ``workdir`` and NEVER downloads — a missing table is a loud error naming it.
    ``fixtures`` first runs the pure-Python reference extractors over the committed pipeline
    fixtures (the fully offline path). The bundle lands under
    ``<workdir>/store/medliner-export`` and is copied to ``out`` when it points elsewhere.
    """
    from dakp_pipeline import medliner_export
    from dakp_pipeline.io.contracts import TaskContext
    from dakp_pipeline.logging_setup import configure_logging
    from dakp_pipeline.paths import Workdir

    workdir_root = Path(workdir) if workdir is not None else _DEFAULT_WORKDIR
    wd = Workdir(workdir_root)
    wd.create()
    configure_logging(wd.root, level=_DEFAULT_LOG_LEVEL, for_airflow=False)
    ctx = TaskContext(workdir=wd.root, fixture_root=_DEFAULT_FIXTURE_ROOT if fixtures else None, params={})
    if fixtures:
        refs = extract_fixture_sources(ctx, _DEFAULT_FIXTURE_ROOT)
    else:
        try:
            refs = export_interim_refs(workdir_root)
        except FileNotFoundError as exc:
            print(f"!!! {exc}")
            return 1
    medliner_export.export(refs, ctx)
    src_dir = wd.store / medliner_export.OUT_DIRNAME
    out_dir = Path(out) if out is not None else src_dir
    if out_dir != src_dir:
        copy_export_bundle(src_dir, out_dir)
    print(f">>> MEDliNER training-data bundle ready: {out_dir}")
    return 0


# --- cyclopts app (console-script entry point: `dakp = dakp_pipeline.cli:app`) -----

app = App(name="dakp", help="DAKP pipeline runner — one command runs the whole Airflow-native pipeline.")


@app.command
def up(
    *,
    fullmap: Annotated[str | None, Parameter(name=["--fullmap", "-f"])] = None,
    small: Annotated[bool, Parameter(name=["--small", "-s"])] = False,
    port: Annotated[int, Parameter(name=["--port", "-p"])] = _DEFAULT_PORT,
    log_level: Annotated[str, Parameter(name=["--log-level", "-l"])] = _DEFAULT_LOG_LEVEL,
    detach: Annotated[bool, Parameter(name=["--detach", "-d"])] = False,
) -> None:
    """Run the pipeline end-to-end via a local Airflow (build Go bundle, trigger, wait).

    Always runs real acquisition. ``--fullmap <path>`` triggers the real Tablassert handoff
    (without it the handoff is deferred, never an error). ``--small`` runs a bounded real-data dev
    run (~1 FAERS quarter + 1 DailyMed release) — same pipeline, less data. ``--detach`` triggers
    and returns immediately instead of waiting.
    """
    raise SystemExit(run_up(fullmap=fullmap, port=port, log_level=log_level, detach=detach, small=small))


@app.command
def export_medliner(
    *,
    out: Annotated[str | None, Parameter(name=["--out", "-o"])] = None,
    workdir: Annotated[str | None, Parameter(name=["--workdir", "-w"])] = None,
    fixtures: Annotated[bool, Parameter(name=["--fixtures"])] = False,
) -> None:
    """Export the MEDliNER training-data bundle from the DailyMed + FAERS extracts.

    Default mode exports from a materialized workdir (after ``dakp up``) and never downloads —
    missing interim tables are a loud error naming them. ``--fixtures`` first runs the
    pure-Python reference extractors over the committed pipeline fixtures (fully offline); it is
    also the documented way to regenerate MEDliNER's committed sample bundle.
    """
    raise SystemExit(run_export_medliner(out=out, workdir=workdir, fixtures=fixtures))


@app.command
def down() -> None:
    """Stop the local Airflow started by ``up``."""
    raise SystemExit(run_down())


@app.command
def clean() -> None:
    """Remove caches, coverage data, ``tmp/``, and the Go worker binary (stops a live NER cache server first)."""
    raise SystemExit(run_clean())


__all__ = [
    "airflow_importable",
    "api_up",
    "app",
    "copy_export_bundle",
    "dag_registered",
    "export_interim_refs",
    "extract_fixture_sources",
    "pid_alive",
    "run_clean",
    "run_down",
    "run_export_medliner",
    "run_state",
    "run_subprocess",
    "run_up",
    "sleep",
    "start_standalone",
    "terminate",
]
