#!/usr/bin/env bash
# One-command DAKP pipeline run via Airflow with native Go SDK workers.
#
#   make up-mock | make up-sample | make up-prod      (or: PROFILE=<p> bash scripts/dakp_up.sh)
#
# All configuration comes from the environment (see `.envrc`): PROFILE, WORKDIR, FIXTURE_ROOT,
# AIRFLOW_HOME, AIRFLOW_PORT, QUARTER_LIMIT, RELEASE_LIMIT, LOG_LEVEL, DOWNLOAD_POOL, EXTRACT_POOL.
# QUARTER_LIMIT/RELEASE_LIMIT empty => profile default (prod default = unbounded full build).
#
# Builds + packs the Go bundle, starts Airflow (standalone) with the ExecutableCoordinator
# configured, sets the dakp_config Variable, triggers the dakp_build DAG, waits for it to finish,
# and prints the build-summary path. Idempotent: reuses a running Airflow if one is already up.
#
# Port 8090 is the default because 8080 is commonly taken (e.g. by the aoe daemon on dev hosts).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${PROFILE:-mock}"
WORKDIR="${WORKDIR:-$REPO_ROOT/tmp/airflow-run/data}"
FIXTURE_ROOT="${FIXTURE_ROOT:-$REPO_ROOT/tests/fixtures/pipeline}"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$REPO_ROOT/tmp/airflow-home}"
PORT="${AIRFLOW_PORT:-8090}"
BUNDLE_DIR="$AIRFLOW_HOME/executable-bundles"
DOWNLOAD_POOL="${DOWNLOAD_POOL:-dakp_download}"
EXTRACT_POOL="${EXTRACT_POOL:-dakp_extract}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
# Run-scope limits from .envrc: empty => JSON null => profile default (prod = unbounded full build).
for _v in "${QUARTER_LIMIT:-}" "${RELEASE_LIMIT:-}"; do
  [[ -z "$_v" || "$_v" =~ ^[0-9]+$ ]] || { echo "!!! QUARTER_LIMIT/RELEASE_LIMIT must be empty or a positive integer (got '$_v')"; exit 1; }
done
json_limit() { [[ -n "${1:-}" ]] && printf '%s' "$1" || printf 'null'; }
QUARTER_JSON="$(json_limit "${QUARTER_LIMIT:-}")"
RELEASE_JSON="$(json_limit "${RELEASE_LIMIT:-}")"
LOG="$AIRFLOW_HOME/standalone.log"
PIDFILE="$AIRFLOW_HOME/standalone.pid"
BASE_URL="http://127.0.0.1:$PORT"

mkdir -p "$AIRFLOW_HOME" "$WORKDIR"

# --- Airflow config (env vars; AIRFLOW__<SECTION>__<KEY>) ---------------------
export AIRFLOW__CORE__DAGS_FOLDER="$REPO_ROOT/src/dakp_pipeline/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__API__PORT="$PORT"
export AIRFLOW__API__HOST=127.0.0.1
# MANDATORY: without this the task supervisor's Execution-API client defaults to localhost:8080.
export AIRFLOW__CORE__EXECUTION_API_SERVER_URL="$BASE_URL/execution/"
export AIRFLOW__SDK__COORDINATORS="{\"go\": {\"classpath\": \"airflow.sdk.coordinators.executable.ExecutableCoordinator\", \"kwargs\": {\"executables_root\": [\"$BUNDLE_DIR\"]}}}"
export AIRFLOW__SDK__QUEUE_TO_COORDINATOR="{\"golang\": \"go\"}"

# --- 1. build + pack the Go bundle into executables_root ----------------------
echo ">>> [1/6] building + packing the native Go bundle"
mkdir -p "$BUNDLE_DIR"
(cd go && go tool airflow-go-pack --output "$BUNDLE_DIR/dakp-bundle" ./cmd/dakp-bundle) || { echo "bundle pack failed"; exit 1; }

# --- 2. start Airflow standalone (reuse if already up) ------------------------
api_up() { curl -sf "$BASE_URL/api/v2/version" >/dev/null 2>&1; }
if api_up; then
  echo ">>> [2/6] Airflow already running on :$PORT (reusing)"
else
  echo ">>> [2/6] starting Airflow standalone on :$PORT (logs: $LOG)"
  nohup uv run airflow standalone > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 90); do api_up && break; sleep 2; done
  api_up || { echo "!!! API server did not come up; tail of $LOG:"; tail -40 "$LOG"; exit 1; }
fi

# --- 3. wait for the DAG to be registered, then unpause -----------------------
echo ">>> [3/6] waiting for DAG registration"
dag_registered() {
  uv run python - "$AIRFLOW_HOME/airflow.db" <<'PY' 2>/dev/null
import sqlite3, sys
try:
    con = sqlite3.connect(sys.argv[1])
    n = con.execute("SELECT count(*) FROM dag WHERE dag_id='dakp_build'").fetchone()[0]
    print("yes" if n else "no")
except Exception:
    print("no")
PY
}
for _ in $(seq 1 45); do [[ "$(dag_registered)" == "yes" ]] && break; sleep 2; done
[[ "$(dag_registered)" == "yes" ]] || { echo "!!! DAG dakp_build never registered; check $LOG"; exit 1; }
uv run airflow dags unpause dakp_build >/dev/null 2>&1

# Provision the concurrency-bounding pools the DAG references (they are not auto-created; tasks on
# a non-existent pool are never scheduled).
uv run airflow pools set "$DOWNLOAD_POOL" 4 "Concurrent source downloads (network I/O)" >/dev/null 2>&1
uv run airflow pools set "$EXTRACT_POOL" 4 "Concurrent raw->interim extracts (Go bundle)" >/dev/null 2>&1

# --- 4. set the per-run config Variable (shared by Python tasks + Go bundle) --
echo ">>> [4/6] setting dakp_config Variable (profile=$PROFILE workdir=$WORKDIR quarter=$QUARTER_JSON release=$RELEASE_JSON)"
uv run airflow variables set dakp_config \
  "{\"workdir\": \"$WORKDIR\", \"profile\": \"$PROFILE\", \"fixture_root\": \"$FIXTURE_ROOT\", \"quarter_limit\": $QUARTER_JSON, \"release_limit\": $RELEASE_JSON, \"log_level\": \"$LOG_LEVEL\"}" \
  >/dev/null 2>&1

# --- 5. trigger the DAG run ---------------------------------------------------
echo ">>> [5/6] triggering dakp_build"
uv run airflow dags trigger dakp_build >/dev/null 2>&1 || { echo "!!! trigger failed"; exit 1; }

run_state() {
  uv run python - "$AIRFLOW_HOME/airflow.db" <<'PY' 2>/dev/null
import sqlite3, sys
try:
    con = sqlite3.connect(sys.argv[1])
    r = con.execute("SELECT state FROM dag_run WHERE dag_id='dakp_build' ORDER BY run_after DESC LIMIT 1").fetchone()
    print(r[0] if r and r[0] else "none")
except Exception:
    print("none")
PY
}

# --- 6. wait for completion ---------------------------------------------------
echo ">>> [6/6] waiting for the run to finish"
final=""
for i in $(seq 1 300); do
  st="$(run_state)"
  printf '    [%d] run_state=%s\n' "$i" "$st"
  case "$st" in
    success) final=success; break;;
    failed)  final=failed; break;;
  esac
  sleep 3
done

echo
if [[ "$final" == "success" ]]; then
  echo ">>> SUCCESS — build summary: $WORKDIR/data/reports/build_summary.json"
  [[ -f "$WORKDIR/data/reports/build_summary.json" ]] && cat "$WORKDIR/data/reports/build_summary.json"
  exit 0
else
  echo "!!! DAG run did not succeed (final=$final). Inspect: $LOG and $AIRFLOW_HOME/logs/"
  exit 1
fi
