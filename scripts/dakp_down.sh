#!/usr/bin/env bash
# Tear down the Airflow standalone started by scripts/dakp_up.sh (make down).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$REPO_ROOT/tmp/airflow-home}"
PIDFILE="$AIRFLOW_HOME/standalone.pid"

if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo ">>> stopping Airflow standalone (pid $pid)"
    kill -TERM "$pid" 2>/dev/null
    sleep 5
  fi
  rm -f "$PIDFILE"
fi
# Reap any lingering standalone components (scheduler/triggerer/workers/uvicorn). Never touch
# unrelated processes (e.g. the aoe daemon that owns port 8080).
pkill -TERM -f "airflow standalone" 2>/dev/null
pkill -TERM -f "airflow scheduler" 2>/dev/null
pkill -TERM -f "airflow triggerer" 2>/dev/null
pkill -TERM -f "airflow worker" 2>/dev/null
pkill -TERM -f "airflow serve-logs" 2>/dev/null
sleep 2
pkill -9 -f "airflow standalone" 2>/dev/null
pkill -9 -f "airflow scheduler" 2>/dev/null
pkill -9 -f "airflow worker" 2>/dev/null
echo ">>> Airflow stopped"
