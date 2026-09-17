"""Logging: ``loguru`` is the primary structured logger, bridged into stdlib ``logging``
so Airflow's task-instance log captures every record.

Four pieces:

* :class:`InterceptHandler` installs on the stdlib root logger (local / test / worker runs) so
  third-party libraries that emit through ``logging`` flow into ``loguru`` (single sink of
  truth).
* :func:`configure_logging` adds an optional file sink under ``<workdir>/logs/`` plus either a
  stderr sink (local runs) or — under Airflow — ONLY the :func:`_stdlib_record_sink` forwarder.
  Airflow 3 owns the stdlib root handler in task processes (its structlog formatter writes the
  per-task log), so under Airflow we forward loguru records INTO stdlib logging and never touch
  the root handlers: no ``InterceptHandler`` (it would clobber Airflow's handler) and no stderr
  sink (Airflow captures subprocess stderr as ERROR-level ``task.stderr`` lines).
* :func:`configure_worker_logging` is the SPAWNED-CHILD counterpart: a fresh interpreter starts
  with loguru's default stderr sink, which Airflow tags as ERROR, so every worker record (and
  every library warning) surfaced in the task log as a false error. It redirects the child's
  stderr FILE DESCRIPTOR to a per-worker file, which is the only capture point that also covers
  libraries owning their own stderr handler (``transformers``, ``torch``) and native output.
* Narration helpers (:func:`stats`, :func:`step`, :func:`progress`) implement the DAKP
  one-stat-per-line convention so task logs stay readable in the Airflow UI.

Forwarded records carry the :data:`FROM_LOGURU_ATTR` sentinel so :class:`InterceptHandler`
never re-ingests them (loop prevention).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger


class InterceptHandler(logging.Handler):
    """Forward stdlib :mod:`logging` records into ``loguru``.

    Mirrors the canonical loguru integration recipe. Installed on the stdlib root
    logger (and selected children) so library logs share loguru's sinks and formatting.
    Records carrying the :data:`FROM_LOGURU_ATTR` sentinel originated FROM loguru and are
    skipped so the bridge can never loop.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, FROM_LOGURU_ATTR, False):
            return  # re-emitted from loguru; re-ingesting it would loop

        # Map stdlib level names to loguru levels (catch custom levels gracefully).
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller that originated the record, not this handler: walk up through the
        # stdlib logging frames. The first frame is always this handler (in this module, not
        # ``logging``), so ``depth == 0`` forces the first step before the filename test applies.
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _stdlib_record_sink(message: Any) -> None:
    """Loguru sink that re-emits records into stdlib logging for Airflow's task-log handler.

    Airflow 3 owns the stdlib root handler in task processes (its structlog formatter writes
    the per-task log), so records are emitted through the ORIGIN module logger with
    propagation left intact: they flow up to the root handler Airflow installed and are
    rendered there with their real level and logger name. The :data:`FROM_LOGURU_ATTR`
    sentinel marks re-emitted records so :class:`InterceptHandler` never re-ingests them.
    """
    record = message.record
    log_record = logging.LogRecord(
        name=record["name"],
        level=record["level"].no,
        pathname=str(record["file"].path),
        lineno=record["line"],
        msg=record["message"],
        args=(),
        exc_info=record["exception"],
    )
    setattr(log_record, FROM_LOGURU_ATTR, True)
    logging.getLogger(record["name"]).handle(log_record)


def configure_logging(workdir: Path | None = None, level: str = "INFO", *, for_airflow: bool = False) -> None:
    """Configure loguru sinks and the stdlib bridge.

    Idempotent: repeated calls replace sinks rather than stacking them.

    Args:
        workdir: If given, a rotating file sink is added at ``<workdir>/logs/dakp.log``.
        level: Minimum level for the sinks (e.g. ``"INFO"`` / ``"DEBUG"``).
        for_airflow: When ``True``, loguru records are forwarded into stdlib logging for
            Airflow's own root handler (correct levels, structured task-log records). The
            stderr sink and the root :class:`InterceptHandler` are NOT installed: Airflow
            captures subprocess stderr as ERROR-level ``task.stderr`` noise, and it already
            owns the stdlib root handler (clobbering it is what hid DAKP logs before).
    """
    logger.remove()

    if workdir is not None:
        log_dir = workdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "dakp.log", level=level, rotation="20 MB", retention=5, compression="gz", backtrace=False, diagnose=False, enqueue=False)

    if for_airflow:
        # Airflow formats + routes task logs itself; just hand it our records.
        sink_id: int = logger.add(_stdlib_record_sink, level=level)
        # Keep a reference so the sink is not GC'd; loguru already retains it.
        _AIRFLOW_SINKS.append(sink_id)
        return

    # Local/test runs: loguru is the single sink of truth. Primary human-facing sink:
    # structured stderr. Auto-colorize only on a real TTY (avoids a terminfo lookup that
    # prints a spurious warning under pytest capture).
    logger.add(sys.stderr, level=level, colorize=None, backtrace=False, diagnose=False, enqueue=False)

    # Bridge stdlib logging -> loguru so third-party libs share our sinks.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.root.setLevel(level)
    for noisy in ("urllib3", "botocore", "airflow.task"):
        logging.getLogger(noisy).setLevel(level)


_AIRFLOW_SINKS: list[int] = []


#: Worker log files live here, one file per worker PROCESS (see :func:`_worker_log_path`).
WORKER_LOG_SUBDIR = Path("logs") / "workers"

#: Level for the worker's own (DAKP) records once its stderr is a private file: keep it verbose.
_WORKER_SINK_LEVEL = "DEBUG"

#: Level for the worker's records when stderr is still Airflow's (no worker file): real problems
#: only. An ERROR genuinely belongs in the task log; INFO/DEBUG is the noise this module removes.
_WORKER_FALLBACK_LEVEL = "ERROR"

#: Root level for bridged stdlib records (``warnings``, libraries without their own handler).
_WORKER_LIBRARY_LEVEL = logging.WARNING

#: Worker logs older than this are pruned at dispatch time (they are per-process diagnostics,
#: so nothing rotates them in place).
WORKER_LOG_MAX_AGE_S = 7 * 24 * 60 * 60

_STDERR_FD = 2


def _worker_log_name(name: str) -> str:
    """Sanitize a worker name into a filesystem-safe stem (``"cuda:0"`` -> ``"cuda-0"``)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.") or "worker"


def _worker_log_path(workdir: Path | str, name: str) -> Path:
    """Per-PROCESS worker log path: ``<workdir>/logs/workers/<name>-<pid>.log``.

    The pid is part of the name because a device is NOT a unique worker key: sequential shape
    tasks and repeated dispatches reuse each device with a fresh process. Keying on the pid
    guarantees no two processes ever
    write to one file, which is also why nothing rotates these files in place (see
    :func:`prune_worker_logs`).
    """
    log_dir = Path(workdir) / WORKER_LOG_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{_worker_log_name(name)}-{os.getpid()}.log"


def configure_worker_logging(workdir: Path | str | None, name: str) -> None:
    """Configure logging inside a ``spawn``-started child process (the per-GPU NER workers).

    A spawned child is a fresh interpreter: loguru starts with its DEFAULT sink (``sys.stderr``,
    level DEBUG) and stdlib ``logging`` has no handler. Airflow captures a task subprocess's
    stderr as ERROR-level ``task.stderr`` lines, which is why healthy worker records
    (``ner_model_load``, ``ner_gpu_lock``), the transformers ``model of type`` notice, and
    gliner2's sdpa-fallback ``RuntimeWarning`` all appeared as ERRORs in the task log.

    The capture point is the stderr FILE DESCRIPTOR, not ``sys.stderr`` or the root logger:
    ``transformers`` and ``torch`` both set ``propagate = False`` on their library logger and
    attach their OWN ``StreamHandler(sys.stderr)``, so a root-level :class:`InterceptHandler`
    never sees those records, and native CUDA / tqdm output never passes through Python at all.
    ``dup2`` onto fd 2 catches every one of them. Nothing is suppressed; it is REDIRECTED to
    ``<workdir>/logs/workers/<name>-<pid>.log``, and on top of the redirect:

    * loguru is re-pointed at ``sys.stderr`` (now the file) at :data:`_WORKER_SINK_LEVEL`;
    * stdlib ``logging`` is bridged into loguru by :class:`InterceptHandler` at
      :data:`_WORKER_LIBRARY_LEVEL`, so propagating library records are formatted like ours
      (libraries holding their own stderr handler land in the same file via the redirect);
    * :mod:`warnings` is routed through ``logging.captureWarnings`` into that same bridge.

    With no usable worker file (``workdir`` None in tests/offline runs, or an unwritable log
    directory) the child keeps Airflow's stderr but logs at :data:`_WORKER_FALLBACK_LEVEL`
    only: a failure still surfaces in the task log, the routine narration does not. Logging
    setup never raises here — an unwritable path must not turn into a failed shard.
    """
    logger.remove()  # drop loguru's default sink; re-added below at the level we actually want

    level = _WORKER_FALLBACK_LEVEL
    if workdir is not None:
        try:
            with _worker_log_path(workdir, name).open("ab", buffering=0) as handle:
                # dup2 duplicates the open file description onto fd 2, so closing `handle`
                # right after leaves fd 2 pointing at the file for the life of the process.
                sys.stderr.flush()
                os.dup2(handle.fileno(), _STDERR_FD)
        except OSError:
            pass  # unwritable workdir: keep Airflow's stderr, stay at the fallback level
        else:
            level = _WORKER_SINK_LEVEL

    logger.add(sys.stderr, level=level, colorize=False, backtrace=False, diagnose=False, enqueue=False)

    # Bridge stdlib logging (and, through it, `warnings`) into loguru. Safe here — unlike the
    # Airflow task process, a spawned child owns its own root logger and has no Airflow handler
    # to clobber. Libraries that set `propagate = False` bypass this and are caught by the fd
    # redirect instead.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.root.setLevel(_WORKER_LIBRARY_LEVEL)
    logging.captureWarnings(True)


def prune_worker_logs(workdir: Path | str, max_age_s: float = WORKER_LOG_MAX_AGE_S) -> int:
    """Delete worker logs older than ``max_age_s``; return how many were removed.

    Worker logs are per-process files written through a redirected file descriptor, so no sink
    rotates or retires them. The dispatching parent calls this once before spawning, which
    bounds the directory without ever touching a file a live worker holds open (those are
    brand new). Best-effort by design: a missing directory or an unlink race is not an error.
    """
    cutoff = time.time() - max_age_s
    removed = 0
    for path in sorted((Path(workdir) / WORKER_LOG_SUBDIR).glob("*.log")):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:  # pragma: no cover - unlink/stat race with a concurrent dispatcher
            continue
    return removed


def bind(**fields: Any) -> Any:
    """Return a loguru logger pre-bound with structured fields for the current task.

    Example::

        log = bind(task_id="extract_faers", shard_id="24Q3", artifact_id=ref.blake3)
        log.info("parsed quarter", rows=12, cache_hit=False)
    """
    return logger.bind(**fields)


# --- one-stat-per-line narration helpers ------------------------------------------
#
# Airflow's task-log viewer reads best as short, self-contained lines, so DAKP never packs
# several stats into one record: every stat gets its own ``event: key = value`` line, prefixed
# with the event name so each line stands alone and is greppable.

#: Attribute set on stdlib records re-emitted from loguru so bridges never re-ingest them.
FROM_LOGURU_ATTR = "_dakp_from_loguru"


def _format_value(value: Any) -> str:
    """Render a stat value for a log line (bools lowercase, everything else ``str``)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def stats(log: Any, event: str, /, *, level: str = "INFO", _depth: int = 1, **fields: Any) -> None:
    """Emit one ``event: key = value`` log line per field, in insertion order.

    Example::

        stats(log, "extract_faers", quarter="24Q3", rows=123, cache_hit=False)
        # extract_faers: quarter = 24Q3
        # extract_faers: rows = 123
        # extract_faers: cache_hit = false

    ``log`` may be the module :data:`logger` or any :func:`bind` result. ``level`` is a loguru
    level name (e.g. ``"DEBUG"`` for verbose per-artifact stats). ``_depth`` attributes the
    record to the caller (helpers like :func:`step` pass 2 so records name THEIR caller).
    """
    for key, value in fields.items():
        log.opt(depth=_depth).log(level, "{}: {} = {}", event, key, _format_value(value))


def _elapsed_s(started: float) -> float:
    return round(time.monotonic() - started, 3)


@contextmanager
def step(log: Any, event: str) -> Iterator[None]:
    """Narrate a pipeline phase: ``event: started`` on entry, finish/fail stats on exit.

    On success emits ``event: finished = true`` and ``event: elapsed_s = <N>`` (one line
    each). On exception emits ``event: failed = true``, ``event: error = <ExcType>`` and the
    elapsed time, then re-raises. Usage::

        with step(log, "acquire_faers"):
            refs = faers.fetch(ctx)
    """
    log.opt(depth=1).info("{}: started", event)
    started = time.monotonic()
    try:
        yield
    except BaseException as exc:
        stats(log, event, _depth=2, failed=True, error=type(exc).__name__, elapsed_s=_elapsed_s(started))
        raise
    stats(log, event, _depth=2, finished=True, elapsed_s=_elapsed_s(started))


def progress(log: Any, event: str, done: int, total: int, *, every: int) -> None:
    """Emit ``event: progress = <done>/<total>`` every ``every`` items; a no-op otherwise.

    Call once per item after incrementing; only multiples of ``every`` (and never ``done=0``)
    log, so long loops stay quiet between milestones.
    """
    if done > 0 and every > 0 and done % every == 0:
        log.opt(depth=1).info("{}: progress = {}/{}", event, done, total)


__all__ = [
    "FROM_LOGURU_ATTR",
    "WORKER_LOG_MAX_AGE_S",
    "WORKER_LOG_SUBDIR",
    "InterceptHandler",
    "bind",
    "configure_logging",
    "configure_worker_logging",
    "logger",
    "progress",
    "prune_worker_logs",
    "stats",
    "step",
]
