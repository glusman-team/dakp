"""Logging: ``loguru`` is the primary structured logger, bridged into stdlib ``logging``
so Airflow's task-instance log captures every record.

Three pieces:

* :class:`InterceptHandler` installs on the stdlib root logger (local / test runs only) so
  third-party libraries that emit through ``logging`` flow into ``loguru`` (single sink of
  truth).
* :func:`configure_logging` adds an optional file sink under ``<workdir>/logs/`` plus either a
  stderr sink (local runs) or — under Airflow — ONLY the :func:`_stdlib_record_sink` forwarder.
  Airflow 3 owns the stdlib root handler in task processes (its structlog formatter writes the
  per-task log), so under Airflow we forward loguru records INTO stdlib logging and never touch
  the root handlers: no ``InterceptHandler`` (it would clobber Airflow's handler) and no stderr
  sink (Airflow captures subprocess stderr as ERROR-level ``task.stderr`` lines).
* Narration helpers (:func:`stats`, :func:`step`, :func:`progress`) implement the DAKP
  one-stat-per-line convention so task logs stay readable in the Airflow UI.

Forwarded records carry the :data:`FROM_LOGURU_ATTR` sentinel so :class:`InterceptHandler`
never re-ingests them (loop prevention).
"""

from __future__ import annotations

import logging
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


__all__ = ["FROM_LOGURU_ATTR", "InterceptHandler", "bind", "configure_logging", "logger", "progress", "stats", "step"]
