"""Logging: ``loguru`` is the primary structured logger, bridged into stdlib ``logging``
so Airflow's task-instance file handler captures every record (per ``PLAN.md`` "Logging
and observability").

Two pieces:

* :class:`InterceptHandler` installs on the stdlib root logger so third-party libraries
  that emit through ``logging`` flow into ``loguru`` (single sink of truth).
* :func:`configure_logging` adds a stderr sink, an optional file sink under
  ``<workdir>/logs/``, and — when running under Airflow — a forwarder that writes loguru
  records into ``logging.getLogger("airflow.task")`` so they appear in Airflow's
  per-task log files. ``airflow.task`` propagation is disabled to prevent recursion.

Structured fields (``task_id``, ``shard_id``, ``artifact_id``, ``input_hash``,
``output_hash``, ``rows``, ``partitions``, ``elapsed_ms``, ``cache_hit``,
``warning_count``) are attached via :func:`bind` / ``logger.contextualize(...)``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """Forward stdlib :mod:`logging` records into ``loguru``.

    Mirrors the canonical loguru integration recipe. Installed on the stdlib root
    logger (and selected children) so library logs share loguru's sinks and formatting.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level names to loguru levels (catch custom levels gracefully).
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller that originated the record, not this handler.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _stdlib_record_sink(message: Any) -> None:
    """Loguru sink that re-emits a record into the ``airflow.task`` stdlib logger.

    Used only when Airflow is the orchestrator so its file handler captures loguru
    output. ``airflow.task`` has ``propagate=False`` (set in :func:`configure_logging`)
    so records never loop back through the root :class:`InterceptHandler`.
    """
    record = message.record
    target = logging.getLogger("airflow.task")
    target.handle(
        logging.LogRecord(
            name="airflow.task",
            level=target.getEffectiveLevel(),
            pathname=str(record["file"].path),
            lineno=record["line"],
            msg=record["message"],
            args=(),
            exc_info=record["exception"],
        )
    )


def configure_logging(workdir: Path | None = None, level: str = "INFO", *, for_airflow: bool = False) -> None:
    """Configure loguru sinks and the stdlib bridge.

    Idempotent: repeated calls replace sinks rather than stacking them.

    Args:
        workdir: If given, a rotating file sink is added at ``<workdir>/logs/dakp.log``.
        level: Minimum level for the stderr and file sinks (e.g. ``"INFO"`` / ``"DEBUG"``).
        for_airflow: When ``True``, also forward loguru records into the ``airflow.task``
            stdlib logger (Airflow must be importable). Tests never set this.
    """
    global _CONFIGURED

    logger.remove()

    # Primary human-facing sink: structured stderr. Auto-colorize only on a real TTY
    # (avoids a terminfo lookup that prints a spurious warning under pytest capture).
    logger.add(sys.stderr, level=level, colorize=None, backtrace=False, diagnose=False, enqueue=False)

    if workdir is not None:
        log_dir = workdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "dakp.log", level=level, rotation="20 MB", retention=5, compression="gz", backtrace=False, diagnose=False, enqueue=False)

    # Bridge stdlib logging -> loguru so third-party libs share our sinks.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.root.setLevel(level)
    for noisy in ("urllib3", "botocore", "airflow.task"):
        logging.getLogger(noisy).setLevel(level)

    if for_airflow:
        # Forward loguru -> airflow.task; disable propagation to avoid a loop with
        # the root InterceptHandler installed above.
        task_logger = logging.getLogger("airflow.task")
        task_logger.propagate = False
        task_logger.setLevel(level)
        sink_id: int = logger.add(_stdlib_record_sink, level=level)
        # Keep a reference so the sink is not GC'd; loguru already retains it.
        _AIRFLOW_SINKS.append(sink_id)

    _CONFIGURED = True


_AIRFLOW_SINKS: list[int] = []


def bind(**fields: Any) -> Any:
    """Return a loguru logger pre-bound with structured fields for the current task.

    Example::

        log = bind(task_id="extract_faers", shard_id="24Q3", artifact_id=ref.blake3)
        log.info("parsed quarter", rows=12, cache_hit=False)
    """
    return logger.bind(**fields)


# Re-export logger type for annotations that need it without importing loguru everywhere.
LoggerLike = Callable[..., None]

__all__ = ["InterceptHandler", "bind", "configure_logging", "logger"]
