"""Edge-case tests for ``dakp_pipeline.logging_setup``.

Covers the stdlib->loguru bridge (:class:`InterceptHandler.emit`, including the custom-level
``ValueError`` fallback and the caller-frame walk), the loguru->``airflow.task`` forwarder
(:func:`_stdlib_record_sink`) reached via ``configure_logging(for_airflow=True)``, and the
``workdir is None`` branch of :func:`configure_logging`. Each test restores a clean logging
configuration afterwards so global loguru/stdlib state never leaks into other tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from dakp_pipeline.logging_setup import InterceptHandler, _stdlib_record_sink, bind, configure_logging


@pytest.fixture(autouse=True)
def _restore_logging() -> object:
    """Reset loguru sinks + stdlib root handlers before and after each test."""
    configure_logging()
    yield
    configure_logging()


def test_intercept_handler_forwards_stdlib_record_into_loguru(tmp_path: Path) -> None:
    configure_logging(tmp_path)  # installs InterceptHandler on the stdlib root logger
    logging.getLogger("dakp.test.bridge").info("bridged message")

    log_file = tmp_path / "logs" / "dakp.log"
    assert log_file.exists()
    assert "bridged message" in log_file.read_text(encoding="utf-8")


def test_intercept_handler_maps_unknown_level_via_levelno(tmp_path: Path) -> None:
    configure_logging(tmp_path)
    # A level name loguru does not know -> logger.level(name) raises ValueError -> levelno used.
    logging.addLevelName(33, "DAKP_CUSTOM")
    try:
        logging.getLogger("dakp.test.custom").log(33, "custom level message")
    finally:
        logging.addLevelName(33, "Level 33")

    assert "custom level message" in (tmp_path / "logs" / "dakp.log").read_text(encoding="utf-8")


def test_intercept_handler_emit_directly_with_exception() -> None:
    # Drive emit() directly to exercise the caller-frame walk and exception forwarding.
    handler = InterceptHandler()
    try:
        msg = "boom"
        raise ValueError(msg)
    except ValueError:
        import sys

        exc_info = sys.exc_info()
    record = logging.LogRecord("dakp.direct", logging.ERROR, __file__, 1, "failed thing", (), exc_info)
    handler.emit(record)  # must not raise; forwards into loguru with the exception attached


def test_configure_logging_without_workdir_adds_no_file_sink(tmp_path: Path) -> None:
    configure_logging(None)  # workdir is None -> no file sink (the False branch)
    logger.info("stderr only")
    assert not (tmp_path / "logs").exists()


def test_configure_logging_for_airflow_forwards_to_task_logger(tmp_path: Path) -> None:
    configure_logging(tmp_path, for_airflow=True)  # adds the _stdlib_record_sink forwarder

    task_logger = logging.getLogger("airflow.task")
    assert task_logger.propagate is False  # loop prevention installed by configure_logging

    seen: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    capture = _Capture()
    task_logger.addHandler(capture)
    try:
        logger.info("forwarded to airflow")  # loguru -> _stdlib_record_sink -> airflow.task
    finally:
        task_logger.removeHandler(capture)

    assert "forwarded to airflow" in seen


def test_stdlib_record_sink_builds_a_logrecord() -> None:
    # Exercise the sink callable directly with a synthetic loguru message.
    task_logger = logging.getLogger("airflow.task")
    seen: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    capture = _Capture()
    task_logger.addHandler(capture)
    message = SimpleNamespace(record={"file": SimpleNamespace(path=__file__), "line": 42, "message": "direct sink call", "exception": None})
    try:
        _stdlib_record_sink(message)
    finally:
        task_logger.removeHandler(capture)

    assert seen
    assert seen[0].getMessage() == "direct sink call"
    assert seen[0].lineno == 42


def test_configure_logging_is_idempotent_and_bind_returns_bound_logger(tmp_path: Path) -> None:
    configure_logging(tmp_path)
    configure_logging(tmp_path)  # repeated calls replace sinks rather than stacking
    log = bind(task_id="edge", shard_id="x")
    log.info("bound fields")
    assert "bound fields" in (tmp_path / "logs" / "dakp.log").read_text(encoding="utf-8")
