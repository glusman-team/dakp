"""Edge-case tests for ``dakp_pipeline.logging_setup``.

Covers the stdlib->loguru bridge (:class:`InterceptHandler.emit`, including the custom-level
``ValueError`` fallback and the caller-frame walk), the loguru->``airflow.task`` forwarder
(:func:`_stdlib_record_sink`) reached via ``configure_logging(for_airflow=True)``, and the
``workdir is None`` branch of :func:`configure_logging`. Each test restores a clean logging
configuration afterwards so global loguru/stdlib state never leaks into other tests.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from dakp_pipeline.logging_setup import FROM_LOGURU_ATTR, InterceptHandler, _stdlib_record_sink, bind, configure_logging, progress, stats, step


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


def test_configure_logging_for_airflow_preserves_existing_root_handlers(tmp_path: Path) -> None:
    # Simulate the handler Airflow already installed on the stdlib root logger. for_airflow
    # must NOT clobber it (no basicConfig(force=True)) — records flow up to it instead.
    seen: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    capture = _Capture()
    logging.root.addHandler(capture)
    try:
        configure_logging(tmp_path, for_airflow=True)
        assert capture in logging.root.handlers  # the pre-existing handler survived
        logger.info("forwarded to airflow")
    finally:
        logging.root.removeHandler(capture)

    forwarded = [r for r in seen if r.getMessage() == "forwarded to airflow"]
    assert forwarded
    assert forwarded[0].levelno == logging.INFO
    assert getattr(forwarded[0], FROM_LOGURU_ATTR, False) is True
    # The workdir file sink still captures the record for offline reading.
    assert "forwarded to airflow" in (tmp_path / "logs" / "dakp.log").read_text(encoding="utf-8")


def test_for_airflow_preserves_real_loguru_level(tmp_path: Path) -> None:
    seen: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    capture = _Capture()
    logging.root.addHandler(capture)
    try:
        configure_logging(tmp_path, for_airflow=True)
        logger.warning("a warning")
    finally:
        logging.root.removeHandler(capture)

    forwarded = [r for r in seen if r.getMessage() == "a warning"]
    assert forwarded
    assert forwarded[0].levelno == logging.WARNING


def test_stdlib_record_sink_builds_a_logrecord() -> None:
    # Exercise the sink callable directly with a synthetic loguru message. The sink emits
    # through the origin-name logger (propagating to root), so capture at the root.
    seen: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    capture = _Capture()
    logging.root.addHandler(capture)
    message = SimpleNamespace(
        record={
            "name": "dakp.synthetic",
            "level": SimpleNamespace(no=logging.WARNING),
            "file": SimpleNamespace(path=__file__),
            "line": 42,
            "message": "direct sink call",
            "exception": None,
        }
    )
    try:
        _stdlib_record_sink(message)
    finally:
        logging.root.removeHandler(capture)

    assert seen
    assert seen[0].getMessage() == "direct sink call"
    assert seen[0].lineno == 42
    assert seen[0].name == "dakp.synthetic"
    assert seen[0].levelno == logging.WARNING
    assert getattr(seen[0], FROM_LOGURU_ATTR, False) is True


def test_intercept_handler_skips_records_reemitted_from_loguru() -> None:
    # A record carrying the sentinel must NOT be re-ingested into loguru (loop guard).
    lines = _capture_sink()
    handler = InterceptHandler()
    record = logging.LogRecord("dakp.loop", logging.INFO, __file__, 1, "echo", (), None)
    setattr(record, FROM_LOGURU_ATTR, True)
    handler.emit(record)
    assert lines == []


def test_configure_logging_is_idempotent_and_bind_returns_bound_logger(tmp_path: Path) -> None:
    configure_logging(tmp_path)
    configure_logging(tmp_path)  # repeated calls replace sinks rather than stacking
    log = bind(task_id="edge", shard_id="x")
    log.info("bound fields")
    assert "bound fields" in (tmp_path / "logs" / "dakp.log").read_text(encoding="utf-8")


# --- one-stat-per-line narration helpers ------------------------------------------


def _capture_sink() -> list[str]:
    """Attach an in-memory loguru sink and return the list it appends rendered lines to."""
    lines: list[str] = []
    logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    return lines


def test_stats_emits_one_line_per_field_in_order() -> None:
    lines = _capture_sink()
    stats(logger, "extract_faers", quarter="24Q3", rows=123, cache_hit=False)
    assert lines == ["extract_faers: quarter = 24Q3", "extract_faers: rows = 123", "extract_faers: cache_hit = false"]


def test_stats_formats_true_lowercase_and_respects_level() -> None:
    seen: list[str] = []
    logger.add(lambda message: seen.append(message.record["level"].name), level="DEBUG")
    lines = _capture_sink()
    stats(logger, "acquire", level="DEBUG", cache_hit=True)
    assert lines == ["acquire: cache_hit = true"]
    assert seen == ["DEBUG"]


def test_stats_with_no_fields_emits_nothing() -> None:
    lines = _capture_sink()
    stats(logger, "quiet")
    assert lines == []


def test_step_logs_started_finished_and_elapsed() -> None:
    lines = _capture_sink()
    with step(logger, "acquire_faers"):
        pass
    assert lines[0] == "acquire_faers: started"
    assert lines[1] == "acquire_faers: finished = true"
    assert re.fullmatch(r"acquire_faers: elapsed_s = \d+(\.\d+)?", lines[2])
    assert len(lines) == 3


def test_step_logs_failed_with_error_type_and_reraises() -> None:
    lines = _capture_sink()
    with pytest.raises(ValueError, match="boom"), step(logger, "acquire_faers"):
        raise ValueError("boom")
    assert lines[0] == "acquire_faers: started"
    assert lines[1] == "acquire_faers: failed = true"
    assert lines[2] == "acquire_faers: error = ValueError"
    assert lines[3].startswith("acquire_faers: elapsed_s = ")
    assert len(lines) == 4


def test_progress_emits_only_on_positive_multiples() -> None:
    lines = _capture_sink()
    progress(logger, "ingest_spl", 0, 12, every=5)  # done=0 never logs
    progress(logger, "ingest_spl", 3, 12, every=5)  # not a multiple
    progress(logger, "ingest_spl", 5, 12, every=5)
    progress(logger, "ingest_spl", 10, 12, every=5)
    progress(logger, "ingest_spl", 7, 12, every=0)  # every=0 disables (no div-by-zero)
    assert lines == ["ingest_spl: progress = 5/12", "ingest_spl: progress = 10/12"]
