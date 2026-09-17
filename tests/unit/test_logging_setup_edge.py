"""Edge-case tests for ``dakp_pipeline.logging_setup``.

Covers the stdlib->loguru bridge (:class:`InterceptHandler.emit`, including the custom-level
``ValueError`` fallback and the caller-frame walk), the loguru->``airflow.task`` forwarder
(:func:`_stdlib_record_sink`) reached via ``configure_logging(for_airflow=True)``, the
``workdir is None`` branch of :func:`configure_logging`, and the spawned-worker configuration
(:func:`configure_worker_logging`, which redirects the child's stderr fd so Airflow stops
reading healthy child records as ERRORs). Each test restores a clean logging configuration
afterwards so global loguru/stdlib state never leaks into other tests.

The worker tests run the real thing in a real ``spawn``-started child (:func:`_run_in_child`):
the fix is a file-descriptor redirect, and a child is the only place where fd 2, a fresh
loguru, and libraries that own their own stderr handler all behave as they do in production.
Asserting against an in-process fake would pin nothing.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import select
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from dakp_pipeline.logging_setup import (
    FROM_LOGURU_ATTR,
    WORKER_LOG_SUBDIR,
    InterceptHandler,
    _stdlib_record_sink,
    _worker_log_name,
    bind,
    configure_logging,
    configure_worker_logging,
    progress,
    prune_worker_logs,
    stats,
    step,
)


@pytest.fixture(autouse=True)
def _restore_logging() -> object:
    """Reset loguru sinks + stdlib root handlers before and after each test."""
    configure_logging()
    yield
    logging.captureWarnings(False)  # configure_worker_logging turns this on globally
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


# --- spawned-worker logging -------------------------------------------------------


def _run_in_child(target: Callable[..., None], *args: object) -> str:
    """Run ``target`` in a real ``spawn`` child with its stderr captured; return that stderr.

    The parent's fd 2 is redirected to a pipe for the duration, because the production symptom
    IS the child's stderr reaching the parent (where Airflow tags it ERROR): asserting the pipe
    is empty is what proves the redirect works. The child is spawned, not forked, so it starts
    with a fresh interpreter exactly as ``ProcessPoolExecutor`` workers do.

    The capture never waits for pipe EOF. On the FIRST ``spawn`` use in an interpreter,
    ``Process.start()`` also spawns the multiprocessing ``resource_tracker``, which inherits
    fd 2 (the pipe's write end) and stays alive for the whole process — so EOF may never
    arrive and a plain ``os.read`` deadlocks (observed as a permanently hung test run when
    this file executed first in its pytest process). The child is already joined (and its
    exit code asserted) before reading, so every byte it will ever write is already in the
    kernel pipe buffer; one bounded ``select`` + drain collects exactly that.
    """
    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    try:
        os.dup2(write_fd, 2)
        process = mp.get_context("spawn").Process(target=target, args=args)
        process.start()
        process.join(timeout=120)
        assert process.exitcode == 0, f"worker child failed with exitcode {process.exitcode}"
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(write_fd)
    captured = b""
    while select.select([read_fd], [], [], 1.0)[0]:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        captured += chunk
    os.close(read_fd)
    return captured.decode("utf-8", "replace")


def _worker_body(workdir: str, device: str) -> None:
    """Child entry point: configure worker logging, then emit on every channel that misbehaved."""
    configure_worker_logging(workdir, device)
    # Imported AFTER configuration, as in production: both libraries attach their OWN
    # StreamHandler(sys.stderr) and set propagate=False, so only the fd redirect catches them.
    import torch
    import transformers

    # The imports themselves are the point (each installs its own stderr handler), not the values.
    assert torch is not None
    assert transformers is not None
    logger.info("ner_model_load: model_id = fastino/gliner2-large-v1")
    logger.debug("ner_gpu_lock: waited = false")
    logging.getLogger("transformers.modeling_utils").warning("You are using a model of type `extractor`")
    logging.getLogger("torch.cuda").warning("torch-owned handler line")
    warnings.warn("Encoder rejected attn_implementation='sdpa'", RuntimeWarning, stacklevel=1)
    print("native-library chatter", file=sys.stderr)  # stands in for native CUDA / tqdm output


def test_worker_logging_redirects_every_stderr_channel_into_the_worker_file(tmp_path: Path) -> None:
    """The regression test: nothing a healthy worker emits reaches the parent's stderr.

    Airflow tags a task subprocess's stderr as ERROR-level ``task.stderr``, so every line that
    escapes here is a false ERROR in the task log. All four channels (loguru, a library holding
    its own stderr handler, :mod:`warnings`, and raw fd-2 writes) must land in the worker file.
    """
    leaked = _run_in_child(_worker_body, str(tmp_path), "cuda:0")

    worker_logs = list((tmp_path / WORKER_LOG_SUBDIR).glob("cuda-0-*.log"))
    assert len(worker_logs) == 1, f"expected exactly one worker log, got {worker_logs}"
    contents = worker_logs[0].read_text(encoding="utf-8")
    for expected in (
        "ner_model_load: model_id = fastino/gliner2-large-v1",
        "ner_gpu_lock: waited = false",
        "You are using a model of type `extractor`",
        "torch-owned handler line",
        "Encoder rejected attn_implementation='sdpa'",
        "native-library chatter",
    ):
        assert expected in contents, f"{expected!r} missing from the worker log"
    assert leaked == "", f"worker output leaked to the parent's stderr: {leaked!r}"


def _degraded_worker_body(workdir: str | None) -> None:
    """Child entry point for the no-usable-file paths: narration is dropped, errors survive."""
    configure_worker_logging(workdir, "cuda:0")  # must not raise even when unwritable
    logger.info("routine narration that must stay out of the task log")
    logger.error("a real worker failure")


def test_worker_logging_without_workdir_keeps_errors_and_drops_narration(tmp_path: Path) -> None:
    """``workdir`` None: stderr stays Airflow's, so only ERROR-and-up is allowed through."""
    captured = _run_in_child(_degraded_worker_body, None)

    assert "a real worker failure" in captured
    assert "routine narration" not in captured
    assert not (tmp_path / WORKER_LOG_SUBDIR).exists()


def test_worker_logging_survives_an_unwritable_log_directory(tmp_path: Path) -> None:
    """An unwritable workdir degrades to the ERROR-only fallback instead of failing the shard."""
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        captured = _run_in_child(_degraded_worker_body, str(readonly))
    finally:
        readonly.chmod(0o700)  # restore so tmp_path cleanup can remove it

    assert "a real worker failure" in captured
    assert "routine narration" not in captured


def _chatty_worker_body(workdir: str, tag: str) -> None:
    """Child entry point: many lines on ONE device, so interleaving would be visible."""
    configure_worker_logging(workdir, "cuda:0")
    for index in range(50):
        logger.info("{}: line = {}", tag, index)


def test_worker_logs_are_per_process_so_two_workers_never_share_a_file(tmp_path: Path) -> None:
    """Two concurrent workers on the SAME device get separate, complete files.

    A device is not a unique worker key (sequential shape tasks and repeated dispatches reuse
    each device with a fresh process), so the pid, not the device, is what
    makes a worker log unique. Sharing one file would interleave the two streams and make
    loguru's rotation unsafe across processes.
    """
    context = mp.get_context("spawn")
    processes = [context.Process(target=_chatty_worker_body, args=(str(tmp_path), tag)) for tag in ("WORKER-A", "WORKER-B")]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    logs = sorted((tmp_path / WORKER_LOG_SUBDIR).glob("cuda-0-*.log"))
    assert len(logs) == 2, f"expected one file per process, got {[p.name for p in logs]}"
    # Each file holds exactly one worker's complete, uninterleaved stream.
    tags_per_file = [{line.split(": line = ")[0].rsplit(" - ", 1)[-1] for line in path.read_text(encoding="utf-8").splitlines()} for path in logs]
    assert sorted(tags_per_file, key=sorted) == [{"WORKER-A"}, {"WORKER-B"}]
    assert [len(path.read_text(encoding="utf-8").splitlines()) for path in logs] == [50, 50]


def test_prune_worker_logs_removes_only_stale_files(tmp_path: Path) -> None:
    """Pruning retires aged worker logs and leaves fresh ones (and non-logs) alone."""
    log_dir = tmp_path / WORKER_LOG_SUBDIR
    log_dir.mkdir(parents=True)
    stale = log_dir / "cuda-0-111.log"
    fresh = log_dir / "cuda-1-222.log"
    other = log_dir / "notes.txt"
    for path in (stale, fresh, other):
        path.write_text("x", encoding="utf-8")
    old = time.time() - (30 * 24 * 60 * 60)
    os.utime(stale, (old, old))

    assert prune_worker_logs(tmp_path) == 1
    assert not stale.exists()
    assert fresh.exists()
    assert other.exists()


def test_prune_worker_logs_on_a_missing_directory_is_a_noop(tmp_path: Path) -> None:
    """A first run (no worker-log directory yet) prunes nothing rather than raising."""
    assert prune_worker_logs(tmp_path) == 0


@pytest.mark.parametrize(("name", "expected"), [("cuda:0", "cuda-0"), ("cpu", "cpu"), ("//", "worker"), (".", "worker")])
def test_worker_log_name_sanitizes_device_strings(name: str, expected: str) -> None:
    """Device strings become filesystem-safe stems; a fully stripped name falls back to 'worker'."""
    assert _worker_log_name(name) == expected


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
