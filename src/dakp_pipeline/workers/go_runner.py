"""Go worker runner: locate/build the ``dakp-worker`` binary and drive its subcommands.

The Go workers (``go/cmd/dakp-worker``) do the heavy parsing/extraction as subcommands
(``dailymed`` / ``faers`` / ``drugsfda`` / ``hash``). This module shells out to the compiled
binary, captures its machine-readable **stdout** (the ``b3:<hex>`` artifact id for
``dailymed``/``faers``/``hash``; a JSON summary for ``drugsfda``), and relays its structured
``log/slog`` JSON **stderr** into the loguru logger so Go and Python logs appear uniformly
(see ``go/README.md`` "How Python invokes the worker").

Stdlib only (``subprocess`` / ``shutil`` / ``threading``); no new dependencies. The binary is
built on demand from the repo ``go/`` tree (derived from this file's location — no hardcoded
absolute paths) and cached under a hash of the Go sources, so unchanged sources reuse a prior
build. A prebuilt binary can be supplied via ``DAKP_WORKER_BIN``; the build cache directory via
``DAKP_GO_CACHE``.

The runner is monkeypatchable: tests substitute :class:`MockGoRunner` (no Go toolchain needed)
or patch :func:`go_available` / :func:`get_runner`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.io.content_hash import hash_bytes, hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import bind

# Registered dakp-worker subcommands (go/cmd/dakp-worker self-registration; see go/README.md).
SUBCOMMANDS: tuple[str, ...] = ("dailymed", "faers", "drugsfda", "hash")

# Environment overrides: a prebuilt binary, and the on-disk build cache directory.
ENV_BINARY = "DAKP_WORKER_BIN"
ENV_CACHE_DIR = "DAKP_GO_CACHE"

# slog JSON level -> loguru level name (slog uses "WARN"; loguru uses "WARNING").
_SLOG_LEVELS = {"DEBUG": "DEBUG", "INFO": "INFO", "WARN": "WARNING", "WARNING": "WARNING", "ERROR": "ERROR"}
# slog fields that are envelope metadata, not structured context to relay.
_SLOG_ENVELOPE = {"level", "msg", "message", "time", "ts"}


class GoUnavailableError(RuntimeError):
    """Raised when the Go toolchain / a prebuilt ``dakp-worker`` binary cannot be found."""


class GoWorkerError(RuntimeError):
    """Raised when a ``dakp-worker`` subcommand exits non-zero."""

    def __init__(self, subcommand: str, returncode: int, stderr: str) -> None:
        self.subcommand = subcommand
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.splitlines()[-20:])
        super().__init__(f"dakp-worker {subcommand} exited {returncode}; last stderr lines:\n{tail}")


@dataclass(frozen=True)
class GoResult:
    """Outcome of one ``dakp-worker`` subcommand invocation.

    ``stdout`` is the captured machine-readable result (stripped); ``stderr`` is the raw log
    stream; ``logs`` holds the parsed ``log/slog`` JSON records (non-JSON lines are relayed to
    the logger but omitted here).
    """

    subcommand: str
    returncode: int
    stdout: str
    stderr: str
    logs: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def artifact_id(self) -> str | None:
        """The ``b3:<hex>`` id on stdout (dailymed/faers/hash emit a single id line)."""
        for line in self.stdout.splitlines():
            text = line.strip()
            if text.startswith("b3:"):
                return text
        return None

    @property
    def summary(self) -> dict[str, Any] | None:
        """The parsed JSON summary on stdout (drugsfda emits one JSON object), else ``None``."""
        text = self.stdout.strip()
        if not text or not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


# --- toolchain / binary discovery ------------------------------------------------


def _repo_root() -> Path:
    """Repository root, derived from this file's location (no hardcoded absolute paths)."""
    # src/dakp_pipeline/workers/go_runner.py -> parents[3] is the repo root.
    return Path(__file__).resolve().parents[3]


def _default_go_dir() -> Path:
    return _repo_root() / "go"


def _default_cache_dir() -> Path:
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return Path(override)
    # ``<repo>/tmp/`` is gitignored, so cached binaries never pollute the working tree.
    return _repo_root() / "tmp" / "dakp-go"


def go_available() -> bool:
    """Whether a ``dakp-worker`` can be produced: a prebuilt binary exists or ``go`` is on PATH.

    Monkeypatchable: tests patch this (or ``shutil.which``) to simulate a Go-less host.
    """
    prebuilt = os.environ.get(ENV_BINARY)
    if prebuilt and Path(prebuilt).is_file():
        return True
    return shutil.which("go") is not None


def hash_go_sources(go_dir: Path) -> str:
    """Deterministic short hex digest of the Go sources (``*.go`` + ``go.mod`` + ``go.sum``).

    Used to key the build cache so unchanged sources reuse a prior binary. Test fixtures and
    other non-source files are intentionally excluded so they never trigger a rebuild.
    """
    files = sorted(path for path in go_dir.rglob("*") if path.is_file() and (path.suffix == ".go" or path.name in {"go.mod", "go.sum"}))
    parts = [f"{path.relative_to(go_dir).as_posix()}\x00{hash_file(path)}\x00" for path in files]
    return hash_bytes("".join(parts).encode("utf-8")).split(":", 1)[1][:16]


# --- stderr relay ----------------------------------------------------------------


def _relay_slog(line: str, log: Any) -> dict[str, Any] | None:
    """Relay one stderr line into the loguru logger; return the parsed record if it was JSON.

    ``log/slog`` JSON lines are logged at their mapped level with the structured fields bound;
    anything else is logged verbatim at INFO so no Go output is lost.
    """
    text = line.strip()
    if not text:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        log.info(text)
        return None
    if not isinstance(record, dict):
        log.info(text)
        return None
    level = _SLOG_LEVELS.get(str(record.get("level", "INFO")).upper(), "INFO")
    message = str(record.get("msg", record.get("message", "")))
    fields = {key: value for key, value in record.items() if key not in _SLOG_ENVELOPE}
    bound = log.bind(**fields) if fields else log
    bound.log(level, message)
    return record


# --- the runner ------------------------------------------------------------------


class GoRunner:
    """Locate/build the ``dakp-worker`` binary and run its subcommands via subprocess."""

    def __init__(self, *, binary: Path | None = None, go_dir: Path | None = None, cache_dir: Path | None = None, log: Any | None = None) -> None:
        self._binary = binary
        self._go_dir = go_dir if go_dir is not None else _default_go_dir()
        self._cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()
        self._log = log if log is not None else bind(task_id="go_runner")
        self._resolved: Path | None = None

    @property
    def go_dir(self) -> Path:
        return self._go_dir

    def ensure_binary(self) -> Path:
        """Return a usable ``dakp-worker`` binary path, building (and caching) it if needed.

        Resolution order: explicit ``binary`` -> ``DAKP_WORKER_BIN`` -> cached build keyed by a
        hash of the Go sources. Raises :class:`GoUnavailableError` when Go cannot be found.
        """
        if self._resolved is not None:
            return self._resolved

        if self._binary is not None:
            if not self._binary.is_file():
                msg = f"configured dakp-worker binary does not exist: {self._binary}"
                raise GoUnavailableError(msg)
            self._resolved = self._binary
            return self._resolved

        prebuilt = os.environ.get(ENV_BINARY)
        if prebuilt:
            path = Path(prebuilt)
            if not path.is_file():
                msg = f"{ENV_BINARY}={prebuilt} does not exist"
                raise GoUnavailableError(msg)
            self._resolved = path
            return self._resolved

        if shutil.which("go") is None:
            msg = "Go toolchain not found on PATH and no prebuilt dakp-worker binary configured (set DAKP_WORKER_BIN)"
            raise GoUnavailableError(msg)

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        binary = self._cache_dir / f"dakp-worker-{hash_go_sources(self._go_dir)}"
        if not binary.is_file():
            self._build(binary)
        self._resolved = binary
        return binary

    def _build(self, out: Path) -> None:
        command = ["go", "build", "-o", str(out), "./cmd/dakp-worker"]
        self._log.info("building dakp-worker", go_dir=str(self._go_dir), out=str(out))
        proc = subprocess.run(command, cwd=self._go_dir, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            msg = f"go build failed (exit {proc.returncode}) in {self._go_dir}:\n{proc.stderr.strip()}"
            raise GoUnavailableError(msg)

    def run(self, subcommand: str, args: Sequence[str | Path] = ()) -> GoResult:
        """Run ``dakp-worker <subcommand> <args...>``, streaming stderr into the logger.

        Captures stdout in full and relays each stderr line (slog JSON) as it arrives. Raises
        :class:`GoWorkerError` on a non-zero exit code.
        """
        binary = self.ensure_binary()
        argv = [str(binary), subcommand, *(str(arg) for arg in args)]
        return self._exec(subcommand, argv)

    def run_table(self, subcommand: str, input_dir: Path, output_dir: Path, extra_args: Sequence[str | Path] = ()) -> GoResult:
        """Convenience for the extractor subcommands: ``<subcommand> [extra...] <in> <out>``."""
        return self.run(subcommand, [*extra_args, str(input_dir), str(output_dir)])

    def _exec(self, subcommand: str, argv: list[str]) -> GoResult:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # argv[0] is the resolved binary path
        if proc.stdout is None or proc.stderr is None:  # pragma: no cover - PIPE guarantees these
            msg = "failed to open pipes for dakp-worker subprocess"
            raise GoWorkerError(subcommand, -1, msg)

        # Drain stdout on a background thread (it is small) while the main thread streams
        # stderr line-by-line into the logger; reading only one pipe on the main thread would
        # otherwise risk a deadlock if the other pipe's buffer filled.
        stdout_chunks: list[bytes] = []

        def _drain_stdout() -> None:
            assert proc.stdout is not None
            stdout_chunks.append(proc.stdout.read())

        reader = threading.Thread(target=_drain_stdout, daemon=True)
        reader.start()

        logs: list[dict[str, Any]] = []
        stderr_lines: list[str] = []
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            stderr_lines.append(line)
            record = _relay_slog(line, self._log)
            if record is not None:
                logs.append(record)

        returncode = proc.wait()
        reader.join()
        stdout = b"".join(stdout_chunks).decode("utf-8", "replace").strip()
        stderr = "\n".join(stderr_lines)
        result = GoResult(subcommand=subcommand, returncode=returncode, stdout=stdout, stderr=stderr, logs=tuple(logs))
        if returncode != 0:
            raise GoWorkerError(subcommand, returncode, stderr)
        return result


# --- test double -----------------------------------------------------------------


class MockGoRunner:
    """In-memory stand-in for :class:`GoRunner` (no Go toolchain required).

    Routes subcommands to canned stdout/stderr so tests can assert subcommand routing, stdout
    parsing (``artifact_id`` / ``summary``), and stderr relay without building anything. Set
    ``go_present=False`` to simulate a host without Go (``ensure_binary`` / ``run`` raise
    :class:`GoUnavailableError`). Register per-subcommand handlers via :meth:`set_handler`.
    """

    def __init__(self, *, go_present: bool = True) -> None:
        self.go_present = go_present
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self._handlers: dict[str, Any] = {}
        self._default_stdout: dict[str, str] = {
            "hash": "b3:" + "0" * 64,
            "dailymed": "b3:" + "1" * 64,
            "faers": "b3:" + "2" * 64,
            "drugsfda": json.dumps({"tables": {"drugsfda_products.tsv": {"artifact_id": "b3:" + "3" * 64, "rows": 1}}}),
        }

    def set_handler(self, subcommand: str, handler: Any) -> None:
        """Register ``handler(args: tuple[str, ...]) -> (stdout, stderr)`` for a subcommand."""
        self._handlers[subcommand] = handler

    def ensure_binary(self) -> Path:
        if not self.go_present:
            msg = "mock: Go toolchain unavailable"
            raise GoUnavailableError(msg)
        return Path("/mock/dakp-worker")

    def run(self, subcommand: str, args: Sequence[str | Path] = ()) -> GoResult:
        self.ensure_binary()
        arg_tuple = tuple(str(arg) for arg in args)
        self.calls.append((subcommand, arg_tuple))
        if subcommand in self._handlers:
            stdout, stderr = self._handlers[subcommand](arg_tuple)
        else:
            stdout = self._default_stdout.get(subcommand, "")
            stderr = json.dumps({"level": "INFO", "msg": f"mock {subcommand}", "task_id": f"extract_{subcommand}", "warnings": 0})
        logs: list[dict[str, Any]] = []
        for line in stderr.splitlines():
            record = _relay_slog(line, bind(task_id="mock_go_runner"))
            if record is not None:
                logs.append(record)
        return GoResult(subcommand=subcommand, returncode=0, stdout=stdout.strip(), stderr=stderr, logs=tuple(logs))

    def run_table(self, subcommand: str, input_dir: Path, output_dir: Path, extra_args: Sequence[str | Path] = ()) -> GoResult:
        return self.run(subcommand, [*extra_args, str(input_dir), str(output_dir)])


# --- module-level accessor + delegation gate -------------------------------------

_RUNNER: GoRunner | MockGoRunner | None = None


def get_runner() -> GoRunner | MockGoRunner:
    """Return the process-wide runner (built lazily). Monkeypatchable via :func:`set_runner`."""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = GoRunner()
    return _RUNNER


def set_runner(runner: GoRunner | MockGoRunner | None) -> None:
    """Override (or reset, with ``None``) the process-wide runner — the test seam."""
    global _RUNNER
    _RUNNER = runner


def should_use_go(ctx: TaskContext, *, override: bool | None = None) -> bool:
    """Delegation gate: the ``use_go_workers`` flag is on AND a Go worker is available.

    The flag comes from ``override`` if given, else ``ctx.params["use_go_workers"]`` (populated
    from :class:`~dakp_pipeline.config.Profile`). Short-circuits on the flag so the common
    (Python) path never even probes for Go.
    """
    flag = override if override is not None else bool(ctx.params.get("use_go_workers", False))
    return bool(flag and go_available())


# --- helpers shared by the delegating extractors ---------------------------------


def stage_inputs(refs: Sequence[ArtifactRef], stage_dir: Path) -> Path:
    """Materialize the input artifacts' files into ``stage_dir`` (hardlink, copy on failure).

    The Go extractor subcommands read a directory of loose source files; this presents a
    ``list[ArtifactRef]`` (possibly scattered across the content-addressed store) as one
    directory, preserving basenames (the Go classifiers key off the filename).
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    for index, ref in enumerate(refs):
        dest = stage_dir / ref.uri.name
        if dest.exists():
            dest = stage_dir / f"{index:04d}_{ref.uri.name}"
        try:
            os.link(ref.uri, dest)
        except OSError:
            shutil.copy2(ref.uri, dest)
    return stage_dir


def read_go_tsv(path: Path) -> pl.DataFrame:
    """Read an uncompressed TSV produced by a Go worker as an all-Utf8 frame (nulls -> "").

    ``infer_schema_length=0`` keeps every column Utf8 — matching the Python extractors' interim
    tables — so reading a Go TSV back never loses leading zeroes or id formatting.
    """
    return pl.read_csv(path, separator="\t", infer_schema_length=0).fill_null("")


def go_rows(frame: pl.DataFrame) -> list[dict[str, str]]:
    """A Go TSV frame as ``list[dict[str, str]]`` for the extractors' row-based writers."""
    return [{key: str(value) for key, value in row.items()} for row in frame.to_dicts()]


def go_warnings(result: GoResult) -> int:
    """Best-effort warning count from a Go result's slog summary (``warnings`` field), else 0."""
    for record in reversed(result.logs):
        if "warnings" in record:
            try:
                return int(record["warnings"])
            except (TypeError, ValueError):
                continue
    return 0


__all__ = [
    "ENV_BINARY",
    "ENV_CACHE_DIR",
    "SUBCOMMANDS",
    "GoResult",
    "GoRunner",
    "GoUnavailableError",
    "GoWorkerError",
    "MockGoRunner",
    "get_runner",
    "go_available",
    "go_rows",
    "go_warnings",
    "hash_go_sources",
    "read_go_tsv",
    "set_runner",
    "should_use_go",
    "stage_inputs",
]
