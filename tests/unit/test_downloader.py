"""Tests for ``dakp_pipeline.io.downloader`` (aria2c-accelerated downloads + urllib fallback).

The suite-wide ``DAKP_ARIA2=0`` fixture (``tests/conftest.py``) keeps these tests deterministic:
they opt back into aria2c by setting ``DAKP_ARIA2=1`` (and stubbing ``resolve_aria2c`` /
``subprocess.run``) so no real binary or network is needed for branch coverage. One real-binary
smoke test (skipped when aria2c is absent) confirms the bundled wheel actually runs over a
loopback HTTP server.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

from dakp_pipeline.io import downloader

# --- resolve_aria2c: env gate + bundled > system > None -------------------------


def test_resolve_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "0")
    assert downloader.resolve_aria2c() is None


def test_resolve_prefers_bundled_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "1")
    bundled = tmp_path / "aria2c"
    bundled.write_text("#")
    monkeypatch.setattr(downloader, "_bundled_aria2c", lambda: bundled)
    monkeypatch.setattr(downloader.shutil, "which", lambda name: "/usr/bin/aria2c")
    assert downloader.resolve_aria2c() == bundled


def test_resolve_falls_back_to_system_aria2c(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "1")
    monkeypatch.setattr(downloader, "_bundled_aria2c", lambda: None)
    monkeypatch.setattr(downloader.shutil, "which", lambda name: "/usr/bin/aria2c")
    assert downloader.resolve_aria2c() == Path("/usr/bin/aria2c")


def test_resolve_returns_none_when_nothing_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "1")
    monkeypatch.setattr(downloader, "_bundled_aria2c", lambda: None)
    monkeypatch.setattr(downloader.shutil, "which", lambda name: None)
    assert downloader.resolve_aria2c() is None


def test_resolve_ignores_bundled_path_that_does_not_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "1")
    monkeypatch.setattr(downloader, "_bundled_aria2c", lambda: Path("/nonexistent/aria2c"))
    monkeypatch.setattr(downloader.shutil, "which", lambda name: None)
    assert downloader.resolve_aria2c() is None


# --- _build_aria2c_args: flag construction (UA, custom headers, connections) -----


def test_build_args_with_user_agent_and_custom_header(tmp_path: Path) -> None:
    exe, dest = tmp_path / "aria2c", tmp_path / "o" / "f.zip"
    args = downloader._build_aria2c_args(exe, "https://x/y.zip", dest, timeout=60, headers={"User-Agent": "UA", "X-Custom": "v"}, connections=16)
    assert args[0] == str(exe)
    assert args[-1] == "https://x/y.zip"  # URL is the final positional
    assert "--max-connection-per-server=16" in args
    assert "--split=16" in args
    assert "--user-agent" in args
    assert args[args.index("--user-agent") + 1] == "UA"
    assert "--header" in args
    assert args[args.index("--header") + 1] == "X-Custom: v"
    assert args[args.index("--dir") + 1] == str(dest.parent)
    assert args[args.index("--out") + 1] == "f.zip"
    assert f"--timeout={60}" in args


def test_build_args_without_headers_omits_user_agent_and_header(tmp_path: Path) -> None:
    exe, dest = tmp_path / "aria2c", tmp_path / "o" / "f.zip"
    args = downloader._build_aria2c_args(exe, "https://x/y.zip", dest, timeout=30, headers=None, connections=8)
    assert "--user-agent" not in args
    assert "--header" not in args
    assert "--max-connection-per-server=8" in args
    assert "--connect-timeout=30" in args  # min(60, 30)


# --- aria2_download: subprocess wiring + failure modes --------------------------


def test_aria2_download_invokes_subprocess_and_writes_dest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "aria2c"
    exe.write_text("#")
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> int:
        captured["args"] = args
        out = Path(args[args.index("--dir") + 1]) / args[args.index("--out") + 1]  # type: ignore[arg-type]
        out.write_bytes(b"aria2c-payload")
        return 0

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    dest = tmp_path / "out" / "file.zip"
    result = downloader.aria2_download("https://x/y.zip", dest, timeout=60, executable=exe)
    assert result == dest
    assert dest.read_bytes() == b"aria2c-payload"
    assert captured["args"]


def test_aria2_download_applies_default_user_agent_when_omitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "aria2c"
    exe.write_text("#")
    seen: list[str] = []
    monkeypatch.setattr(downloader.subprocess, "run", lambda args, **kw: seen.extend(args) or 0)
    downloader.aria2_download("https://x/y.zip", tmp_path / "o" / "f.zip", timeout=60, executable=exe)
    assert "--user-agent" in seen
    assert seen[seen.index("--user-agent") + 1] == downloader._USER_AGENT


def test_aria2_download_preserves_supplied_user_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "aria2c"
    exe.write_text("#")
    seen: list[str] = []
    monkeypatch.setattr(downloader.subprocess, "run", lambda args, **kw: seen.extend(args) or 0)
    downloader.aria2_download("https://x/y.zip", tmp_path / "o" / "f.zip", timeout=60, executable=exe, headers={"User-Agent": "custom-ua"})
    assert seen[seen.index("--user-agent") + 1] == "custom-ua"


def test_aria2_download_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "aria2c"
    exe.write_text("#")
    monkeypatch.setattr(downloader.subprocess, "run", lambda args, **kw: (_ for _ in ()).throw(subprocess.CalledProcessError(1, args)))
    with pytest.raises(subprocess.CalledProcessError):
        downloader.aria2_download("https://x/y.zip", tmp_path / "o" / "f.zip", executable=exe)


def test_aria2_download_without_executable_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "1")
    monkeypatch.setattr(downloader, "_bundled_aria2c", lambda: None)
    monkeypatch.setattr(downloader.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="aria2c unavailable"):
        downloader.aria2_download("https://x/y.zip", Path("/tmp/f.zip"))


# --- stream_download: stdlib fallback -------------------------------------------


def test_stream_download_copies_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello world")
    dest = tmp_path / "out" / "dst.bin"
    assert downloader.stream_download(src.as_uri(), dest, timeout=10) == dest
    assert dest.read_bytes() == b"hello world"


def test_stream_download_passes_headers_to_request(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"with-headers")
    dest = tmp_path / "dst.bin"
    # file:// ignores headers, but building the Request with a non-empty headers dict exercises
    # the `headers or {}` truthy branch.
    assert downloader.stream_download(src.as_uri(), dest, timeout=10, headers={"X-Test": "1"}) == dest
    assert dest.read_bytes() == b"with-headers"


# --- download: backend selection + fallback -------------------------------------


def test_download_uses_aria2c_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "1")
    exe = tmp_path / "aria2c"
    exe.write_text("#")
    monkeypatch.setattr(downloader, "_bundled_aria2c", lambda: exe)
    monkeypatch.setattr(
        downloader.subprocess,
        "run",
        lambda args, **kw: (Path(args[args.index("--dir") + 1]) / args[args.index("--out") + 1]).write_bytes(b"fast") or 0,
    )
    dest = tmp_path / "out" / "f.zip"
    assert downloader.download("https://x/y.zip", dest, timeout=60) == dest
    assert dest.read_bytes() == b"fast"


def test_download_falls_back_to_urllib_on_aria2c_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "1")
    exe = tmp_path / "aria2c"
    exe.write_text("#")
    monkeypatch.setattr(downloader, "_bundled_aria2c", lambda: exe)
    monkeypatch.setattr(downloader.subprocess, "run", lambda args, **kw: (_ for _ in ()).throw(subprocess.CalledProcessError(1, args)))
    src = tmp_path / "src.bin"
    src.write_bytes(b"fallback")
    dest = tmp_path / "out" / "f.bin"
    assert downloader.download(src.as_uri(), dest, timeout=10) == dest
    assert dest.read_bytes() == b"fallback"


def test_download_uses_urllib_when_aria2c_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DAKP_ARIA2", "0")
    src = tmp_path / "s.bin"
    src.write_bytes(b"plain")
    dest = tmp_path / "out" / "d.bin"
    assert downloader.download(src.as_uri(), dest, timeout=10) == dest
    assert dest.read_bytes() == b"plain"


# --- real-binary smoke (skipped when aria2c is absent) --------------------------


def test_real_aria2c_downloads_over_loopback_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The bundled aria2c actually runs and fetches a file over a loopback HTTP server."""
    monkeypatch.delenv("DAKP_ARIA2", raising=False)  # re-enable (suite fixture disables it)
    exe = downloader.resolve_aria2c()
    if exe is None:
        pytest.skip("aria2c binary not available")

    payload = b"x" * 4096
    served_dir = tmp_path / "srv"
    served_dir.mkdir()
    (served_dir / "blob.bin").write_bytes(payload)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served_dir))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            dest = tmp_path / "out" / "blob.bin"
            downloader.aria2_download(f"http://127.0.0.1:{port}/blob.bin", dest, timeout=30, executable=exe)
            assert dest.read_bytes() == payload
        finally:
            httpd.shutdown()
            httpd.server_close()
