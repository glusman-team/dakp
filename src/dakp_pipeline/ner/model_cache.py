"""Idempotent NER model download + cache (laptop-safe, offline-testable).

The production mode of the NER backend (:mod:`dakp_pipeline.ner.ner`) needs large model
weights. This module gives it one idempotent place to fetch and cache those weights under a
configurable directory, with a small provenance manifest (model id, source, BLAKE3 tree hash)
so a cached model is reused **by hash** — never re-downloaded on a cache hit, and verifiable
on demand.

Design constraints (mirrors the rest of DAKP):

* **No heavy imports at module load.** ``huggingface_hub`` is imported lazily, inside the
  default downloader only. The cache logic itself is stdlib + ``blake3`` (a base dep), so
  ``import dakp_pipeline.ner.model_cache`` stays light (no torch / transformers at import time).
* **Dependency-injected downloader.** :func:`ensure_model` accepts an optional ``downloader``
  callable; tests pass a fake that writes bytes locally, so idempotency, manifests, and
  drift detection are all tested with **no network and no ``huggingface_hub``**.
* **No absolute paths.** The cache dir defaults to ``<workdir>/models`` when a workdir is
  supplied, else ``$XDG_CACHE_HOME/dakp/models`` (``~/.cache/dakp/models``).

Layout under the cache dir (the ``content/`` subdir is what :func:`hash_tree` hashes, so the
manifest itself is never part of the content hash)::

    <cache>/<source>/<sanitized-model-id>/
        manifest.json     # provenance: schema_version, model_id, source, b3, retrieved_at
        content/          # the downloaded model files (what a backend loads from)
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dakp_pipeline.io.content_hash import hash_tree
from dakp_pipeline.logging_setup import logger, stats

MANIFEST_NAME = "manifest.json"
CONTENT_DIRNAME = "content"
SCHEMA_VERSION = "dakp.ner.model.v1"
DEFAULT_SOURCE = "huggingface"

# A downloader fetches ``model_id`` into ``dest`` (the content dir). Dependency-injected so
# tests never touch the network.
Downloader = Callable[[str, Path], None]


class NERDependencyError(ImportError):
    """A NER backend's heavy dependency (a core DAKP dep) is unexpectedly not importable.

    Subclasses :class:`ImportError` so callers may catch either. The message always names the
    missing module and the install command (``uv sync``).
    """


@dataclass(frozen=True)
class ModelRef:
    """Handle to a cached model: where its files live plus provenance.

    ``path`` is the ``content/`` directory a backend loads from; ``b3`` is its BLAKE3 tree
    hash; ``manifest`` is the JSON provenance file.
    """

    model_id: str
    source: str
    path: Path
    b3: str
    manifest: Path


# --- paths ---------------------------------------------------------------------


def default_model_cache_dir(workdir: Path | None = None) -> Path:
    """Default model cache dir: ``<workdir>/models`` if given, else the user cache.

    The user cache honors ``$XDG_CACHE_HOME`` and falls back to ``~/.cache/dakp/models``.
    """
    if workdir is not None:
        return Path(workdir) / "models"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "dakp" / "models"


def model_root(cache_dir: Path, model_id: str, source: str = DEFAULT_SOURCE) -> Path:
    """Per-model directory: ``<cache>/<source>/<model_id with '/' -> '--'>``."""
    return Path(cache_dir) / source / model_id.replace("/", "--")


def content_dir(root: Path) -> Path:
    """The ``content/`` subdir holding the actual model files (the hashed, loaded dir)."""
    return Path(root) / CONTENT_DIRNAME


def manifest_path(root: Path) -> Path:
    """The ``manifest.json`` provenance file for a model root."""
    return Path(root) / MANIFEST_NAME


# --- manifest I/O --------------------------------------------------------------


def read_manifest(path: Path) -> dict[str, Any] | None:
    """Read a manifest JSON, returning ``None`` if absent or unreadable/corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_manifest(path: Path, data: Mapping[str, Any]) -> None:
    """Write a manifest JSON (sorted keys, trailing newline) creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- downloaders ---------------------------------------------------------------


def default_downloader(model_id: str, dest: Path) -> None:
    """Download ``model_id`` from the Hugging Face Hub into ``dest`` (lazy import).

    ``huggingface_hub`` is a core DAKP dependency (transitively via ``gliner``); if it is
    somehow not importable, raises :class:`NERDependencyError` with the install command.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            "downloading NER model weights requires 'huggingface_hub' (a core DAKP dependency) but it is not importable. "
            "Install all dependencies with: uv sync"
        )
        raise NERDependencyError(msg) from exc
    snapshot_download(repo_id=model_id, local_dir=dest)


# --- idempotent cache entrypoint -----------------------------------------------


def ensure_model(
    model_id: str,
    *,
    source: str = DEFAULT_SOURCE,
    cache_dir: Path | str | None = None,
    workdir: Path | str | None = None,
    downloader: Downloader | None = None,
    force: bool = False,
    verify: bool = True,
) -> ModelRef:
    """Return a cached model, downloading it exactly once (idempotent).

    Cache hit (manifest present, matching id/source, and — when ``verify`` — a content tree
    hash that still matches) returns immediately without invoking the downloader. Otherwise
    the downloader fetches into ``content/``, the tree hash is recorded, and a manifest is
    written. ``force`` re-downloads unconditionally.

    Args:
        model_id: Upstream model identifier (e.g. ``"gliner-community/gliner_large-v2.5"``).
        source: Provenance label / cache subdir (default ``"huggingface"``).
        cache_dir: Explicit cache root. Wins over ``workdir`` when both are given.
        workdir: Pipeline workdir; the cache lives at ``<workdir>/models``.
        downloader: Injectable fetcher ``(model_id, dest) -> None``; defaults to the Hugging
            Face Hub downloader. Tests pass a fake to avoid network.
        force: Re-download even on a cache hit.
        verify: On a would-be hit, re-hash ``content/`` and re-download if it drifted.
    """
    event = "ner_model_cache"
    resolved_cache = Path(cache_dir) if cache_dir is not None else default_model_cache_dir(Path(workdir) if workdir is not None else None)
    root = model_root(resolved_cache, model_id, source)
    content = content_dir(root)
    manifest = manifest_path(root)

    if manifest.exists() and not force:
        data = read_manifest(manifest)
        if data is not None and data.get("model_id") == model_id and data.get("source") == source:
            cached_b3 = str(data.get("b3", ""))
            if not verify or (content.exists() and hash_tree(content) == cached_b3):
                stats(logger, event, model_id=model_id, source=source, cache_hit=True, b3=cached_b3)
                return ModelRef(model_id=model_id, source=source, path=content, b3=cached_b3, manifest=manifest)
            logger.warning("{}: cached model content drifted from manifest; re-downloading", event)
            stats(logger, event, model_id=model_id, drifted_b3=cached_b3)

    stats(logger, event, model_id=model_id, source=source, downloading=True)
    content.mkdir(parents=True, exist_ok=True)
    (downloader or default_downloader)(model_id, content)
    b3 = hash_tree(content)
    write_manifest(
        manifest, {"schema_version": SCHEMA_VERSION, "model_id": model_id, "source": source, "b3": b3, "retrieved_at": datetime.now(UTC).isoformat()}
    )
    file_count = sum(1 for path in content.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in content.rglob("*") if path.is_file())
    stats(logger, event, model_id=model_id, cache_hit=False, b3=b3, files=file_count, bytes=total_bytes)
    return ModelRef(model_id=model_id, source=source, path=content, b3=b3, manifest=manifest)


__all__ = [
    "CONTENT_DIRNAME",
    "DEFAULT_SOURCE",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "Downloader",
    "ModelRef",
    "NERDependencyError",
    "content_dir",
    "default_downloader",
    "default_model_cache_dir",
    "ensure_model",
    "manifest_path",
    "model_root",
    "read_manifest",
    "write_manifest",
]
