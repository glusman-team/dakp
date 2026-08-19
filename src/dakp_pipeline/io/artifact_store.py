"""Content-addressed artifact store.

Every raw download and derived artifact is hashed with BLAKE3 and stored immutably under
``data/raw/by-hash/<hex>/``; human-readable aliases (under ``data/raw/aliases/``) point at
the hash. Re-ingesting an identical artifact is a cache hit (no copy, manifest reused).
Reuse is keyed by content hash, never by filename or mtime.

Two ingest modes:

* :meth:`ArtifactStore.ingest`  — copy an external file (e.g. a fixture or a download)
  into the content-addressed store and return a ref to the stored copy.
* :meth:`ArtifactStore.register` — register an artifact that already lives in the workdir
  (e.g. an interim parquet or a generated TSV) in place; hash + manifest only, no copy.

A small operation index (``data/manifests/_index.json``, see :func:`op_index_key`) maps
``operation + sorted input ids`` to the outputs registered for them, so extract/shape tasks
can skip work whose inputs have not changed (:meth:`ArtifactStore.find_by_operation`).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from dakp_pipeline.io.content_hash import digest_dirname, hash_bytes, hash_file_with_sri, hash_tree
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.io.downloads import infer_media_type
from dakp_pipeline.io.manifests import ArtifactManifest, EnvironmentBlock, HashBlock, OperationBlock, SourceBlock, TableBlock, read_manifest
from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.paths import Workdir

#: Name of the operation-index sidecar file inside ``data/manifests/`` (see :func:`op_index_key`).
OP_INDEX_FILENAME = "_index.json"

#: Shared Python/Go operation-index key spec (mirrored byte-for-byte by
#: ``go/internal/airflow/opindex.go`` — keep the two in lockstep):
#:
#: * ``canonical = operation + "|" + "|".join(sorted(inputs))`` (UTF-8; ``sorted`` is Unicode
#:   code-point order, which matches Go's byte-wise ``sort.Strings`` for these ASCII ids);
#: * ``key = hex(BLAKE3(canonical))`` — 64 lowercase hex chars, WITHOUT the ``b3:`` prefix.
#:
#: The index JSON is ``{"version": 1, "entries": {<key>: {"operation": str, "inputs":
#: [sorted ids], "outputs": [{"artifact_id", "path", "media_type", "rows",
#: "schema_fingerprint"}]}}}``.
OP_INDEX_KEY_SPEC = "operation + '|' + '|'.join(sorted(inputs)) -> hex(BLAKE3)"


def op_index_key(operation: str, inputs: list[str]) -> str:
    """The 64-hex BLAKE3 index key for an operation + its exact input id set.

    Inputs are sorted before hashing so call-site ordering never fragments the cache.
    """
    canonical = f"{operation}|{'|'.join(sorted(inputs))}"
    return digest_dirname(hash_bytes(canonical.encode("utf-8")))


class ArtifactStore:
    """BLAKE3 content-addressed store bound to a :class:`~dakp_pipeline.paths.Workdir`."""

    def __init__(self, workdir: Workdir) -> None:
        self.workdir = workdir
        self._by_hash = workdir.by_hash
        self._aliases = workdir.aliases
        self._manifests = workdir.manifests

    # -- paths -----------------------------------------------------------------
    def stored_path(self, artifact_id: str, filename: str) -> Path:
        """Compute the immutable store path for an artifact id + filename."""
        return self._by_hash / digest_dirname(artifact_id) / filename

    def manifest_path(self, artifact_id: str) -> Path:
        return self._manifests / f"{digest_dirname(artifact_id)}.json"

    # -- ingest (copy into store) ---------------------------------------------
    def ingest(
        self,
        src: Path,
        *,
        media_type: str | None = None,
        alias: str | None = None,
        inputs: list[str] | None = None,
        operation: OperationBlock | None = None,
        source: SourceBlock | None = None,
        environment: EnvironmentBlock | None = None,
    ) -> tuple[ArtifactRef, bool]:
        """Content-addressed ingest of an external file.

        Copies ``src`` into ``by-hash/<hex>/<name>`` unless an identical artifact is
        already present (cache hit). Writes an alias and a manifest. Returns the
        :class:`ArtifactRef` and a ``cache_hit`` flag.
        """
        if not src.exists():
            msg = f"cannot ingest missing file: {src}"
            raise FileNotFoundError(msg)

        artifact_id, computed_sri = hash_file_with_sri(src)  # one read of the source
        dest = self.stored_path(artifact_id, src.name)
        cache_hit = dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cache_hit:
            fd, staged_name = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
            os.close(fd)
            staged = Path(staged_name)
            try:
                shutil.copy2(src, staged)
                os.replace(staged, dest)
            finally:
                staged.unlink(missing_ok=True)

        if alias is not None:
            self._write_alias(alias, artifact_id, dest)

        media = media_type if media_type is not None else infer_media_type(src)
        existing_sri: str | None = None
        if cache_hit and self.manifest_path(artifact_id).exists():
            with suppress(OSError, ValueError):
                existing_sri = read_manifest(self.manifest_path(artifact_id)).hash.sha256_sri
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            path=str(dest),
            media_type=media,
            hash=HashBlock(file=artifact_id, sha256_sri=existing_sri if existing_sri is not None else computed_sri),
            inputs=list(inputs) if inputs else [],
            operation=operation,
            source=source if source is not None else SourceBlock(),
            environment=environment if environment is not None else EnvironmentBlock(),
        )
        manifest.write(self.manifest_path(artifact_id))

        # Per-artifact detail is DEBUG (DailyMed ingests tens of thousands of SPL docs); the
        # calling stage emits the INFO-level narration for its own artifacts.
        stats(
            logger,
            "store ingest",
            level="DEBUG",
            alias=alias if alias is not None else "-",
            path=str(dest),
            blake3=artifact_id,
            bytes=src.stat().st_size,
            cache_hit=cache_hit,
            media_type=media,
        )
        ref = ArtifactRef(uri=dest, blake3=artifact_id, media_type=media, manifest=self.manifest_path(artifact_id))
        return ref, cache_hit

    # -- register (in-place, for workdir outputs) -----------------------------
    def register(
        self,
        path: Path,
        *,
        media_type: str | None = None,
        rows: int | None = None,
        schema_fingerprint: str | None = None,
        inputs: list[str] | None = None,
        operation: OperationBlock | None = None,
        source: SourceBlock | None = None,
        table: TableBlock | None = None,
        is_tree: bool = False,
    ) -> ArtifactRef:
        """Register an artifact already living in the workdir (no copy).

        Used for interim parquet and generated TSV outputs. Hashes in place, writes a
        manifest, and returns an :class:`ArtifactRef` whose ``uri`` is ``path``.
        """
        if not path.exists():
            msg = f"cannot register missing path: {path}"
            raise FileNotFoundError(msg)

        if is_tree:
            artifact_id = hash_tree(path)
            sri = None
        else:
            artifact_id, sri = hash_file_with_sri(path)  # one read of the file

        media = media_type if media_type is not None else infer_media_type(path)
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            path=str(path),
            media_type=media,
            hash=HashBlock(file=None if is_tree else artifact_id, tree=artifact_id if is_tree else None, sha256_sri=sri),
            inputs=list(inputs) if inputs else [],
            operation=operation,
            source=source if source is not None else SourceBlock(),
            table=table if table is not None else TableBlock(rows=rows, schema_fingerprint=schema_fingerprint),
        )
        manifest.write(self.manifest_path(artifact_id))
        stats(
            logger,
            "store register",
            level="DEBUG",
            path=str(path),
            blake3=artifact_id,
            bytes=path.stat().st_size,
            rows=rows if rows is not None else "-",
            media_type=media,
        )
        if operation is not None and inputs:
            self._index_record(
                operation.name,
                list(inputs),
                {"artifact_id": artifact_id, "path": str(path), "media_type": media, "rows": rows, "schema_fingerprint": schema_fingerprint},
            )
        return ArtifactRef(
            uri=path, blake3=artifact_id, media_type=media, rows=rows, schema_fingerprint=schema_fingerprint, manifest=self.manifest_path(artifact_id)
        )

    # -- operation index (already-done skip) -------------------------------------
    def find_by_operation(self, operation: str, inputs: list[str]) -> list[ArtifactRef] | None:
        """Refs to previously-registered outputs of ``operation`` over exactly ``inputs``, or None.

        Backs the "already done" skip: a task whose inputs (artifact ids + any synthetic config
        fingerprint) match a prior run's registered outputs can return those refs instead of
        recomputing. The lookup is an O(1) read of the :data:`OP_INDEX_FILENAME` sidecar —
        never a scan of every manifest. Every referenced output path and manifest must still
        exist and parse; a stale entry (deleted artifact, torn write) is pruned and reported
        as a miss so the caller simply redoes the work.
        """
        if not inputs:
            return None
        key = op_index_key(operation, inputs)
        entry = self._index_read().get("entries", {}).get(key)
        if not isinstance(entry, dict):
            return None
        outputs = entry.get("outputs")
        if (
            entry.get("operation") != operation
            or entry.get("inputs") != sorted(inputs)
            or not isinstance(outputs, list)
            or not outputs
            or not all(isinstance(output, dict) for output in outputs)
        ):
            return None

        refs: list[ArtifactRef] = []
        for output in outputs:
            try:
                artifact_id = str(output["artifact_id"])
                uri = Path(str(output["path"]))
                media = str(output["media_type"])
            except KeyError:
                return self._index_prune(key)
            manifest = self.read_manifest(artifact_id)
            # The manifest must exist and belong to the entry; per-manifest operation/inputs
            # equality is deliberately NOT enforced — multi-output fan-out tasks (the Go
            # extractors) register each member under its own per-artifact operation name and
            # parsed-input id list, while the entry records the task-level operation.
            if not uri.exists() or manifest is None or manifest.artifact_id != artifact_id:
                return self._index_prune(key)
            rows = output.get("rows")
            refs.append(
                ArtifactRef(
                    uri=uri,
                    blake3=artifact_id,
                    media_type=media,
                    rows=rows if isinstance(rows, int) else None,
                    schema_fingerprint=output.get("schema_fingerprint"),
                    manifest=self.manifest_path(artifact_id),
                )
            )
        return refs

    def record_operation(self, operation: str, inputs: list[str], refs: Iterable[ArtifactRef]) -> None:
        """Atomically replace the index entry for ``operation`` + ``inputs`` with ``refs``.

        Used by multi-output tasks (and tests) to publish the full ordered output set in one
        shot; :meth:`register` maintains single-artifact entries incrementally on its own.
        """
        outputs = [
            {
                "artifact_id": ref.blake3,
                "path": str(ref.uri),
                "media_type": ref.media_type,
                "rows": ref.rows,
                "schema_fingerprint": ref.schema_fingerprint,
            }
            for ref in refs
        ]
        with self._index_locked() as data:
            data["entries"][op_index_key(operation, inputs)] = {"operation": operation, "inputs": sorted(inputs), "outputs": outputs}

    def _index_path(self) -> Path:
        return self._manifests / OP_INDEX_FILENAME

    def _index_read(self) -> dict[str, Any]:
        """Best-effort read of the index (writers replace atomically, so reads need no lock)."""
        try:
            data = json.loads(self._index_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "entries": {}}
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            return {"version": 1, "entries": {}}
        return data

    def _index_record(self, operation: str, inputs: list[str], output: dict[str, object]) -> None:
        """Add/replace one output under its operation+inputs key (deduped by output path).

        Path dedup keeps re-runs idempotent: extract/shape outputs live at fixed workdir
        paths, so a re-run replaces the prior artifact id at that path instead of
        accumulating stale duplicates.
        """
        with self._index_locked() as data:
            key = op_index_key(operation, inputs)
            entry: dict[str, Any] = data["entries"].get(key)
            if not isinstance(entry, dict) or entry.get("operation") != operation or entry.get("inputs") != sorted(inputs):
                entry = {"operation": operation, "inputs": sorted(inputs), "outputs": []}
            outputs = [existing for existing in entry["outputs"] if isinstance(existing, dict) and existing.get("path") != output["path"]]
            outputs.append(output)
            entry["outputs"] = outputs
            data["entries"][key] = entry

    def _index_prune(self, key: str) -> None:
        """Drop a stale index entry (missing artifact/manifest); the caller reports a miss."""
        with self._index_locked() as data:
            data["entries"].pop(key, None)
        return

    @contextmanager
    def _index_locked(self) -> Iterator[dict[str, Any]]:
        """Exclusive-flock read-modify-write of the operation index (atomic rename on exit).

        Registration is sequential within a task but concurrent ACROSS tasks (the three Go
        extracts run in parallel), so the read-modify-write is serialized on a kernel flock —
        the same mechanism as the per-GPU model locks in ``ner/ner.py``.
        """
        self._manifests.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._manifests / "_index.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            data = self._index_read()
            data.setdefault("version", 1)
            yield data
            self._atomic_write(self._index_path(), json.dumps(data, indent=2))
        finally:
            os.close(fd)

    # -- read ------------------------------------------------------------------
    def read_manifest(self, artifact_id: str) -> ArtifactManifest | None:
        """Read the manifest for an artifact id, or ``None`` if absent."""
        path = self.manifest_path(artifact_id)
        return read_manifest(path) if path.exists() else None

    def cached_ref(self, alias: str) -> ArtifactRef | None:
        """Reconstruct the ref of an already-ingested artifact from its alias.

        Reads the alias + sibling ``.path`` pointer written by :meth:`ingest`; ``None`` when
        either is missing. Callers must still verify ``ref.uri.exists()`` before reuse — the
        alias record can outlive a file removed from the store.
        """
        id_path = self._aliases / alias
        if not id_path.exists():
            return None
        artifact_id = id_path.read_text(encoding="utf-8").strip()
        path_file = self._aliases / f"{alias}.path"
        if not path_file.exists():
            return None
        uri = Path(path_file.read_text(encoding="utf-8").strip())
        return ArtifactRef(uri=uri, blake3=artifact_id, media_type=infer_media_type(uri), manifest=self.manifest_path(artifact_id))

    def cached_refs(self, alias: str, artifact_id: str) -> list[ArtifactRef] | None:
        """Return a completed, validated fan-out cache set, or ``None``.

        ``alias`` identifies the source artifact and ``artifact_id`` prevents member aliases from
        an older replacement of the same fixed-name source ZIP being reused. The completion record
        is written only after expansion succeeds, so an interrupted expansion falls through to
        the normal recovery path.
        """
        marker = self._cached_refs_marker(alias)
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            member_aliases = record["aliases"]
            if (
                record["artifact_id"] != artifact_id
                or not isinstance(member_aliases, list)
                or not all(isinstance(item, str) for item in member_aliases)
            ):
                return None
        except (OSError, ValueError, KeyError, TypeError):
            return None

        prefix = f"{alias}::"
        if any(not member_alias.startswith(prefix) for member_alias in member_aliases) or len(set(member_aliases)) != len(member_aliases):
            return None
        parent = self._aliases / alias.rsplit("/", 1)[0]
        name_prefix = alias.rsplit("/", 1)[-1] + "::"
        try:
            actual_aliases = {
                path.relative_to(self._aliases).as_posix()
                for path in parent.iterdir()
                if path.is_file() and path.name.startswith(name_prefix) and not path.name.endswith(".path")
            }
        except OSError:
            return None
        if actual_aliases != set(member_aliases):
            return None

        refs: list[ArtifactRef] = []
        for member_alias in member_aliases:
            id_path = self._aliases / member_alias
            try:
                artifact_id_for_member = id_path.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            path_file = self._aliases / f"{member_alias}.path"
            try:
                uri = Path(path_file.read_text(encoding="utf-8").strip())
            except OSError:
                return None
            if not uri.exists() or not self.manifest_path(artifact_id_for_member).exists():
                return None
            refs.append(
                ArtifactRef(
                    uri=uri, blake3=artifact_id_for_member, media_type=infer_media_type(uri), manifest=self.manifest_path(artifact_id_for_member)
                )
            )
        return refs

    def invalidate_cached_refs(self, alias: str) -> None:
        """Invalidate a fan-out completion record before rebuilding its members."""
        self._cached_refs_marker(alias).unlink(missing_ok=True)

    def write_cached_refs(self, alias: str, artifact_id: str, member_aliases: Iterable[str]) -> None:
        """Atomically publish the completed member set for a fan-out artifact.

        Stale member aliases from a prior fixed-name source release are removed before the
        completion record is published. The marker is deliberately written last.
        """
        ordered_aliases = list(dict.fromkeys(member_aliases))
        expected = set(ordered_aliases)
        prefix = f"{alias}::"
        if any(not member_alias.startswith(prefix) for member_alias in expected):
            raise ValueError(f"member alias is outside source alias: {alias}")
        parent = self._aliases / alias.rsplit("/", 1)[0]
        name_prefix = alias.rsplit("/", 1)[-1] + "::"
        parent.mkdir(parents=True, exist_ok=True)
        for path in list(parent.iterdir()):
            if not path.is_file() or not path.name.startswith(name_prefix) or path.name.endswith(".path"):
                continue
            member_alias = path.relative_to(self._aliases).as_posix()
            if member_alias not in expected:
                path.unlink(missing_ok=True)
                (self._aliases / f"{member_alias}.path").unlink(missing_ok=True)
        payload = json.dumps({"artifact_id": artifact_id, "aliases": ordered_aliases}, indent=2)
        self._atomic_write(self._cached_refs_marker(alias), payload)

    def _cached_refs_marker(self, alias: str) -> Path:
        return self._aliases / f"{alias}.members.json"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    # -- internals -------------------------------------------------------------
    def _write_alias(self, alias: str, artifact_id: str, dest: Path) -> None:
        alias_path = self._aliases / alias
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        # Alias stores the artifact id (resolvable to a path via the store).
        self._atomic_write(alias_path, artifact_id)
        # Also keep a sibling symlink-free pointer for convenience.
        self._atomic_write(self._aliases / f"{alias}.path", str(dest))


__all__ = ["OP_INDEX_FILENAME", "OP_INDEX_KEY_SPEC", "ArtifactStore", "op_index_key"]
