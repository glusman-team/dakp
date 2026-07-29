"""Tablassert runner (stub / monkeypatch point).

In the ``mock`` profile (or whenever ``run_tablassert`` is disabled) this writes a
handoff manifest recording the assertion inputs and generated configs and returns — it
does **not** compile a graph or emit KGX (there is no local fallback KGX compiler).

In real/full runs it would delegate to ``../Tablassert``; that integration lands in
**Milestone 7**. Tests monkeypatch ``dakp_pipeline.tablassert.run`` (PLAN.md sketch).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir


def run(assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Hand off assertion tables + configs to Tablassert.

    Returns a list with one ArtifactRef to the handoff manifest (mock) or KGX outputs
    (full). Mock mode never touches the network or ``../Tablassert``.
    """
    run_real = bool(ctx.params.get("run_tablassert")) and ctx.profile != "mock"
    if run_real:
        msg = "real ../Tablassert integration lands in Milestone 7; no local KGX fallback"
        raise NotImplementedError(msg)

    store = ArtifactStore(Workdir(ctx.workdir))
    manifest_path = Workdir(ctx.workdir).reports / "tablassert_handoff.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    handoff = {
        "stage": "tablassert_handoff",
        "mode": "mock",
        "status": "deferred",
        "reason": "mock profile / run_tablassert disabled; canonical resolution + KGX compilation delegated to ../Tablassert in Milestone 7",
        "generated_at": datetime.now(UTC).isoformat(),
        "assertion_inputs": [{"table": ref.uri.stem, "artifact_id": ref.blake3, "rows": ref.rows} for ref in assertion_refs],
        "config_inputs": [str(ref.uri) for ref in config_refs],
    }
    manifest_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    ref = store.register(manifest_path, media_type="application/json", inputs=[ref.blake3 for ref in assertion_refs])
    return [ref]


__all__ = ["run"]
