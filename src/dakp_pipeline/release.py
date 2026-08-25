"""Legacy-named release copies of the final DAKP build artifacts.

Downstream consumers know the pre-rewrite naming schema
(``drug_approvals_kg_{nodes,edges}_v0.5.3.tsv.gz``); the Tablassert handoff instead emits
``data/DRUG_APPROVALS_KP_<version>.{nodes,edges}.ndjson`` and the legacy-TSV stage adds the
``.tsv`` pair under that same stem. This stage copies the final artifacts — the KGX ndjson pair,
the legacy TSV pair, and the Tablassert-generated RIG — to the legacy schema under
``<workdir>/data``: ``drug_approvals_kg_nodes_v<version>.ndjson`` /
``drug_approvals_kg_edges_v<version>.ndjson``, the same stems as ``.tsv``, and
``drug_approvals_kg_v<version>.RIG.yaml`` (no nodes/edges kind). The published yaml is the
Resource Ingest Guide Tablassert emits next to the KGX pair (``<name>_<version>.RIG.yaml``) —
NOT the ``tables/graph.yaml`` build config, which is an input, not a release artifact. Copies,
not renames: the Tablassert-stemmed originals stay for the build summary and debugging.

The stage entry point :func:`publish` is Airflow-free. A DEFERRED handoff (no fullmap ->
no ``build-kg`` -> no ndjson to name) returns an EMPTY ref list — never an error — mirroring
the legacy-TSV convention; the DAG task turns that into an ``AirflowSkipException``.
"""

from __future__ import annotations

import json
import shutil

from dakp_pipeline import __version__
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.io.schemas import TSV_MEDIA_TYPE
from dakp_pipeline.legacy_tsv import _single_glob
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import GRAPH_NAME, REPORT_NAME

#: Legacy artifact stem (the ``drug_approvals_kg_nodes_v0.5.3`` schema).
LEGACY_STEM = "drug_approvals_kg"

_NDJSON_MEDIA_TYPE = "application/x-ndjson"
_YAML_MEDIA_TYPE = "application/yaml"

_PUBLISH_OPERATION = "publish_release_artifacts"


def _legacy_name(kind: str | None, suffix: str) -> str:
    """``drug_approvals_kg_<kind>_v<version>.<suffix>``; the RIG yaml carries no kind."""
    stem = f"{LEGACY_STEM}_{kind}" if kind else LEGACY_STEM
    return f"{stem}_v{__version__}.{suffix}"


def publish(kgx_refs: list[ArtifactRef], legacy_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Copy the final build artifacts to the legacy ``drug_approvals_kg_*_v<version>`` names.

    Returns ``[]`` on a deferred handoff (the report among ``kgx_refs`` records that no
    ``build-kg`` ran). Expects the current graph/version KGX ndjson pair and the matching
    ``<name>_<version>.RIG.yaml`` under ``<workdir>/data``, plus exactly two legacy TSV refs
    (``.nodes.tsv`` / ``.edges.tsv``) from :func:`dakp_pipeline.legacy_tsv.export`; anything
    missing is a loud ``RuntimeError``. Returns the published refs in legacy-name order:
    nodes/edges ndjson, nodes/edges tsv, RIG yaml.
    """
    event = _PUBLISH_OPERATION
    report_refs = [ref for ref in kgx_refs if ref.uri.name == REPORT_NAME]
    if len(report_refs) != 1:
        msg = f"expected exactly one {REPORT_NAME} among the tablassert handoff refs, found {len(report_refs)}"
        raise RuntimeError(msg)
    report = json.loads(report_refs[0].uri.read_text(encoding="utf-8"))
    if report.get("mode") == "deferred":
        stats(logger, event, mode="deferred", reason="no Tablassert handoff ran; nothing to publish")
        return []

    tsv_by_kind = {ref.uri.name.removesuffix(".tsv").rsplit(".", 1)[-1]: ref.uri for ref in legacy_refs}
    if sorted(tsv_by_kind) != ["edges", "nodes"]:
        msg = f"expected the legacy .nodes.tsv/.edges.tsv pair among the legacy refs, found {sorted(ref.uri.name for ref in legacy_refs)}"
        raise RuntimeError(msg)

    workdir = Workdir(ctx.workdir)
    data_dir = workdir.root / "data"
    kgx_stem = f"{GRAPH_NAME}_{__version__}"
    with step(logger, event):
        store = ArtifactStore(workdir)
        operation = OperationBlock(name=event)
        sources = [
            (_single_glob(data_dir, f"{kgx_stem}.nodes.ndjson"), "nodes", "ndjson", _NDJSON_MEDIA_TYPE),
            (_single_glob(data_dir, f"{kgx_stem}.edges.ndjson"), "edges", "ndjson", _NDJSON_MEDIA_TYPE),
            (tsv_by_kind["nodes"], "nodes", "tsv", TSV_MEDIA_TYPE),
            (tsv_by_kind["edges"], "edges", "tsv", TSV_MEDIA_TYPE),
            (_single_glob(data_dir, f"{kgx_stem}.RIG.yaml"), None, "RIG.yaml", _YAML_MEDIA_TYPE),
        ]
        refs: list[ArtifactRef] = []
        for source, kind, suffix, media_type in sources:
            target = data_dir / _legacy_name(kind, suffix)
            shutil.copyfile(source, target)
            refs.append(store.register(target, media_type=media_type, inputs=[hash_file(source)], operation=operation))
            stats(logger, event, published=str(target), source=str(source))
    return refs


__all__ = ["LEGACY_STEM", "publish"]
