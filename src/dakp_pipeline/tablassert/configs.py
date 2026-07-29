"""Generate Tablassert Graph + per-table config YAMLs.

Configs are emitted by string templating (no ``pyyaml`` dependency) and match the shapes
documented in ``PLAN.md`` ("Tablassert handoff config sketch"): ``source.kind: text``
with a tab delimiter, a column-encoded subject/object/predicate, a
``provenance.override`` (ManualProvenance) block, and per-annotation column encodings.
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.paths import Workdir

# Assertion table -> (predicate, upstream infores list, knowledge_level).
_TABLE_PROVENANCE: dict[str, tuple[str, list[str], str]] = {
    "approved_treats_assertions": ("treats", ["infores:dailymed", "infores:faers"], "knowledge_assertion"),
    "faers_applied_to_treat_assertions": ("applied_to_treat", ["infores:faers", "infores:dailymed"], "observation"),
    "contraindication_assertions": ("contraindicated_in", ["infores:medi", "infores:dailymed"], "knowledge_assertion"),
}

_CONFIGS_DIR_NAME = "tablassert"


def generate(assertion_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Write ``tables/graph.yaml`` plus one table config per assertion table.

    Returns refs to the generated config files (graph first, then tables in order).
    """
    store = ArtifactStore(Workdir(ctx.workdir))
    configs_root = Workdir(ctx.workdir).store / _CONFIGS_DIR_NAME / "tables"
    configs_root.mkdir(parents=True, exist_ok=True)

    written: list[ArtifactRef] = []
    table_config_paths: list[str] = []
    input_ids: list[str] = []

    for ref in assertion_refs:
        table = ref.uri.stem  # e.g. "approved_treats_assertions"
        if table not in _TABLE_PROVENANCE:
            continue
        predicate, upstream, kl = _TABLE_PROVENANCE[table]
        rel_tsv = Path("data/tabular") / f"{table}.tsv"
        config_path = configs_root / f"{table}.yaml"
        config_path.write_text(_table_config(table, predicate, upstream, kl, rel_tsv), encoding="utf-8")
        table_config_paths.append(f"tables/{table}.yaml")
        input_ids.append(ref.blake3)
        written.append(
            store.register(
                config_path, media_type="application/yaml", inputs=[ref.blake3], operation=OperationBlock(name="generate_tablassert_configs")
            )
        )

    graph_path = configs_root / "graph.yaml"
    graph_path.write_text(_graph_config(table_config_paths), encoding="utf-8")
    graph_ref = store.register(
        graph_path, media_type="application/yaml", inputs=input_ids, operation=OperationBlock(name="generate_tablassert_configs")
    )
    return [graph_ref, *written]


def _graph_config(table_config_paths: list[str]) -> str:
    tables_block = "\n".join(f"  - {p}" for p in table_config_paths)
    return (
        "name: dakp\n"
        'version: "0.1.0"\n'
        "description: >-\n"
        "  Drug Approvals Knowledge Provider: FDA-approved treatment relationships,\n"
        "  FAERS-observed applied-to-treat uses, and contraindications, modeled from\n"
        "  DailyMed, Drugs@FDA, FAERS, and MEDI.\n"
        "infores: infores:multiomics-drugapprovals\n"
        "fullmap: .fullmap\n"
        "tables:\n"
        f"{tables_block}\n"
    )


def _table_config(table: str, predicate: str, upstream: list[str], knowledge_level: str, rel_tsv: Path) -> str:
    upstream_block = "\n".join(f"      - {u}" for u in upstream)
    return (
        "source:\n"
        "  kind: text\n"
        f"  local: {rel_tsv.as_posix()}\n"
        f"  url: https://example.invalid/dakp/generated/{rel_tsv.name}\n"
        '  delimiter: "\\t"\n'
        "statement:\n"
        "  subject:\n"
        "    method: column\n"
        "    encoding: A\n"
        "    prioritize: [Drug, SmallMolecule, ChemicalEntity]\n"
        f"  predicate: {predicate}\n"
        "  object:\n"
        "    method: column\n"
        "    encoding: F\n"
        "    prioritize: [Disease, PhenotypicFeature]\n"
        "provenance:\n"
        "  override:\n"
        "    infores: infores:multiomics-drugapprovals\n"
        "    upstream_resource_ids:\n"
        f"{upstream_block}\n"
        f"    knowledge_level: {knowledge_level}\n"
        "    agent_type: manual_validation_of_automated_agent\n"
    )


__all__ = ["generate"]
