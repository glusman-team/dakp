from __future__ import annotations

from pathlib import Path

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import configs as tablassert_configs


def _write_assertion_tsv(table: str, workdir: Workdir) -> Path:
    columns = schemas.columns_for(table)
    path = workdir.tabular / f"{table}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(columns) + "\n", encoding="utf-8")
    return path


def test_generate_configs_writes_graph_and_table_yamls(tmp_path: Path) -> None:
    from dakp_pipeline.io.contracts import TaskContext

    wd = Workdir(tmp_path / "work")
    wd.create()
    store = ArtifactStore(wd)

    assertion_refs = []
    for table in ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions"):
        path = _write_assertion_tsv(table, wd)
        ref = store.register(path, media_type=schemas.TSV_MEDIA_TYPE, rows=1)
        assertion_refs.append(ref)

    ctx = TaskContext(profile="mock", workdir=wd.root, fixture_root=None, threads=1, memory_budget_gb=1, params={})
    refs = tablassert_configs.generate(assertion_refs, ctx)

    # Graph config is first, then one config per table.
    graph = refs[0]
    assert graph.uri.name == "graph.yaml"
    graph_text = graph.uri.read_text(encoding="utf-8")
    assert "name: dakp" in graph_text
    assert "infores:multiomics-drugapprovals" in graph_text
    assert "tables/approved_treats_assertions.yaml" in graph_text

    # Each table config carries its predicate + the ManualProvenance override.
    by_name = {ref.uri.stem: ref.uri.read_text(encoding="utf-8") for ref in refs[1:]}
    assert "predicate: treats" in by_name["approved_treats_assertions"]
    assert "predicate: applied_to_treat" in by_name["faers_applied_to_treat_assertions"]
    assert "predicate: contraindicated_in" in by_name["contraindication_assertions"]
    for text in by_name.values():
        assert "kind: text" in text
        assert 'delimiter: "\\t"' in text
        assert "infores:multiomics-drugapprovals" in text
        assert "upstream_resource_ids:" in text
