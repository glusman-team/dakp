# Guard Tablassert configs against wacky CURIE categories

## Context

DAKP generates one Tablassert Graph config + one table config per assertion table
(`src/dakp_pipeline/tablassert.py`), and `tablassert build-kg` resolves the mention text
(`subject_text` / `object_text`) to CURIEs + categories via the fullmap. Today the configs only
**softly** steer resolution with `prioritize` (a ranking boost) — there is no hard filter, so any
category present in the fullmap can win and land in the KGX graph.

The real fullmap on this box (`~/Desktop/fullmap`) contains **46 biolink categories**, most with no
business in a drug↔disease graph: `OrganismTaxon`, `Publication`, `Gene`, `Protein`, `CellLine`,
`GeographicLocation`, `Human`, `Cohort`, `Device`, `Event`, `Phenomenon`, `NamedThing`, …

**Mechanism (verified in `../Tablassert/src/tablassert/models.py` + `fullmap.py`):**
`NodeEncoding.avoid: list[Categories]` drops candidates with those categories **entirely** during
entity resolution. There is no allow-list field and no graph-level category filter, so an allow-list
must be expressed as `avoid` = complement of the allowed set. `avoid` entries are validated against
Tablassert's `Categories` enum (built dynamically from Biolink 4.4.3, 159 members) at config-load —
unknown names fail the build. When `avoid` filters out *all* candidates for a mention, the term stays
unresolved and the row produces no edge (logged, not an error): a little recall is traded for
guaranteed category sanity. Wacky categories leak today exactly when they are a mention's *only*
candidate (`prioritize` already out-ranks them when a sane candidate exists).

**Decisions (user-approved / defaulted at review):**
1. **Ironclad allow-list**: compute `avoid` as the complement of a small allowed set against the
   installed Tablassert's `Categories` enum at generation time — future-proof against fullmap
   rebuilds and always consistent with the binary that validates the config.
2. **Allowed sets mirror the existing `prioritize` tuples** (single source of truth):
   subject `Drug/SmallMolecule/ChemicalEntity`, object `Disease/PhenotypicFeature`.
3. **Regenerate + recommit the `tables/*.yaml` snapshots and push.** The committed snapshots have
   drifted anyway (they still carry `infores:` inside `provenance.override`, which Tablassert 8.0.1
   forbids there); regeneration fixes that for free.
4. The in-flight uncommitted work in the tree (the `plans/tablassert-output-legacy-parity.md`
   annotation renames: `has_evidence` / `supporting_documents`) is committed FIRST as its own
   commit, then the category guard lands as a second commit.

## Approach

In `src/dakp_pipeline/tablassert.py`:

* New `category_avoid_list(allowed: Sequence[str]) -> list[str]`: lazy-import
  `tablassert.biolink.Categories` (core dep; CI `uv sync --group dev` installs it), return the
  **sorted** complement of `allowed` over the enum's values; raise `ValueError` if an `allowed`
  entry is not a real category (fail loudly at config time). Sorting keeps the emitted YAML
  deterministic regardless of enum enumeration order.
* `table_config()` adds `"avoid": category_avoid_list(SUBJECT_PRIORITIZE)` to the subject encoding
  and `"avoid": category_avoid_list(OBJECT_PRIORITIZE)` to the object encoding (all three tables).
  The allow-list IS the `prioritize` tuple — documented as such in the module docstring + comments.
* Update the module docstring bullet list (the config shape description) to mention the hard
  category allow-list.
* The existing stdlib YAML emitter handles the new lists unchanged (CamelCase names are safe plain
  scalars).

Config size impact: ~155-entry `avoid` list per node encoding (6 total) in generated + committed
YAML. Acceptable: generated once, machine-written.

### Tests

* `tests/unit/test_tablassert_configs.py` (always runs; tablassert is now required for
  `table_config()` — update the module docstring line claiming otherwise):
  * `test_table_config_structure`: assert `avoid` present on subject + object; disjoint from the
    side's `prioritize`; contains known wackies (`OrganismTaxon`, `Publication`, `NamedThing`,
    `Gene`); object `avoid` also contains the drug-side categories and vice versa (cross-side
    exclusion); `prioritize ∪ avoid == set(tablassert.biolink.Categories)` (exact partition).
  * New test: `category_avoid_list(["NotARealCategory"])` raises `ValueError` (covers the branch;
    100% branch-coverage gate).
  * New test: each emitted `table_yaml` parses through `tablassert.models.Section` (proves every
    `avoid` entry validates against the installed enum) — the config structure tests already imply
    tablassert, so no importorskip needed.
* `tests/integration/test_kgx_end_to_end.py`: extend the node-shape assertion — every KGX node's
  `category` entries must be within the five allowed CURIEs
  (`biolink:Drug/SmallMolecule/ChemicalEntity/Disease/PhenotypicFeature`). The tiny fullmap fixture
  already uses only `SmallMolecule`/`Disease`/`PhenotypicFeature`, so the build stays green.
  (A causal "decoy wacky candidate" fixture test is deliberately skipped: a decoy only changes
  outcomes when it is a mention's sole candidate, which would require new mention text in the
  pipeline fixture TSVs + NER behavior — disproportionate ripple. The hard-filter semantics live in
  Tablassert's own tested `filter_and_rank`.)

### Regenerate committed snapshots

One-off (no new script needed):

```bash
uv run python - <<'PY'
from pathlib import Path
from dakp_pipeline.tablassert import _TABLE_ORDER, _TABLE_SPECS, graph_yaml, table_yaml
for table in _TABLE_ORDER:
    Path(f"tables/{_TABLE_SPECS[table][0]}.yaml").write_text(table_yaml(table), encoding="utf-8")
Path("tables/graph.yaml").write_text(graph_yaml(), encoding="utf-8")
PY
```

(`graph.yaml` should come out byte-identical — version `0.1.0` matches `__version__`; verify.)

## Files to modify

* `src/dakp_pipeline/tablassert.py` — `category_avoid_list()` + `avoid` in `table_config()` + docs.
* `tables/{approved_treats,faers_applied_to_treat,contraindications}.yaml` — regenerated
  (also drops the stale `infores:` from `provenance.override`).
* `tests/unit/test_tablassert_configs.py` — assertions above + docstring fix.
* `tests/integration/test_kgx_end_to_end.py` — allowed-category node assertion.

## Reuse

* `SUBJECT_PRIORITIZE` / `OBJECT_PRIORITIZE` (`tablassert.py`) double as the allow-lists.
* `_dump_yaml` emitter (`tablassert.py`) — no changes needed for the new lists.
* `tablassert.biolink.Categories` (installed package) — the category universe; matches the enum
  that validates the emitted config.
* `tiny_fullmap.TERMS` categories already satisfy the new allow-lists.

## Steps

- [x] 1. Commit the in-flight legacy-parity work already in the working tree
      (`src/dakp_pipeline/tablassert.py`, `tests/unit/test_tablassert_configs.py`,
      `tables/approved_treats.yaml`, `tables/contraindications.yaml`,
      `plans/tablassert-output-legacy-parity.md`) — run the unit tests first to confirm green.
      (commit `be94d69`)
- [x] 2. Implement `category_avoid_list()` + wire `avoid` into `table_config()`; update module
      docstring/comments.
- [x] 3. Update/extend `tests/unit/test_tablassert_configs.py` (partition asserts, wacky-membership
      asserts, `ValueError` branch test, `Section` model-validation test); fix its "never requires
      tablassert" docstring claim.
- [x] 4. Extend `tests/integration/test_kgx_end_to_end.py` node-category assertion to the five
      allowed biolink CURIEs.
- [x] 5. Regenerate the four committed `tables/*.yaml` snapshots (snippet above); eyeball the diff
      (expect: new sorted `avoid` blocks, removal of the stale `infores:` override lines,
      `graph.yaml` unchanged). (`graph.yaml` only re-flowed its folded description line breaks.)
- [x] 6. Lint/type/tests: `uv run ruff check . && uv run ruff format --check .`,
      `uv run pyright src/dakp_pipeline/tablassert.py tests/unit/test_tablassert_configs.py`,
      `uv run pytest tests/unit tests/integration -q` (coverage gate included).
      (canonical `uv run pytest`: 692 passed, coverage 100.00%. Note: passing explicit path args
      alongside addopts' `--cov` triggers a pre-existing collection quirk — use bare `uv run pytest`
      like CI does.)
- [x] 7. Commit the category-guard change (src + tests + regenerated snapshots + this plan file) on
      `rebuild/airflow-pipeline`, then `git push origin rebuild/airflow-pipeline`.

## Verification

* `uv run pytest tests/unit/test_tablassert_configs.py -q` — partition/wacky/ValueError/model tests.
* `uv run pytest tests/integration/test_kgx_end_to_end.py -q` — real `tablassert build-kg` over the
  tiny fullmap; nodes only carry allowed categories.
* Spot-check one regenerated config: `avoid` sorted, no overlap with `prioritize`, ~155 entries.
* Optional probe against the real fullmap: resolve a wacky-only mention via
  `tablassert.fullmap.filter_and_rank` with/without the generated `avoid` list and confirm the
  wacky candidate is dropped.

## Not doing (decided)

* No `exclude_prefixes` / `exclude_regex` (CURIE-level filters) — the ask is category-level.
* No widening of the allowed sets beyond the `prioritize` lists (user can veto at review; recall
  impact is the tradeoff).
* No full pipeline run / KGX artifact push — "push outputs" = commit + push the regenerated configs.
