# `wenceslaus` full-build runbook

The **full production build** of the DAKP knowledge graph. This is the only place the multi-TB
sources, the ~120 GiB fullmap, and the live Tablassert KGX compilation run. Read
[`architecture.md`](./architecture.md) and [`tablassert-handoff.md`](./tablassert-handoff.md) first;
this document is the exact command sequence.

## What runs where

The fullmap build and the full prod KG need workstation-class RAM and **cannot run on a laptop**.
Mock/sample runs and NER are laptop-safe.

| Step | Laptop (30 GiB, RTX 5070 Ti) | `wenceslaus` (187 GiB RAM) |
| --- | --- | --- |
| `uv sync` / tests / lint (`uv run pre-commit run --all-files`) | ✅ | ✅ |
| `mock` / `sample` pipeline runs | ✅ | ✅ |
| NER (offline gazetteer; production GLiNER on the RTX 5070 Ti) | ✅ | ✅ (P100 optional) |
| bounded `prod` smoke run (`tests/integration/test_prod_smoke.py`, offline) | ✅ | ✅ |
| **fullmap build** (`tablassert build-fullmap`, ~120 GiB RAM, ~2 h) | ❌ | ✅ **wenceslaus-only** |
| **full `prod` build** (all DailyMed releases + FAERS quarters) | ❌ | ✅ **wenceslaus-only** |
| **full KGX compilation** (`tablassert build-kg` over the full graph) | ❌ | ✅ **wenceslaus-only** |

## Host

`wenceslaus` — Ubuntu 24.04, dual Xeon Gold 6230 (80 logical CPUs), 187 GiB RAM,
`/local_raid1` ~1.75 TiB, optional NVIDIA P100 GPUs. Put all large artifacts (downloads, interim
tables, the fullmap, KGX) on `/local_raid1`, never on the boot volume.

## Prerequisites

```bash
cd <dakp-checkout>
uv sync                       # ONE command: full runtime (Airflow 3, GLiNER NER, tablassert[qc]) + the `dakp` CLI
```

A Go toolchain ≥ 1.24 must be on `PATH`: `uv run dakp up` builds + packs the native Go extract bundle
(`go tool airflow-go-pack ./cmd/dakp-bundle`) automatically as its first step — no separate manual Go
build is needed.

## Step 1 — build the fullmap database (wenceslaus-only)

The fullmap redb is Tablassert's canonical-resolution database (BABEL-backed). It needs ~120 GiB
RAM and ~2 h. Build it once into `/local_raid1`; reuse it across DAKP rebuilds.

```bash
uv run tablassert build-fullmap \
  --output /local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap.redb \
  --threads 64
```

`build-fullmap` also takes `--cache <dir>` (download cache, default `fullmap/downloads`) and
`--version <tag>` (the BABEL snapshot; pinned default). Keep the cache on `/local_raid1`.

## Step 2 — run the full `prod` pipeline (acquire → extract → NER → aggregate)

Real DailyMed / Drugs@FDA / FAERS downloads, extraction, NER mining, and assertion aggregation.
Unbounded scope (all releases / quarters). This writes the three assertion TSVs + the generated
Tablassert configs into the workdir, and (because `prod` sets `run_tablassert=True`) attempts the
live Tablassert handoff.

The CLI workdir is hardcoded to `<repo>/tmp/airflow-run/data`, so on wenceslaus keep the checkout (or
a `tmp/` symlink) on `/local_raid1` to keep the multi-TB artifacts off the boot volume. Then run the
full-scope build, pointing `--fullmap` at the Step 1 database:

```bash
uv run dakp up prod --fullmap /local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap.redb
```

The heavy DailyMed/FAERS/Drugs@FDA extraction always runs as **native Go workers** (the Airflow Go
SDK bundle); `uv run dakp up prod` builds + packs the bundle, starts Airflow with the Go coordinator,
and triggers `dakp_build`. The Go extractors are parity-locked to the pure-Python reference extractors
(golden-file parity in `go test ./...`). The `prod` run is always full-scope (all DailyMed releases +
FAERS quarters); add `--detach` to trigger and walk away instead of waiting.

After this step the workdir holds:

```
/local_raid1/dakp/work/
  data/tabular/{approved_treats,faers_applied_to_treat,contraindication}_assertions.tsv
  tables/graph.yaml + tables/{approved_treats,faers_applied_to_treat,contraindications}.yaml
  data/reports/build_summary.json
```

## Step 3 — compile the KGX graph (wenceslaus-only)

Run Tablassert from the **workdir root** so the config's workdir-relative references
(`tables/*.yaml`, `data/tabular/*.tsv`) resolve, pointing `--fullmap` at the Step 1 database:

```bash
cd /local_raid1/dakp/work
uv run tablassert build-kg tables/graph.yaml \
  --fullmap /local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap.redb
```

Add `--qc` for the embedding-based audit (the QC runtime ships with the required `tablassert[qc]`
install) and
`--release` for a release build. Tablassert resolves the mention text / UNII subjects to ontology
CURIEs (fullmap), assigns categories, mints deterministic edge ids, dedups, and writes KGX NDJSON
(`drug_approvals_kg_nodes.jsonl.gz` / `drug_approvals_kg_edges.jsonl.gz`). DAKP ships no local KGX
compiler — this step is the delegation boundary in action.

## Step 4 — verify

```bash
uv run pre-commit run --all-files   # Python (lint + format + pyright + tests @ 100% coverage)
cd go && go build ./... && go vet ./... && go test ./... && test -z "$(gofmt -l .)"   # Go gate (build + vet + parity + gofmt)
```

Then sanity-check the build summary and KGX:

```bash
jq . /local_raid1/dakp/work/data/reports/build_summary.json
uv run pytest tests/integration/test_semantic_equivalence.py -q   # preserved-semantics guardrail
```

## Laptop-safe day-to-day loop

Everything except Steps 1 and 3 (and the unbounded Step 2) runs on the laptop:

```bash
uv run pre-commit run --all-files                                    # full Python quality gate (+ cd go && go test ./...)
uv run dakp up                                                       # mocked end-to-end pipeline via Airflow (native Go workers)
uv run pytest tests/integration/test_prod_smoke.py -q                # real path, bounded + offline (HTTP mocked)
uv run pytest tests/eval -q                                           # NER benchmark (GLiNER on the RTX 5070 Ti)
```

## Troubleshooting

- **`build-fullmap` OOM** — it needs ~120 GiB; run only on wenceslaus, close other consumers, and
  keep `--cache` on `/local_raid1`.
- **`RuntimeError: tablassert is not available`** in Step 2/3 — `uv sync` (or set
  `DAKP_TABLASERT_DIR` for a local editable checkout). See [`runbook.md`](./runbook.md).
- **Go workers not engaging** — the extract tasks are native Go SDK bundle workers; confirm the
  bundle was packed into the coordinator's `executables_root` (`uv run dakp up` does this) and that a Go
  toolchain ≥ 1.24 is available to build it. Check the `extract_*` task logs in Airflow.
- **Download failures** — check connectivity to the FDA/DailyMed endpoints; for a small offline
  exercise of the real path, run `uv run pytest tests/integration/test_prod_smoke.py -q`. See [`runbook.md`](./runbook.md).

## Related

- [`architecture.md`](./architecture.md) — layered pipeline, sharding/concurrency, BLAKE3 store.
- [`tablassert-handoff.md`](./tablassert-handoff.md) — the generated configs + `tablassert` CLI.
- [`sources.md`](./sources.md) — per-source acquisition/extraction endpoints.
- [`runbook.md`](./runbook.md) — common failures, reruns, cache invalidation.
- [`semantic-equivalence.md`](./semantic-equivalence.md) — what the produced KG preserves vs improves.
