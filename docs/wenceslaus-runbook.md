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
| `uv sync` / tests / lint (`make check-all`) | ✅ | ✅ |
| `mock` / `sample` pipeline runs | ✅ | ✅ |
| NER (offline gazetteer; production GLiNER on the RTX 5070 Ti) | ✅ | ✅ (P100 optional) |
| bounded `prod` smoke run (`--quarter-limit 1 --release-limit 1`) | ✅ (with network) | ✅ |
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
uv sync --all-extras          # base + ner + airflow + kg + kg-qc
# or, minimally for the KG build:
uv sync --extra kg --extra ner
make build-go                 # build the Go dakp-worker (byte-parity extractors) for prod speed
```

`make build-go` produces `go/dakp-worker`. Point the runner at it with
`export DAKP_WORKER_BIN=$PWD/go/dakp-worker` (or just keep `go` on `PATH` and the runner builds +
caches one under `<repo>/tmp/dakp-go/`, keyed by a hash of the Go sources).

## Step 1 — build the fullmap database (wenceslaus-only)

The fullmap redb is Tablassert's canonical-resolution database (BABEL-backed). It needs ~120 GiB
RAM and ~2 h. Build it once into `/local_raid1`; reuse it across DAKP rebuilds.

```bash
uv run --extra kg tablassert build-fullmap \
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

```bash
uv run dakp run --profile prod --workdir /local_raid1/dakp/work
```

**To engage the byte-parity Go extractors** (recommended on wenceslaus for the hot DailyMed/FAERS
parsers): the CLI has no `--use-go-workers` flag, so enable it through the `use_go_workers` param.
With `DAKP_WORKER_BIN` exported (Step prerequisites) and the binary present, run:

```bash
uv run python -c "from dakp_pipeline.pipeline import run_pipeline; \
  run_pipeline(profile='prod', workdir='/local_raid1/dakp/work', params={'use_go_workers': True})"
```

The extractors delegate to the Go `dakp-worker` when `use_go_workers` is on **and** a binary is
available, and fall back to the pure-Python extractors automatically otherwise — output is
byte-for-byte identical either way (golden-file parity in `go test ./...`).

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
uv run --extra kg tablassert build-kg tables/graph.yaml \
  --fullmap /local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap.redb
```

Add `--qc` for the embedding-based audit (needs the `[kg-qc]` extra: `uv sync --extra kg-qc`) and
`--release` for a release build. Tablassert resolves the mention text / UNII subjects to ontology
CURIEs (fullmap), assigns categories, mints deterministic edge ids, dedups, and writes KGX NDJSON
(`drug_approvals_kg_nodes.jsonl.gz` / `drug_approvals_kg_edges.jsonl.gz`). DAKP ships no local KGX
compiler — this step is the delegation boundary in action.

## Step 4 — verify

```bash
make check-all        # Python (lint + format + pyright + tests @ 100% coverage) + Go (build + vet + parity + gofmt)
```

Then sanity-check the build summary and KGX:

```bash
jq . /local_raid1/dakp/work/data/reports/build_summary.json
uv run pytest tests/integration/test_semantic_equivalence.py -q   # preserved-semantics guardrail
```

## Laptop-safe day-to-day loop

Everything except Steps 1 and 3 (and the unbounded Step 2) runs on the laptop:

```bash
make check-all                                                       # full quality gate
uv run dakp run --profile mock --fixture-root tests/fixtures/pipeline --workdir tmp/mock-run
uv run dakp run --profile prod --quarter-limit 1 --release-limit 1 --workdir /tmp/dakp-prod-smoke  # real, bounded
uv sync --extra ner && uv run pytest tests/eval -q                   # NER benchmark (GLiNER on the RTX 5070 Ti)
```

## Troubleshooting

- **`build-fullmap` OOM** — it needs ~120 GiB; run only on wenceslaus, close other consumers, and
  keep `--cache` on `/local_raid1`.
- **`RuntimeError: tablassert is not available`** in Step 2/3 — `uv sync --extra kg` (or set
  `DAKP_TABLASERT_DIR` for a local editable checkout). See [`runbook.md`](./runbook.md).
- **Go workers not engaging** — confirm `echo $DAKP_WORKER_BIN` points at a real binary (or `go` is
  on `PATH`) and that `use_go_workers` is set; otherwise the Python extractors run (same bytes).
- **Download failures** — check connectivity to the FDA/DailyMed endpoints; bound scope with
  `--quarter-limit` / `--release-limit` to isolate a bad source. See [`runbook.md`](./runbook.md).

## Related

- [`architecture.md`](./architecture.md) — layered pipeline, sharding/concurrency, BLAKE3 store.
- [`tablassert-handoff.md`](./tablassert-handoff.md) — the generated configs + `tablassert` CLI.
- [`sources.md`](./sources.md) — per-source acquisition/extraction endpoints.
- [`runbook.md`](./runbook.md) — common failures, reruns, cache invalidation.
- [`semantic-equivalence.md`](./semantic-equivalence.md) — what the produced KG preserves vs improves.
