# DAKP

[![CI](https://github.com/glusman-team/dakp/actions/workflows/ci.yml/badge.svg)](https://github.com/glusman-team/dakp/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/glusman-team/dakp/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/glusman-team/dakp.svg)](https://github.com/glusman-team/dakp/stargazers)

> **Drug Approvals Knowledge Provider** — a single, reproducible pipeline that turns DailyMed,
> Drugs@FDA, and FAERS into Translator assertion tables, ready for
> [Tablassert](https://pypi.org/project/tablassert/) KGX modeling.

DAKP downloads real FDA data sources, extracts treatment and contraindication assertions, mines
disease mentions with NER, and aggregates everything into three TSV assertion tables. It then
generates Tablassert configs and hands off canonical resolution and KGX compilation to the
installed `tablassert` CLI — DAKP ships no local KGX compiler.

## Quick Start

```bash
uv sync                 # installs everything, including the dakp CLI
uv run dakp up --small  # bounded real-data dev run (~1 FAERS quarter + 1 DailyMed release)
```

For a full production run with the KGX handoff:

```bash
uv run dakp up --fullmap /path/to/fullmap.redb
```

`dakp up` builds the native Go bundle, starts a local Airflow, triggers the `dakp_pipeline` DAG,
waits, and prints the build summary. `uv run dakp down` stops the local Airflow. Without
`--fullmap`, the Tablassert handoff is deferred (a manifest is written) — never an error.

To export the MEDliNER training-data bundle without running Airflow:

```bash
uv run dakp export-medliner --out /path/to/bundle                 # from a materialized workdir (after `dakp up`)
uv run dakp export-medliner --fixtures --out tmp/medliner-bundle  # offline: reference extractors over the committed fixtures
```

The default mode reads the already-extracted interim tables and never downloads — missing
tables are a loud error naming them. `--fixtures` first runs the pure-Python reference
extractors over the committed pipeline fixtures (fully offline); it is also the documented way
to regenerate MEDliNER's committed sample bundle.

## The Pipeline

```mermaid
flowchart LR
    acquire --> extract --> NER --> aggregate --> tablassert["Tablassert KGX handoff"] --> legacy["legacy TSV export"]
```

- **acquire** — real downloaders for DailyMed full releases, Drugs@FDA, and FAERS quarterly
  extracts; content-addressed (BLAKE3), idempotent, and manifest-recorded. DailyMed and Drugs@FDA
  downloads are freshness-gated (7-day cache window by default), so re-runs skip tens of GB of
  re-downloads. Drugs@FDA and FAERS download via the bundled **aria2c** binary, falling back to
  stdlib HTTP (`DAKP_ARIA2=0` forces the fallback).
- **extract** — heavy parsers run as **native Go bundle workers** ([`go/`](./go)); the DAG's
  `extract_*` tasks are `@task.stub(queue="golang")` declarations the coordinator forks per task
  instance.
- **NER** — a composite DiseaseNER backend (curated gazetteer + GLiNER zero-shot) mines
  disease/phenotype mentions from DailyMed contraindication sections. Emits mentions only —
  never ontology CURIEs.
- **aggregate** — joins extracted tables and NER mentions into three uncompressed TSV assertion
  tables.
- **Tablassert handoff** — generates a graph config plus one table config per assertion table,
  then delegates to `tablassert build-kg`.
- **legacy TSV export** — retrofits the KGX pair into the pre-rewrite DAKP TSV schema
  (`<workdir>/data/DRUG_APPROVALS_KP_<version>.{nodes,edges}.tsv`: 3-column nodes, 12-column edges, `NA`
  fills, comma-joined multi-values) for the internal service that still consumes it. The task
  skips cleanly when the handoff was deferred (no `--fullmap` → no KGX to convert).
- **MEDliNER export** — `export_medliner_training_data` (the `medliner` TaskGroup) hands the
  annotation corpus to MEDliNER as a self-describing, deterministic `dakp.medliner.export.v1`
  bundle (`manifest.json` + `candidates.ndjson` + the NER gold benchmark) under
  `<workdir>/data/store/medliner-export`. It consumes only the DailyMed and FAERS extracts, so
  it runs alongside the shape stage and never gates the build summary.

## Output Tables

| Assertion table | Predicate | Subject → Object |
| --------------- | --------- | ---------------- |
| approved-treats | `biolink:treats` | drug → disease/phenotype |
| observed-use | `biolink:applied_to_treat` | drug → disease/phenotype |
| contraindication | `biolink:contraindicated_in` | drug → disease/phenotype |

## Resource Ingest Guide (RIG)

A Resource Ingest Guide is the Translator-standard document telling ingest maintainers how a
knowledge source is produced, what it contains, and who to credit. The graph config carries
one adapted from the Translator ingest working group's DINGO-reviewed DAKP RIG in
[NCATSTranslator/translator-ingests](https://github.com/NCATSTranslator/translator-ingests)
(review issue #416), grounded in this repository's actual download URLs, sources, and pipeline
behavior. `tables/graph.yaml` is generated from the pipeline's `_rig_config()` — regenerate
it, never hand-edit: the pipeline's `generate` task (`src/dakp_pipeline/tablassert.py`)
rewrites it into the run workdir, and the committed copy at the repo root must byte-equal
that output (the test suite enforces the match).

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — `uv sync` installs every Python dependency (Airflow 3,
  GLiNER, `tablassert[qc]`) plus the `dakp` CLI. There are no optional extras.
- A **Go toolchain** — to build the native bundle (`dakp up` does this automatically).

Acquisition is always real; "offline" is only a test concern (the suite monkeypatches the HTTP
layer and runs on committed fixtures).

## Developing

```bash
uv run pytest -q --cov        # tests; 100% branch coverage gate (fail_under = 100)
uv run ruff check             # lint
uv run ruff format --check    # formatting
uv run pyright                # type check
```

## License

Apache License 2.0. (The bundled aria2c binary is GPLv2 but runs as a separate subprocess, so it
does not affect DAKP's license.)
