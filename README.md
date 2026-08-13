# DAKP

[![CI](https://github.com/glusman-team/dakp/actions/workflows/ci.yml/badge.svg)](https://github.com/glusman-team/dakp/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/glusman-team/dakp/blob/main/LICENSE)

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

`dakp up` builds the native Go bundle, starts a local Airflow, triggers the `dakp_build` DAG,
waits, and prints the build summary. `uv run dakp down` stops the local Airflow. Without
`--fullmap`, the Tablassert handoff is deferred (a manifest is written) — never an error.

## The Pipeline

```text
acquire ─▶ extract ─▶ NER ─▶ aggregate ─▶ Tablassert KGX handoff
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

## Output Tables

| Assertion table | Predicate | Subject → Object |
| --------------- | --------- | ---------------- |
| approved-treats | `biolink:treats` | drug → disease/phenotype |
| observed-use | `biolink:applied_to_treat` | drug → disease/phenotype |
| contraindication | `biolink:contraindicated_in` | drug → disease/phenotype |

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
