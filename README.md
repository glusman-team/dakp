# DAKP — Drug Approvals Knowledge Provider

DAKP is a single, reproducible pipeline that builds three Translator assertion tables —
**approved-treats**, **observed-use**, and **contraindication** — from DailyMed, Drugs@FDA,
and FAERS, then hands them to [Tablassert](https://pypi.org/project/tablassert/) for KGX
modeling. Acquisition is always **real**; "offline" is only ever a test concern (the test
suite monkeypatches the HTTP layer and runs on committed fixtures).

## The pipeline

One pipeline, five stages:

```text
acquire ─▶ extract ─▶ NER ─▶ aggregate ─▶ Tablassert KGX handoff
```

- **acquire** — real downloaders for DailyMed full releases, Drugs@FDA, and FAERS quarterly
  extracts; content-addressed (BLAKE3), idempotent, manifest-recorded. Drugs@FDA and FAERS
  download via the bundled **aria2c** binary (multi-connection; the PyPI `aria2` wheel — no
  separate install), falling back to stdlib HTTP when aria2c is unavailable or `DAKP_ARIA2=0`;
  DailyMed keeps a stdlib conditional-GET path for its 304/ETag freshness semantics. DailyMed
  releases are freshness-gated: a stored release fetched within `dailymed_max_age_days`
  (default **7** days) is reused with zero ZIP downloads — DailyMed replaces its fixed-name
  full-release ZIPs in place, so without the gate every new release re-downloads the whole
  snapshot (~tens of GB). Drugs@FDA uses the same seven-day cache window via `drugsfda_max_age_days`. `force` bypasses both gates; `<= 0` disables them.
- **extract** — the heavy parsers run as **native Go bundle workers** (an Airflow Go SDK bundle
  under [`go/`](./go)); the DAG's `extract_*` tasks are `@task.stub(queue="golang")`
  declarations the coordinator forks per task instance.
- **NER** — one composite DiseaseNER backend (curated gazetteer + GLiNER zero-shot) that mines
  disease/phenotype mentions from the DailyMed contraindication sections. Emits mentions only —
  never ontology CURIEs.
- **aggregate** — joins the extracted tables and NER mentions into three uncompressed TSV
  assertion tables.
- **Tablassert KGX handoff** — generates a Graph config plus one table config per assertion
  table, then delegates canonical resolution and KGX compilation to the installed `tablassert`
  CLI. DAKP ships no local KGX compiler.

The three assertion TSVs:

| Assertion table | Predicate | Subject → Object |
| --- | --- | --- |
| approved-treats | `biolink:treats` | drug → disease/phenotype |
| observed-use | `biolink:applied_to_treat` | drug → disease/phenotype |
| contraindication | `biolink:contraindicated_in` | drug → disease/phenotype |

## Running

There is one way to run the pipeline:

```bash
uv run dakp up [--fullmap PATH] [--small]
```

`dakp up` builds and packs the native Go bundle, starts a local Airflow, triggers the
`dakp_build` DAG, waits for it to finish, and prints the build summary.

- `--fullmap PATH` — trigger the **real** Tablassert handoff against a prebuilt fullmap redb.
  Without it the handoff is **deferred** (a manifest is written) — never an error.
- `--small` — a bounded real-data dev run (~1 FAERS quarter + 1 DailyMed release): the same
  pipeline, less data.

`uv run dakp down` stops the local Airflow started by `up`.

## Prerequisites

- `uv sync` — installs every Python dependency (Airflow 3, the NER backend, `tablassert[qc]`)
  plus the `dakp` CLI. This includes the `aria2` wheel, which bundles a statically-linked
  **aria2c** binary (GPLv2); it runs as a separate subprocess (not linked), so it does not
  affect DAKP's Apache-2.0 license. Set `DAKP_ARIA2=0` to force the stdlib HTTP fallback.
- A Go toolchain — to build and pack the native bundle. `dakp up` does this automatically.

## Tests

```bash
uv run pytest -q --cov        # tests; 100% branch coverage gate (fail_under = 100)
uv run ruff check             # lint
uv run ruff format --check    # formatting
uv run pyright                # type check
```
