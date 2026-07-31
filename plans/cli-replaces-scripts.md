# Replace shell scripts + Makefile + direnv with a single `uv run dakp` CLI (cyclopts)

## Context

The run surface is three layered tools a user must learn and set up:

- **`scripts/dakp_up.sh` / `scripts/dakp_down.sh`** — ~200 lines of bash orchestrating Airflow
  (preflight self-heal, Go bundle pack, start/stop standalone, poll DAG/run state via sqlite, set
  Variable/pools, trigger). This is the user's #1 pain ("I hate the shell scripts the most").
- **`Makefile`** — thin `up-mock/up-sample/up-prod/down/clean/install` wrappers that just call the
  bash scripts with `PROFILE=` set.
- **`.envrc` (direnv)** — run config the user must `direnv allow` / `source`; the user says "I don't
  even need to set up all these env vars."

Goal: one self-contained Python CLI — `uv run dakp up` / `uv run dakp down` — with **all defaults
built in** (zero required env vars, zero required flags), built with **cyclopts** (user request).
Delete the shell scripts; retire the Makefile and the direnv requirement. The CLI reads **no
environment variables at all** — run locations (workdir, Airflow home, fixture root) and scope are
hardcoded constants, and the only inputs are the `profile` positional plus a few short-aliased flags
(`--fullmap/-f`, `--port/-p`, `--log-level/-l`, `--detach/-d`).

Two more user requirements:
- **`--fullmap <path>` is the ONE input the user actually enters** — the path to the prebuilt
  fullmap redb (built separately via `tablassert build-fullmap` on wenceslaus). Everything else defaults.
- **Remove anything that downloads a fullmap** — DAKP must never trigger a fullmap download (neither
  the acquire-layer `fullmap.redb` stub nor a `tablassert build-kg --fullmap .fullmap` auto-resolve).
  The fullmap is always an explicit existing path the user provides. This also means the whole
  `acquire_ontologies` task goes away (it existed only to fetch the fullmap; its output is never read).

## Approach

New module **`src/dakp_pipeline/cli.py`** (cyclopts `App`), exposed as a console script so
`uv run dakp …` works after `uv sync`:

```toml
[project.scripts]
dakp = "dakp_pipeline.cli:app"
```

### Commands

| Command | Replaces | Notes |
| --- | --- | --- |
| `uv run dakp up [profile]` | `make up-mock/up-sample/up-prod` + `dakp_up.sh` | `profile` optional positional, default `mock` (laptop-safe). So `uv run dakp up` == old `make up-mock`; `uv run dakp up prod` == old `make up-prod`. |
| `uv run dakp down` | `make down` + `dakp_down.sh` | stops the local Airflow |
| `uv run dakp clean` | `make clean` | removes caches / `tmp/` / Go binary |

`up` flags — deliberately tiny: only what you'd actually type; everything else is a built-in constant
(below). Each long flag has a one-letter alias:

| Flag | Alias | Default | Notes |
| --- | --- | --- | --- |
| `--fullmap` | `-f` | `None` | Path to the prebuilt fullmap redb — the ONE input you enter. Not needed for mock/deferred runs; **required for a real handoff** (clear error if missing). |
| `--port` | `-p` | `8090` | Airflow API/webserver port (8080 is often taken by the aoe daemon). |
| `--log-level` | `-l` | `INFO` | Pipeline log level (DEBUG / INFO / WARNING / ERROR). |
| `--detach` | `-d` | off | Trigger the DAG and return immediately instead of waiting for it to finish. |

`profile` is an optional positional (`uv run dakp up prod`), default `mock` (laptop-safe). So
`uv run dakp up` == old `make up-mock`; `uv run dakp up prod` == old `make up-prod`.

**Hardcoded in the workflow (not flags, not env vars).** The CLI reads **no environment variables**;
these are module-level constants in `cli.py`, derived from the repo root (`cli.py`'s `__file__`):

| Constant | Value |
| --- | --- |
| workdir | `<repo>/tmp/airflow-run/data` |
| fixture root | `<repo>/tests/fixtures/pipeline` |
| Airflow home | `<repo>/tmp/airflow-home` |
| quarter / release limit | always `None` → profile default (mock/sample = 1 quarter; prod = unbounded full build) |
| download / extract pools | `dakp_download` / `dakp_extract` (imported from the DAG constants) |

Consequences to be aware of (both accepted):
- A `prod` run via the CLI is always a **full-scope** build — the old `QUARTER_LIMIT=1 RELEASE_LIMIT=1`
  bounding is gone from the CLI. The bounded path is still exercised offline by
  `tests/integration/test_prod_smoke.py` (via the `run_pipeline` harness), just not via `dakp up`.
- Artifacts always land under `<repo>/tmp/`. On `wenceslaus`, keep the checkout (or a `tmp/` symlink)
  on `/local_raid1` so the multi-TB build doesn't fill the boot volume.

`None` limits serialize to JSON `null` in the `dakp_config` Variable — `runtime.build_context_from_config`
treats `null` as "profile default". `--fullmap` is added to the Variable as `"fullmap": <path|null>`
and threaded into `ctx.params["fullmap"]` (see the fullmap section below).

### Orchestration = a faithful Python port of `dakp_up.sh`

Same 6 steps, same behavior, same messages — but in Python with **module-level monkeypatchable
side-effect functions** (mirroring the existing `tablassert/run.py` pattern, which the tests already
monkeypatch):

- `run_subprocess(command, cwd=None, env=None) -> CompletedProcess` — never raises (monkeypatch point).
- `airflow_importable() -> bool` — `importlib.util.find_spec("airflow")`.
- `api_up(base_url) -> bool` — stdlib `urllib.request` GET `/api/v2/version` (no `requests`).
- `dag_registered(db) -> bool` / `run_state(db) -> str` — stdlib `sqlite3`.
- `sleep(seconds)` — thin wrapper so poll loops are instant in tests.

Ported logic (1:1 with the bash):
1. **Preflight self-heal** — if `import airflow` fails: `uv sync --reinstall-package
   apache-airflow-core`; still fails → `uv cache clean apache-airflow-core` + reinstall; still fails
   → clear error + non-zero exit. (Keeps the corrupt-venv/RAID self-heal the bash documents.)
2. **Airflow env** — build an env dict merging the `AIRFLOW__*` vars (`DAGS_FOLDER`,
   `LOAD_EXAMPLES=False`, `EXECUTOR=LocalExecutor`, `API__PORT/HOST`, `EXECUTION_API_SERVER_URL`,
   `SDK__COORDINATORS` with the Go bundle path, `SDK__QUEUE_TO_COORDINATOR`) over `os.environ`.
3. **Go bundle** — `go tool airflow-go-pack --output <home>/executable-bundles/dakp-bundle
   ./cmd/dakp-bundle` (cwd `go/`); fail loudly on non-zero.
4. **Start Airflow** — reuse if `api_up`; else `Popen(["uv","run","airflow","standalone"])` with
   stdout/stderr → `standalone.log`, write `standalone.pid`, poll `api_up` (90×2s); tail log + exit
   if it never comes up.
5. **Register + provision** — poll `dag_registered` (45×2s; error if never); `airflow dags unpause`,
   `airflow pools set` both pools.
6. **Config + trigger + wait** — `airflow variables set dakp_config '<json>'`, `airflow dags trigger
   dakp_build`. With `--detach`, print the trigger result + log/URL pointers and return (exit 0)
   without polling. Otherwise poll `run_state` (300×3s); on `success` print + cat
   `build_summary.json` (exit 0), on `failed`/timeout print log pointers (exit 1).

`down` ports `dakp_down.sh`: `os.kill(pid, SIGTERM)` for the pidfile pid if alive, remove pidfile,
then the same `pkill -f "airflow …"` catch-alls via `run_subprocess` (no psutil dep).

### Fullmap: thread `--fullmap`, remove every download path

The plumbing to consume `ctx.params["fullmap"]` **already exists and is tested**
(`tablassert/run.py` reads `ctx.params.get("fullmap")`; `test_tablassert_configs.py:435` passes
`fullmap="data/fullmap"` → `--fullmap data/fullmap`). What's missing is producing it from the run
config, plus deleting the download paths:

1. **Thread it** — `runtime.build_context_from_config` adds `cfg.get("fullmap")` to the `extra`
   params dict (alongside the existing `drugsfda_url` handling) so it lands in `ctx.params["fullmap"]`.
   The CLI writes `"fullmap": <path|null>` into the `dakp_config` Variable.
2. **`tablassert/run.py`** — drop `DEFAULT_FULLMAP = ".fullmap"`. For a **real** (non-deferred) run,
   `fullmap` must be present in `ctx.params`; if absent, raise a clear error ("pass `--fullmap
   <path>` — DAKP no longer downloads a fullmap"). Mock/deferred runs never call tablassert, so
   they're unaffected and need no fullmap.
3. **`tablassert/configs.py`** — drop the `"fullmap": FULLMAP_DEFAULT` key from the generated
   `graph.yaml` (remove `FULLMAP_DEFAULT = ".fullmap"`). DAKP always supplies `--fullmap <path>`
   explicitly on the `build-kg` command, so nothing in the generated config points tablassert at a
   downloadable `.fullmap`.
4. **Delete `acquire_ontologies` entirely** (`acquire.py`) — with the fullmap gone it is vestigial:
   the DAG passes its refs to `run_tablassert` and immediately `del ontologies` (never read), and the
   mock disease map is loaded **directly** from `fixture_root/ontology/disease_map.tsv` in
   `runtime.build_context` (not from the acquired store). Remove `acquire_ontologies`,
   `default_ontology_sources`, `_ingest_ontology_fixtures`, `DEFAULT_FULLMAP_SOURCE`,
   `_ONTOLOGY_FIXTURE_GLOB`, the acquire-local `_download_to` + `_now_iso` + `Downloader` alias (all
   used only by this task), and the now-unused imports (`shutil`, `urllib.request`, `SourceBlock`,
   `Mapping`, `datetime`/`UTC` — whatever ruff/pyright flag). Drop the `"ontologies"` job from
   `acquire_all` (its `downloader` param retypes to `model_cache.Downloader | None`; docstring
   "five acquisitions" → "four"). The ontology fixture **files stay** in
   `tests/fixtures/pipeline/ontology/` (still read directly by `runtime`/the NER backend).
5. **`config.py`** — delete `DownloadConfig.fullmap_source` **and** `DownloadConfig.ontology_sources`
   (both now dead). `DownloadConfig` keeps `concurrency`, `ner_model_ids`, `drugsfda_url`.
6. **DAG (`dags/dakp_build.py`)** — delete the `acquire_ontologies` task, the `ontologies =
   acquire_ontologies()` wiring, and the `ontologies` parameter (and `del ontologies` + stale comment)
   on `run_tablassert`; its signature becomes `run_tablassert(approved, uses, contra, configs)`.

### Removals

- Delete `scripts/dakp_up.sh`, `scripts/dakp_down.sh`, and the now-empty `scripts/` dir.
- Delete `Makefile` (`install` is just `uv sync`; `clean` → `uv run dakp clean`).
- Delete `.envrc` and drop direnv from docs; remove the `.direnv/` line comment in `.gitignore`.

### Dependency change

- Add `cyclopts` to `dependencies` (runtime — the CLI is the entrypoint). Pulls `rich`/`attrs`;
  acceptable given the CLI replaces the whole shell layer. Regenerate `uv.lock`.

## Files to modify

- **New:** `src/dakp_pipeline/cli.py`
- **New tests:** `tests/unit/test_cli.py` (+ `tests/unit/test_cli_edge.py` for the real side-effect
  bodies, mirroring `test_tablassert_run_edge.py`)
- `pyproject.toml` — add `cyclopts` dep + `[project.scripts]`; `uv.lock` regenerated
- **Fullmap / acquire_ontologies removal:** `src/dakp_pipeline/runtime.py` (thread `fullmap` into
  params), `src/dakp_pipeline/tablassert/run.py` (require fullmap for real runs; drop `.fullmap`
  default), `src/dakp_pipeline/tablassert/configs.py` (drop `fullmap` key from graph.yaml),
  `src/dakp_pipeline/acquire.py` (delete `acquire_ontologies` + helpers + fullmap stub; trim
  `acquire_all`), `src/dakp_pipeline/config.py` (drop `DownloadConfig.fullmap_source` +
  `ontology_sources`), `src/dakp_pipeline/dags/dakp_build.py` (delete the task + wiring + param)
- **Existing tests to update:** `tests/unit/test_dag_downloads.py` (delete the three
  `acquire_ontologies`/`default_ontology_sources` tests; drop `fullmap_source`/`ontology_sources`
  asserts; drop `"ontologies"` from the `acquire_all` + DAG-structure asserts),
  `tests/unit/test_dag.py` (drop `acquire_ontologies` from `_ACQUIRE_IDS` + the `run_tablassert`
  upstream set), `tests/unit/test_acquire_edge.py` (delete the `acquire_ontologies` + `_download_to`
  edge tests; fix the module docstring), `tests/unit/test_tablassert_configs.py` (graph.yaml no longer
  has `fullmap`; `build-kg` gets the explicit path)
- **Delete:** `scripts/dakp_up.sh`, `scripts/dakp_down.sh`, `scripts/`, `Makefile`, `.envrc`
- `.gitignore` — drop the `.direnv/` comment
- Docs: `README.md`, `docs/runbook.md`, `docs/wenceslaus-runbook.md`, `docs/tablassert-handoff.md`,
  `docs/sources.md`, `go/README.md`. The bounded-`prod`-smoke-via-env sections (README, runbook,
  wenceslaus) are rewritten: the CLI can't bound scope and reads no env, so a real run is
  `uv run dakp up prod --fullmap <path>` (full-scope), and the offline bounded smoke is pointed at
  `tests/integration/test_prod_smoke.py`. (+ docstring pass: `pipeline.py`, `dags/__init__.py`,
  `dags/dakp_build.py`, `tests/integration/test_mock_pipeline.py`)

## Reuse

- `runtime.build_context_from_config` already maps `null` limits → profile default → no config change.
- `dags/dakp_build.py` constants (`DAG_ID`, `CONFIG_VARIABLE`, `DOWNLOAD_POOL`, `EXTRACT_POOL`,
  `GO_QUEUE`) — import them so the CLI and DAG can't drift.
- `tablassert/run.py`'s `run_subprocess` + monkeypatch-point convention — copy the pattern for the
  CLI's side-effect functions so tests need no real Airflow/Go/network.

## Tests (must hold the 100% branch-coverage gate)

Deleting `acquire_ontologies`/`_download_to` removes their tests *and* their covered branches together
(net-neutral for coverage); the new `cli.py` is what adds branches that need covering.

`tests/unit/test_cli.py` monkeypatches the side-effect functions + `sleep` to drive every branch:
- preflight: healthy · reinstall-heals · cache-clean-heals · still-broken (exit≠0)
- go pack failure (exit≠0)
- api: already-up (reuse) · start→up · start→never-up (error, log tail)
- dag: registered · never-registered (error)
- trigger failure
- `--detach` / `-d`: returns right after trigger (no polling) · default waits
- run_state: success (prints + cats summary) · failed · timeout
- `dakp_config` Variable always carries `null` limits + the `fullmap` path (or null); short aliases
  (`-f`, `-p`, `-l`) resolve to the same values as the long flags
- down: pidfile alive · pidfile dead · no pidfile; pkill calls issued
- clean: removes the expected paths

`tests/unit/test_cli_edge.py` exercises the *real* bodies (no monkeypatch): `run_subprocess`
capturing a real exit code, `api_up` against a closed port → `False`, the sqlite helpers against a
temp db, `airflow_importable`.

## Verification

1. `uv sync` (materializes the `dakp` console script + cyclopts).
2. `uv run dakp --help` / `uv run dakp up --help` render cyclopts help with all defaulted options.
3. `uv run dakp up` (no flags, no env, no direnv) runs the mock DAG end-to-end, prints `SUCCESS` +
   `build_summary.json` — the one-command happy path. (Also proves deleting `acquire_ontologies`
   didn't break the mock run: contraindications still get their disease map from `fixture_root`.)
4. `uv run dakp down` stops Airflow (`pgrep -af airflow` empty).
5. `uv run dakp up prod --fullmap /local_raid1/.../fullmap.redb` — a real full-scope build (the only
   prod shape the CLI offers now; scope bounding moved out of the CLI).
5b. `uv run dakp up prod -f /local_raid1/.../fullmap.redb -d` triggers and returns immediately (short
   aliases work).
6. Short aliases + defaults: `uv run dakp up -p 8091` starts on :8091; `uv run dakp up` with no flags
   uses every built-in constant (no env vars set or read).
7. `uv run pytest -q --cov` stays at 100% branch coverage; `uv run pre-commit run --all-files` clean.
8. `grep -rn "make up-\|make down\|dakp_up.sh\|dakp_down.sh\|direnv\|\.envrc\|QUARTER_LIMIT\|RELEASE_LIMIT\|WORKDIR=" README.md docs/ go/README.md src/`
   → no stale references (outside `plans/`, `PLAN.md`).

## Decisions (confirmed with the user)

1. **Delete the Makefile** entirely — `clean` → `uv run dakp clean`; install is just `uv sync`.
2. **Delete `.envrc` and drop direnv** — defaults live in the CLI; the CLI reads no env vars (see #5).
3. **Command shape** — profile is an optional positional (`uv run dakp up prod`); add a `--detach`
   flag (trigger and return without waiting).
4. **Fullmap** — thread `--fullmap` to `build-kg`; require it for a real handoff (clear error if
   missing, mock unaffected); drop `fullmap` from generated `graph.yaml`; and **delete
   `acquire_ontologies` entirely** (plus `DownloadConfig.fullmap_source` + `ontology_sources`, the
   fullmap.redb stub, and the DAG task/wiring). It was vestigial: its refs were `del`-ed unread in
   the DAG, and the mock disease map is read straight from `fixture_root`, not the acquired store.
5. **Minimal flags + hardcoding** — the only `up` flags are `--fullmap/-f`, `--port/-p`,
   `--log-level/-l`, `--detach/-d` (plus the `profile` positional). Workdir, fixture root, Airflow
   home, the task pools, and the (always-`null`) scope limits are hardcoded constants in `cli.py`.
   The CLI reads **no env vars**. Consequence: a CLI `prod` run is always full-scope (no bounding);
   the bounded smoke lives on only in `tests/integration/test_prod_smoke.py`.
