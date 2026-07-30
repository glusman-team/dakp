# Plan: Airflow-Native Rebuild — Native Go SDK Workers + Airflow-Dependent Design

## Context

**The question that prompted this:** does the pipeline use Airflow's native Go workers (the
Airflow 3 Go Task SDK) instead of OS commands? **No.**

- The repo is pinned to **Airflow 2.x** (`apache-airflow>=2.10,<3`). The Go Task SDK requires
  **Airflow 3** (AIP-72 / Task Execution Interface) and does not exist in Airflow 2.
- There are **zero** references to `go-sdk`, `task.stub`, `bundlev1`, `ExecutableCoordinator`,
  or `airflow-go-pack` anywhere in the repo.
- The Go workers are a **standalone CLI** (`go/cmd/dakp-worker`, custom `internal/registry`
  dispatcher). Python drives them the OS-command way: `workers/go_runner.py` shells out via
  `subprocess.Popen`, parses `b3:` artifact ids / JSON off stdout, relays `log/slog` JSON from
  stderr into loguru. The three extractors (`extract/spl_xml.py`, `extract/faers_ascii.py`,
  `extract/drugsfda_products.py`) gate this behind `go_runner.should_use_go(ctx)`.

**User decision (this effort):** rebuild the design to be **entirely Airflow-dependent**,
**allow Airflow dependencies** (Airflow stops being an optional extra), and convert the Go
extractors into **native Airflow Go SDK bundle workers** (no more subprocess).

This **deliberately retires** the long-standing invariants that the Go workers are
Airflow-independent and that `pipeline.run_pipeline` is the canonical, Airflow-optional runner.
Those were the main reasons *not* to do this; the user has chosen to drop them.

### What the Airflow Go SDK is (Airflow 3, experimental)

- DAGs are still **defined in Python**; individual **task implementations are written in Go**.
- Go tasks are compiled ahead of time into a single self-contained **bundle** executable via
  `airflow-go-pack` (shipped through the Go 1.24 `tool` directive). The bundle embeds its DAG
  source + a `dag_id`/`task_id` manifest footer; the executable *is* the bundle.
- **Coordinator mode (recommended):** the Airflow worker's Python supervisor forks the bundle
  once per task instance; the bundle dials back over TCP loopback (msgpack-over-IPC). Inherits
  remote logs (S3/GCS), full task states, and alternate XCom backends. **No separate Go worker
  process.** (Alternative: **Edge Worker** — a long-running pure-Go process, no Python in the
  data path, but missing remote logs / non-default XCom backends / non-success-failed states.)
- A Go task is an ordinary function; the runtime injects args by type: `sdk.TIRunContext`
  (cancellation + TI/DAG-run ids), `*slog.Logger` (routed to the Airflow task log), and
  `sdk.Client` / narrower interfaces (`GetVariable`, `GetConnection`, `GetXCom`, `PushXCom`).
  An `(any, error)` return becomes the task's `return_value` XCom; a non-nil error fails the task.
- The Python side declares Go tasks as `@task.stub(queue="golang")` (shape + dependencies only);
  `[sdk] coordinators` + `queue_to_coordinator` in `airflow.cfg` route the queue to the
  `ExecutableCoordinator`, which scans `executables_root` for bundles.
- **Requires:** Go 1.24+ (repo `go.mod` is `go 1.26.4` — OK), Airflow 3.x,
  `apache-airflow-task-sdk` (ships with Airflow). **Experimental:** APIs / wire protocols /
  tooling may change between releases without notice.

## Current state (what changes)

- **Dual-path orchestration.** `pipeline.run_pipeline` is the canonical runner; `dags/dakp_build.py`
  is a TaskFlow wrapper over the *same* `STAGE_CALLABLES`, reusing `pipeline._build_context` and
  `pipeline._write_build_summary`. `tests/unit/test_dag.py` asserts the DAG mirrors the runner.
- **Go via subprocess.** `workers/go_runner.py`: `GoRunner` (Popen + build cache keyed by a hash
  of the Go sources), `MockGoRunner` (test seam), `should_use_go` (delegation gate on
  `Profile.use_go_workers`), helpers `stage_inputs` / `read_go_tsv` / `go_rows` / `go_warnings`.
- **Airflow is optional.** `[project.optional-dependencies].airflow`; the DAG module is
  import-safe without Airflow (no-op `@dag`/`@task` fallbacks); the whole test suite runs without
  Airflow; coverage is 100%.
- **Go module** `github.com/glusman-team/dakp/go`: standalone CLI; extractors in
  `internal/{dailymed,faers,drugsfda}`; `internal/blake3store` (BLAKE3 file/tree hashing +
  `ArtifactManifest` round-trip, byte-compatible with Python). Deps: `zeebo/blake3`,
  `golang.org/x/sync`. **No** parquet writer (explicitly deferred in `go/README.md`). **No** go-sdk.
- **Data plane.** Tasks pass `ArtifactRef` manifests (paths + BLAKE3 ids), never dataframes.
  Heavy data moves through the BLAKE3 content-addressed filesystem store. **Interim tables are
  parquet** (`spl_*.parquet`, `cases.parquet`, `products.parquet`, …); the shaping stage reads
  them by name (`assertions/evidence.py`: `spl_approvals.parquet`, `spl_ingredients.parquet`,
  `cases.parquet`, `products.parquet`). The Tablassert handoff is uncompressed TSV.
- **No deployment config in-repo** (no `airflow.cfg`, `docker-compose`, `Dockerfile`).

## Target state

- **Airflow 3 is a hard dependency** (moved into `dependencies`; import-guard fallbacks deleted).
- **The DAG is the single orchestrator and entrypoint.** The `run_pipeline` dual path is collapsed:
  `run_pipeline` is deleted and `dakp run` becomes a thin DAG-run trigger (decision B).
- **Go extractors are native Go SDK bundle tasks** (Coordinator mode) for `extract_dailymed`,
  `extract_faers`, `extract_drugsfda`. Each task: reads the upstream `list[ArtifactRef]` from XCom,
  stages the input files, runs the existing `internal/{...}` parser, finalizes artifacts into the
  content-addressed store, and pushes the output `list[ArtifactRef]` as XCom. Heavy bytes still
  move via the shared filesystem store; XCom carries only the small manifests.
- **The subprocess layer is deleted** (`go_runner.py`, `should_use_go`, `_extract_via_go`,
  `MockGoRunner`, `Profile.use_go_workers`). The pure-Python extractors are retained as the
  reference implementation / parity oracle / mock path (decision C).
- **Acquisition / shaping / Tablassert / contract / regression / summary stay Python** TaskFlow
  tasks (real `@task`, Airflow-native), reusing the existing stage modules.
- **A deployment story exists**: `airflow.cfg` `[sdk]` coordinator config, `executables_root`,
  a local `airflow standalone` / docker-compose, and a Makefile target that builds + packs the
  bundle into `executables_root`.

## Decisions (locked)

- **A. Deployment mode: Coordinator.** The Airflow worker's Python supervisor forks the Go bundle
  once per task instance; inherits remote logs (S3/GCS), full task states, and alternate XCom
  backends. No separate Go worker process to run. (Edge Worker rejected: missing those features.)
- **B. Runner/CLI: delete the CLI entirely (user: "remove CLI if airflow does the same things").**
  `run_pipeline`, `PipelineResult`/`TableResult`, and `cli.py` (the `[project.scripts] dakp`
  entrypoint) are all removed — no separate Python CLI/dev tool to maintain. Airflow is the only
  way to run the pipeline; the one-command entrypoint is `make run` (the orchestrator script),
  which triggers via `airflow dags trigger` directly.
- **C. Scope: 3 extractors → Go bundle; Python extractors stay.** Only `extract_dailymed`,
  `extract_faers`, `extract_drugsfda` become native Go SDK tasks (`hash` stays a dev/utility CLI
  subcommand, not a DAG task). Acquisition / shaping / Tablassert / contract / regression / summary
  stay Python TaskFlow. The pure-Python extractors are retained as the reference implementation +
  golden-fixture parity oracle + mock path; only the subprocess bridge is deleted. The `dakp-worker`
  CLI is kept for dev / `make check-go` / parity.
- **D. Self-contained Go tasks via a Go parquet writer (D1).** Add `github.com/parquet-go/parquet-go`
  so each Go extract task writes the parquet interim tables + registers them in the BLAKE3
  content-addressed store + writes the `ArtifactManifest` itself (reusing `internal/blake3store`).
  The Python shaping stage is unchanged (it keeps reading `spl_*.parquet` / `cases.parquet` / etc.).
- **Framing accepted:** the Go SDK is experimental (moving wire protocols) and Airflow 3 becomes a
  hard dependency; even the mock smoke test requires a running local Airflow + a packed bundle.
- **E. One-command install + one-command run (user requirement).**
  - *Install:* a single `uv sync --all-extras` installs everything needed to run the FULL pipeline
    (Airflow 3 + task-sdk, the NER backend, Tablassert). Airflow moves from an optional extra into
    the required `dependencies`; the heavy `ner`/`kg` extras stay optional but a documented
    `make install` (= `uv sync --all-extras`) brings them in one shot. No juggling multiple extras.
  - *Run:* a single `make run` (backed by `scripts/dakp_up.sh`, using `airflow dags trigger`) does the
    whole thing with no manual steps: build+pack the Go bundle into `executables_root` -> start
    Airflow (`airflow standalone`, or docker-compose) with the `[sdk]` coordinator config -> wait
    for health -> set the `dakp_config` Variable (workdir/profile/fixtures) -> trigger `dakp_build`
    -> wait for the run to finish -> print the build-summary path. `make down` tears it down.

## Approach (phased)

### Phase 0 — Airflow 2→3 upgrade + hard dependency + deployment skeleton
- [ ] Bump `pyproject.toml`: move Airflow into `dependencies` as `apache-airflow>=3,<4`
      (+ `apache-airflow-task-sdk`); delete the `[project.optional-dependencies].airflow` extra.
- [ ] Update DAG imports to the Airflow 3 Task SDK (`from airflow.sdk import dag, task`); delete
      the `try/except ImportError` no-op fallbacks in `dags/dakp_build.py` (Airflow is now required).
- [ ] Audit Airflow 2→3 breaking changes used here: `airflow.decorators` → `airflow.sdk`,
      `pendulum.datetime` start_date, `params` handling, pool names, executor/config defaults.
- [ ] Add a local deployment: `airflow standalone` instructions and/or `docker-compose.yaml`
      (API server + scheduler + worker + the `[sdk]` coordinator config). Airflow is now needed to
      run anything, so a one-command local bring-up is required.
- [ ] Update `uv.lock`, CI, and the dev docs to install Airflow unconditionally.

### Phase 1 — Native Go SDK bundle (the extract workers)
- [ ] Add the Go SDK to `go/go.mod`: `require github.com/apache/airflow/go-sdk` and the
      `tool github.com/apache/airflow/go-sdk/cmd/airflow-go-pack` directive.
- [ ] New bundle entrypoint (e.g. `go/cmd/dakp-bundle/main.go`) implementing
      `bundlev1.BundleProvider`: `GetBundleVersion` + `RegisterDags` → `AddDag("dakp_build")`
      with `AddTask(extract_dailymed)`, `AddTask(extract_faers)`, `AddTask(extract_drugsfda)`.
      (Task function **names must match** the Python `@task.stub` names; the SDK takes `task_id`
      from the Go function name, so these are snake_case Go funcs.)
- [ ] Implement the three SDK task funcs
      `func extract_<src>(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger) (any, error)`:
      - `client.GetXCom` → upstream `list[ArtifactRef]` (JSON → `[]any`); stage the files (reuse
        the staging logic currently in Python `go_runner.stage_inputs`).
      - Call the existing `internal/{dailymed,faers,drugsfda}` parser (unchanged).
      - **Finalize artifacts (D1):** write the parquet interim tables via `parquet-go` (matching the
        Python polars schema: same column names/order/types), register them in the BLAKE3
        content-addressed store, and write the `ArtifactManifest` — reusing `internal/blake3store`
        for hashing/manifests plus a small new Go store-write helper mirroring `io/artifact_store.py`.
      - Return / `client.PushXCom` the output `list[ArtifactRef]` (manifests only; bytes are on the
        shared FS). Honor `ctx` cancellation; log via the injected `*slog.Logger`.
- [ ] Keep the `dakp-worker` CLI for dev/parity (the bundle is an additional entrypoint, not a
      replacement) — per decision C.
- [ ] Go tests: keep the existing TSV byte-parity goldens for the parser core, and add bundle-wrapper
      tests (XCom in → parse → parquet artifacts + manifests out). Parquet parity is asserted at the
      **logical-table** level (columns + rows + the `ArtifactRef` contract), not parquet bytes —
      parquet bytes legitimately differ across writers.

### Phase 2 — Rewire the DAG as the single source of truth
- [ ] In `dags/dakp_build.py`: declare the three extract tasks as `@task.stub(queue="golang")`
      (no Python body); keep acquisition / shaping / Tablassert / contract / regression / summary
      as real Python `@task`s wrapping the existing stage modules.
- [ ] Invert the "DAG mirrors `run_pipeline`" invariant: the DAG *is* the pipeline. Drop the
      `STAGE_CALLABLES`-completeness coupling to `run_pipeline` and the `test_dag.py` mirror assertion.
- [ ] XCom contract: acquisition tasks push `list[ArtifactRef]`; extract stubs consume/produce
      `list[ArtifactRef]`; shaping consumes refs. Manifests serialize cleanly to JSON XCom.

### Phase 3 — Delete the subprocess layer + the CLI (decision B = delete CLI)
- [ ] Delete `workers/go_runner.py` (`GoRunner`, `MockGoRunner`, `should_use_go`, `stage_inputs`,
      `read_go_tsv`, `go_rows`, `go_warnings`, build-cache, `ENV_BINARY`/`ENV_CACHE_DIR`).
- [ ] Delete `_extract_via_go` from `extract/{spl_xml,faers_ascii,drugsfda_products}.py` and the
      `should_use_go` gates; keep the pure-Python `extract` paths (decision C).
- [ ] Remove `Profile.use_go_workers` (`config.py`) and its propagation through the context builder.
- [ ] Extract the shared helpers the DAG reuses — `_build_context`, `_write_build_summary`,
      `_load_disease_map` — out of `pipeline.py` into a new `runtime.py`; delete `run_pipeline` and
      `PipelineResult`/`TableResult`. The DAG imports these helpers from `runtime.py`.
- [ ] Delete the CLI entirely: remove `cli.py` + the `[project.scripts] dakp` entrypoint, and delete
      `run_pipeline` / `PipelineResult` / `TableResult` (the shared `_build_context` /
      `_write_build_summary` / `_load_disease_map` helpers move to `runtime.py`, imported by the DAG).
      Airflow is the only entrypoint; `make run` triggers via `airflow dags trigger`.
- [ ] Delete/rewrite the tests that exercised the removed seams (`test_go_runner*.py`, the
      `test_dag*.py` mirror assertion, `run_pipeline`-based integration tests → DAG-trigger +
      Airflow integration tests).

### Phase 4 — One-command install + one-command end-to-end run (decision E)
- [ ] `pyproject.toml`: Airflow 3 in required `dependencies`; keep `ner`/`kg` as extras; document
      `uv sync --all-extras` as the single full install. `make install` wraps it.
- [ ] `airflow.cfg` / `AIRFLOW__SDK__*`: register the `go` coordinator
      (`airflow.sdk.coordinators.executable.ExecutableCoordinator`, `executables_root`) and
      `queue_to_coordinator = {"golang": "go"}`. Pin the task-sdk schema to the bundle's
      `supervisor_schema_version` (2026-06-16).
- [ ] `scripts/dakp_up.sh` (the one-command orchestrator): build+pack the bundle -> start Airflow
      standalone (background, logs to workdir) -> poll the API server until healthy -> create the
      `dakp_config` Variable -> `dakp run` (trigger + wait) -> print build-summary path. Idempotent
      (reuses a running Airflow if already up).
- [ ] Makefile targets: `install`, `bundle` (`go tool airflow-go-pack --output <executables_root>/dakp-bundle ./go/cmd/dakp-bundle`,
      with `--goos/--goarch` cross-build), `up`/`run` (the orchestrator), `down`.
- [ ] Document the shared-volume requirement (worker + bundle must see the same content-addressed
      store / workdir for the filesystem data plane to work).

### Phase 5 — Testing rework (Airflow now in the test env)
- [ ] Install Airflow in CI/dev unconditionally; re-establish coverage (the 100% gate will need
      the Airflow-only paths brought under test or explicitly excluded).
- [ ] DAG structure tests via Airflow test utils (task graph, stub↔bundle name matching).
- [ ] Integration: run the mock DAG end-to-end against a local Airflow + the packed bundle,
      asserting the same assertion-table outputs the old `run_pipeline` mock produced.
- [ ] Keep the Go golden-parity tests as the extractor correctness oracle.

### Phase 6 — Docs rewrite
- [ ] `docs/architecture.md`, `README.md`, `go/README.md`, `docs/runbook.md`: reflect the
      Airflow-native, Airflow-required design; drop "Airflow-independent runner" / "optional
      extra" / "shell out to the worker" language; document the bundle build + coordinator config.

## Files to modify / add / delete

**Add:**
- `go/cmd/dakp-bundle/main.go` (+ per-task files) — the Go SDK bundle entrypoint + 3 task funcs.
- A small Go store-write helper (mirror of Python `io/artifact_store.py`) under `internal/` for
  Go-side artifact registration + manifest writing (reuses `internal/blake3store`).
- `src/dakp_pipeline/runtime.py` — the shared `_build_context` / `_write_build_summary` /
  `_load_disease_map` helpers extracted from `pipeline.py` (imported by the DAG).
- Deployment: `docker-compose.yaml` and/or `airflow.cfg` (or documented `AIRFLOW__SDK__*` env),
  `executables_root` location.

**Modify:**
- `pyproject.toml` (Airflow → hard dep; drop the extra), `uv.lock`.
- `dags/dakp_build.py` (Airflow 3 imports; extract tasks → `@task.stub`; DAG = source of truth).
- `extract/{spl_xml,faers_ascii,drugsfda_products}.py` (delete `_extract_via_go` + gates).
- `config.py` (drop `use_go_workers`), `dags/dakp_build.py` (Airflow 3 imports; extract tasks →
  `@task.stub`; import shared helpers from `runtime.py`; DAG = source of truth).
- `go/go.mod` / `go/go.sum` (add go-sdk + the `airflow-go-pack` tool directive + `parquet-go`).
- `Makefile` (bundle pack target; Airflow bring-up).
- Docs: `README.md`, `docs/architecture.md`, `go/README.md`, `docs/runbook.md`.

**Delete:**
- `workers/go_runner.py` (+ `test_go_runner.py`, `test_go_runner_edge.py`).
- `cli.py` (+ `test_cli.py`, `test_cli_edge.py`) and the `[project.scripts] dakp` entrypoint.
- `pipeline.py` once its helpers have moved to `runtime.py` (`run_pipeline`, `PipelineResult`,
  `TableResult` removed) — plus `test_pipeline_edge.py` / `test_mock_pipeline.py` as rewritten.
- Tests coupled to the removed seams (per Phase 3/5).

## Reuse

- `internal/{dailymed,faers,drugsfda}` Go parsers — reused **unchanged** inside the bundle tasks
  (this is the actual heavy lifting; only the invocation wrapper changes).
- `internal/blake3store` — BLAKE3 file/tree hashing + `ArtifactManifest` round-trip (byte-compatible
  with Python) — reused for Go-side artifact finalization (store registration + manifests).
- Python stage modules (`acquire`, `assertions/*`, `tablassert/*`, `translator/*`) — reused as the
  bodies of the remaining Python TaskFlow tasks.
- Existing Go golden-parity fixtures (`internal/*/testdata/golden`) — reused as the bundle's
  correctness oracle.
- `ArtifactRef` manifest model — already small + JSON-serializable, so it flows over XCom cleanly.

## Verification

- [ ] `uv sync` installs Airflow 3 unconditionally; `uv run python -c "import airflow"` works.
- [ ] `go build ./... && go test ./...` green in `go/` (incl. new bundle-wrapper tests + parity).
- [ ] `go tool airflow-go-pack --output <executables_root>/dakp-bundle ./go/cmd/dakp-bundle`
      produces a bundle whose `--airflow-metadata` lists `dag_id: dakp_build` + the 3 task ids.
- [ ] Local Airflow brings up (standalone/docker-compose); the coordinator discovers the bundle.
- [ ] **One command runs the whole pipeline:** `make install && make run` (mock profile) brings up
      Airflow, packs the bundle, triggers `dakp_build`, and completes with no other manual steps;
      extract tasks execute in the Go bundle (no Python subprocess), and the assertion tables +
      `build_summary.json` match the old `run_pipeline` mock output byte-for-byte. `make down` cleans up.
- [ ] `uv run ruff check`, `ruff format --check`, `pyright` clean; `uv run pytest` green with the
      re-established coverage gate.
- [ ] `grep -rn "subprocess\|should_use_go\|use_go_workers\|MockGoRunner" src/` → no matches.

## Risks / tradeoffs (accepted by this direction)

- **Experimental SDK:** wire protocols / APIs / tooling may change between Airflow releases
  without notice → maintenance overhead pinning/adjusting.
- **Airflow 2→3 migration breadth:** execution API server, assets, DAG bundles, executor/config
  changes — a project in itself, front-loaded in Phase 0.
- **Loss of the Airflow-independent dev loop:** running *anything* (even the mock smoke test)
  now requires a running local Airflow + a packed bundle (`dakp run` survives only as a trigger
  against that Airflow). Heavier local dev and CI.
- **Heavier tests:** Airflow + bundle packaging enter the test path; the 100%-coverage invariant
  needs deliberate re-establishment.
- **Shared-filesystem coupling:** the filesystem data plane requires the worker/bundle to see the
  same store; this constrains deployment topology (single volume / networked store).
