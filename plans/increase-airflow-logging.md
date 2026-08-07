# Dramatically increase per-step Airflow logging (one stat per line)

## Context

The user can't tell what the pipeline is doing from the Airflow UI. Investigation of the last
local run (`tmp/airflow-home/logs/dag_id=dakp_build/...`) found **four compounding problems**:

1. **Stats are silently dropped (Python).** Stage code logs stats as loguru kwargs —
   `log.info("faers quarter acquired", quarter=..., artifact_id=..., cache_hit=...)` — but the
   sinks use loguru's default format which does NOT render `{extra}`, and the Airflow forwarder
   only ships `record["message"]`. Verified: the log line renders as bare
   `faers quarter acquired` with every stat discarded.
2. **The Airflow bridge is dead code on Airflow 3.3.** `configure_logging(for_airflow=True)`
   (a) replaces the stdlib root handlers with `InterceptHandler` (`basicConfig(force=True)`),
   destroying the structlog handler Airflow 3 installs to write task logs, and (b) forwards into
   `logging.getLogger("airflow.task")` which has no handlers → records vanish. The only reason
   anything is visible is that loguru's stderr sink is captured by Airflow's subprocess capture
   as `task.stderr` at **level ERROR**, double-stamped with loguru's own prefix.
3. **Go extract tasks are nearly silent.** `extract_dailymed` / `extract_faers` /
   `extract_drugsfda` (the heaviest steps — minutes to hours) log only `extract start` /
   `extract done` from the bundle adapter; `internal/airflow/Extract*` logs nothing.
4. **Multi-stat lines are hard to read in Airflow.** Go slog records render as one raw JSON
   line with all attributes inline (`{"event":"extract start","inputs":1,...}`); the user wants
   each stat on its own log line (rows separate from blake3 hash, etc.).

Intended outcome: every DAG step narrates what it is doing (start, progress, counts, hashes,
elapsed), and every individual stat is its own self-contained, greppable log line.

## Decisions (confirmed with user)

1. **Line format**: `event: key = value`, event name repeated on every line (self-contained/greppable).
2. **DailyMed per-document**: per-document lines at **DEBUG** + periodic **INFO** progress every
   N documents (default `2500`) + per-release summary.
3. **Tablassert**: stream subprocess output **live** into the task log, filtering progress-bar
   redraw noise (only real log lines). Verified feasible: Tablassert renders its progress via
   rich (`Console(stderr=True)` + `Live`); under a pipe (non-TTY) rich emits the bar's final
   state once per task — no redraw flood (tested locally with the installed rich). Defensive
   filtering still applies: strip ANSI escapes, collapse `\r` runs to the last segment, drop
   blank and duplicate-consecutive lines.
4. **Go bundle**: included (repacked automatically by `uv run dakp up`).
5. **Bridge fix**: included (prerequisite for readable Python logs).

## Approach (proposed)

### Log line convention: one stat per line

Event-prefixed self-contained lines, one stat each:

```
extract_faers: quarter = 24Q3
extract_faers: rows = 1234567
extract_faers: blake3 = b3:9317...
extract_faers: cache_hit = true
extract_faers: elapsed_s = 42.7
```

Same convention in Python (loguru message text) and Go (slog event text), so lines stay
readable whether Airflow renders the `event` field or the raw JSON.

### Python

- `logging_setup.py`:
  - new `stats(log, event, **fields)` helper → one `event: key = value` line per field
    (insertion order preserved); optional level.
  - new `step(log, event)` context manager → `event: started` on entry; on success
    `event: finished = true` + `event: elapsed_s = N` (one line each); on exception
    `event: failed = true` + `event: error = <ExcType>` + `event: elapsed_s = N`, re-raise.
  - new `progress(log, event, done, total, every)` → emits `event: progress = <done>/<total>`
    whenever `done % every == 0` (used by DailyMed expansion; cheap no-op otherwise).
  - **Bridge fix** (`configure_logging(for_airflow=True)` path):
    - do NOT install `InterceptHandler` / `basicConfig(force=True)` — Airflow 3 already owns the
      stdlib root handler (structlog ProcessorFormatter → task log socket); clobbering it is why
      forwarded records currently vanish.
    - do NOT add the loguru stderr sink (Airflow captures it as ERROR-level `task.stderr`
      noise — the double-stamped lines in today's logs). Keep the `<workdir>/logs/dakp.log`
      file sink.
    - `_stdlib_record_sink` rebuilds each record with the REAL loguru level (WARNING shows as
      WARNING, not flattened to the logger's effective level), keeps the origin module as the
      logger name, tags a sentinel attribute, and emits via `logging.getLogger(name).handle()`
      so it propagates to Airflow's root handler.
    - `InterceptHandler.emit` skips records carrying the sentinel (loop-proof if both bridges
      are ever active). Non-Airflow path (`for_airflow=False`: tests/local) keeps today's
      stderr+intercept behavior unchanged.
- Stage narration (start / inputs / progress / per-stat completion lines, elapsed per phase):
  - `dags/dakp_build.py` — every task body: started + input ref counts; finished + output ref
    count + elapsed (task bodies are `# pragma: no cover`, so no coverage impact).
  - `sources/dailymed.py` — index fetch (url, cached/304, bytes), releases discovered (count;
    each url at DEBUG), per-release: download start/complete (bytes, elapsed_s, cache), zip
    ingest blake3, expansion: documents discovered, `progress()` every 2,500 docs ingested
    (INFO), per-doc ingest at DEBUG, per-release summary (xml_count, elapsed_s).
  - `sources/faers.py` — index fetch, quarters discovered (count; each quarter at DEBUG),
    per-quarter: cache-hit (blake3) or download (url, bytes, elapsed_s) + ingest (blake3,
    cache_hit), acquisition complete (artifacts, elapsed_s).
  - `sources/drugsfda.py` — download start (url), complete (bytes, elapsed_s), ingest
    (blake3, cache_hit).
  - `acquire.py` + `ner/model_cache.py` — per-model cache-hit/download lines (model_id, b3,
    source, cache_hit) split one-per-line.
  - `io/artifact_store.py` — ingest/register DEBUG lines (alias, path, blake3, bytes, rows,
    cache_hit, media_type) — DEBUG because DailyMed ingests tens of thousands of SPL docs.
  - `assertions/{approved_treats,observed_uses,contraindications,evidence}.py` — step
    wrappers around each transform; evidence index sizes (approvals, sets, ingredients,
    indication docs, contraindication docs — one line each), FAERS case-table resolution
    (path, rows, projected columns), candidate counts, stop-list drops, aggregated/output row
    counts.
  - `ner/ner.py` — GLiNER load (model_id, device, elapsed_s); contraindication multi-GPU:
    work items, worker/shard counts, per-shard completion, total mentions.
  - `tablassert.py` — generate: per-config written (path, blake3) + fullmap choice; run:
    mode/command one-stat-per-line, **live subprocess streaming** via a new monkeypatchable
    `stream_subprocess(command, cwd)` (Popen + two reader threads, line filter as in Decision
    3, each surviving line logged as `tablassert: <line>`, full output still accumulated for
    the handoff report), report status/exit_code/elapsed_s; deferred-mode reason.
  - `translator.py` + `runtime.write_build_summary` — per-table contract results (rows,
    missing columns), regression row_count/families_seen/per-violation lines, per-table
    summary (name, rows, blake3) + report path.
- Python reference extractors (`extract/spl_xml.py`, `extract/faers_ascii.py`,
  `extract/drugsfda_products.py`): convert their multi-stat summary lines to the same
  one-stat-per-line form for consistency (NOT on the Airflow path — Go is — but parity
  harnesses run them).

### Go (native Airflow bundle)

`slog.Default()` inside the extractors reaches the Airflow task log: in coordinator mode the
Go SDK installs its socket log handler as the default logger before user code runs
(verified in go-sdk v1.0.0-beta3 `pkg/execution/server.go:95-97`). Package-level
`slog.InfoContext(ctx, ...)` therefore needs no signature changes. (The socket handler
encodes each record as one JSON line — one record per stat keeps every line short and
readable, matching the observed raw-JSON rendering in existing Go task logs.)

- `internal/airflow/log.go` (new) — tiny helpers:
  `stat(ctx, event, key, value)` → `slog.InfoContext(ctx, "<event>: <key> = <value>")`,
  `statDebug(...)`, and `elapsed(start) string`.
- `cmd/dakp-bundle/main.go` — runExtract: config dump (workdir/profile/threads/limits/force
  one line each), upstream input listing (per input: uri, blake3, rows — one line each),
  start/done markers gain elapsed_s + total output rows.
- `extract_faers.go` — staged file count, quarters discovered (count; each at DEBUG),
  per-quarter INFO block (quarter, parsed_rows, cases, kept, superseded, elapsed_s — one line
  each; per-family row counts at DEBUG), merge totals (merged_rows, elapsed_s), per-output
  registration (name, rows, blake3, schema_fingerprint, warnings — one line each), summary.
- `extract_dailymed.go` — staged inputs, SPL file count, worker limit, parse totals
  (documents/sets/approvals/ingredients/sections/warnings/elapsed_s one line each),
  per-table write + register stats, summary.
- `extract_drugsfda.go` — staged inputs, zip members unpacked, discovered inputs (per key),
  per-table parse rows, per-output registration stats, summary.
- `store.go` Register — DEBUG stat lines (path, blake3, rows).

Bundle repack happens automatically: `uv run dakp up` builds + packs the bundle before the run.

## Files to modify

Python:
- `src/dakp_pipeline/logging_setup.py` (helpers + bridge fix)
- `src/dakp_pipeline/dags/dakp_build.py`
- `src/dakp_pipeline/sources/{dailymed,faers,drugsfda}.py`
- `src/dakp_pipeline/acquire.py`, `src/dakp_pipeline/ner/{model_cache,ner}.py`
- `src/dakp_pipeline/io/artifact_store.py`
- `src/dakp_pipeline/assertions/{approved_treats,observed_uses,contraindications,evidence}.py`
- `src/dakp_pipeline/tablassert.py`, `src/dakp_pipeline/translator.py`, `src/dakp_pipeline/runtime.py`
- `src/dakp_pipeline/extract/{spl_xml,faers_ascii,drugsfda_products}.py` (consistency)
- tests: `tests/unit/test_logging_setup_edge.py` + new coverage for helpers (100% branch gate)

Go:
- `go/internal/airflow/log.go` (new), `extract_faers.go`, `extract_dailymed.go`,
  `extract_drugsfda.go`, `store.go`
- `go/cmd/dakp-bundle/main.go`

## Reuse

- Existing `bind()` (`logging_setup.py`) for task-scoped context; new helpers live beside it.
- Airflow 3's own structlog root handler (already installed by the SDK in task processes) does
  the formatting once the bridge stops clobbering it.
- Go SDK default slog logger (socket handler) — no new transport needed.

## Steps

- [ ] 1. `logging_setup.py`: `stats()` + `step()` + `progress()` helpers + tests
- [ ] 2. `logging_setup.py`: Airflow-3 bridge fix (real levels, sentinel, no root-handler
      clobber, no duplicate stderr sink) + update `test_logging_setup_edge.py` (current tests
      pin the old broken behavior: `airflow.task` propagate=False assertions, etc.)
- [ ] 3. Acquisition narration: `sources/*`, `acquire.py`, `ner/model_cache.py`
- [ ] 4. `io/artifact_store.py` DEBUG ingest/register lines
- [ ] 5. Assertion shaping narration (+ NER load + multi-GPU progress)
- [ ] 6. Tablassert generate/run + `stream_subprocess` live streaming (update the 8
      `run_subprocess` monkeypatch tests in `test_tablassert_configs.py` to the new seam) +
      translator + build summary
- [ ] 7. DAG task wrappers start/finish/elapsed
- [ ] 8. Python reference extractors consistency pass (summary lines → one stat per line)
- [ ] 9. Go: `log.go` helper + FAERS narration
- [ ] 10. Go: DailyMed + Drugs@FDA narration + `store.go` DEBUG + bundle main listing
- [ ] 11. Verification run (`uv run dakp up --small`) + inspect task logs

## Verification

- `uv run pytest` — full suite green, 100% branch coverage gate holds (`fail_under = 100`).
- `cd go && go build ./... && go vet ./... && go test ./... && gofmt -l .` (empty).
- Real end-to-end: `uv run dakp up --small` (bounded: 1 FAERS quarter + 1 DailyMed release;
  bundle is rebuilt/repacked automatically), then inspect the fresh task logs under
  `tmp/airflow-home/logs/dag_id=dakp_build/<run>/task_id=*/attempt=1.log`:
  - Python task records appear as structured records at the correct level (no more
    ERROR-level `task.stderr` double-stamped lines; no dropped stats).
  - Every stat is its own line (`rows`, `blake3`, `elapsed_s`, ...).
  - Go extract tasks narrate each phase with one-stat-per-line.
  - Tablassert stage lines (`Stage N of M`, completion rows) appear live in `run_tablassert`.
- Also spot-check `<workdir>/logs/dakp.log` still captures the run for offline reading.
