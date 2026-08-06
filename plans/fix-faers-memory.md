# Fix: FAERS steps OOM — stream the extract, bound memory under 50 GB

## Context

A full production build (`dakp up`, no `--small`) OOMs on a 188 GB machine somewhere in the
FAERS path: the **`extract_faers`** task (native Go bundle worker — "Extract Faers") or
**`shape_faers_use_tables`** ("Process Faers"). Target: production build stays **< 50 GB** RSS.

### Production data volume (measured against the live FDA index, 2026-08-06)

- The FDA index lists **37 quarters** (`faers_ascii_2012q4.zip` … `faers_ascii_2026q2.zip`;
  2019–2021 are absent from the index). This is what an unbounded run acquires.
- Newest quarter zip = 63 MB compressed → **~335 MB ASCII** (DEMO 59M, DRUG 151M, INDI 53M,
  REAC 53M, THER 12M, OUTC 7M, RPSR 0.3M, DELETE 0.04M).
- Full corpus ≈ **8–12 GB raw ASCII**; the case join yields roughly **15–25 M case rows**.

### Why `extract_faers` (Go) OOMs — several overlapping full-corpus copies

Production path: DAG stub `extract_faers` → `cmd/dakp-bundle` → `airflow.ExtractFAERS`
(`go/internal/airflow/extract_faers.go`) → `faers.Extract` (`go/internal/faers/faers.go`).

1. `loadFAERSSources` reads **every file of every quarter fully into memory** (`os.ReadFile` /
   `io.ReadAll` of each zip member) → 8–12 GB of raw `[]byte` held for the whole run.
2. `ParseSourcesConcurrent` runs with `limit = cfg.Threads` = **`os.cpu_count()`** (64+ on the
   prod box), so up to 64 files parse at once; every parsed `[][]string` table (2–3× raw size)
   for **all 37 quarters** is held simultaneously in `byQF`, and returned as `Result.Normalized`
   — which `ExtractFAERS` **never writes and never frees** (kept alive to function end).
3. All per-quarter case joins accumulate in `perQuarter [][]Case` (17 string fields ≈ 272 B/row
   + string data) for all quarters at once.
4. `ReduceCases` concats **all** cases into one slice, builds a `winning` map over every dedup
   key, then splits into `kept` + `superseded` + audit — 3–4 more full-case-set copies.
5. `ExtractFAERS` then builds `caseRows [][]string` — yet another full copy — for the parquet
   write, while everything above is still live.

Peak ≈ raw corpus + parsed corpus + ~5 copies of the case set ≫ 188 GB. This is the OOM.

### Secondary: `shape_faers_use_tables` (Python "Process Faers")

`find_faers_cases` (`assertions/evidence.py`) eagerly reads the **entire** global
`cases.parquet` (all 17 columns), then `build_observed_use_rows` (`assertions/observed_uses.py`)
iterates 15–25 M rows via `iter_rows(named=True)` accumulating a Python
`dict[tuple, set[str]]` of (drug, indication) → primaryids. Not the primary OOM cause
(~10–15 GB peak), but it takes hours and is the other FAERS-heavy step. Fix it too.

### Tertiary: FAERS quarters are re-downloaded every run

Confirmed: `FAERSFetcher.download_quarter` (`sources/faers.py`) calls `_http_download`
**unconditionally** for all 37 zips (~2.5 GB) on every run. `store.ingest` only deduplicates
the *stored copy* (`cache_hit`) — the network transfer already happened. Contrast:

- DailyMed (`sources/dailymed.py`) uses conditional GET (`If-None-Match`/`If-Modified-Since` →
  304 → reuse via alias lookup) and streams downloads in 1 MiB chunks.
- The NER model cache (`ner/model_cache.py`) skips the download entirely on a manifest/hash hit,
  honoring `ctx.params["force"]` to re-download — **the established skip-download pattern**.
- FAERS ignores the existing `force` param, never records ETag/Last-Modified, reads each whole
  zip into memory (`response.read()`), and never removes the staged `dest` file.
- Verified against the live server (2026-08-06): `fis.fda.gov` sends **no `ETag` and no
  `Last-Modified`** (only `Cache-Control: private, must-revalidate, max-age=0`), so DailyMed-style
  conditional GET is impossible there; alias-based content reuse is the right mechanism (FAERS
  quarterly zips are immutable published snapshots). Drugs@FDA re-downloads too, but it is one
  small zip — note only, not part of this fix.

## Approach

### A. Go `extract_faers`: quarter-streaming pipeline with external merge (primary fix)

Keep all parsing/join/dedup **semantics** and the output contract (same 5 ArtifactRefs, same
names/order, byte-identical TSV, deterministic parquet) — change only the orchestration so at
most **one quarter's data** is in memory at a time, and the global sorted output is produced by
a k-way merge of per-quarter sorted run files.

New flow in `airflow.ExtractFAERS`:

1. **Lazy source inventory** (no bulk content reads): walk staged inputs; for each loose `.txt`
   and each `.txt` member of each quarterly zip, record a handle
   `{quarter, family, name, open func() (io.ReadCloser, error)}`. One zip == one quarter; keep a
   `zip.ReadCloser` open only while its quarter is processed, then close.
2. **Process quarters sequentially, most-recent-first** (the existing order). For each quarter:
   a. Parse its ≤6 family files **concurrently** (errgroup, bounded by file count — NOT
      `cfg.Threads` across the corpus) via a new streaming parser
      `faers.ParseStream(r io.Reader, quarter, family, name, warn) *Table` that hashes content
      through a tee'd BLAKE3 as it reads (same `b3:<hex>` as today's `HashBytes(content)`, so
      `source_record_id`s are unchanged). `faers.ParseSource` delegates to it (tests unchanged).
      Per-quarter tables ≈ ~1 GB peak, freed after the quarter.
   b. `faers.BuildQuarterCases` unchanged (DELETE-filtered, intra-quarter deduped, sorted by
      `caseSortKey`).
   c. **Streaming cross-quarter dedup** (replaces `ReduceCases` for the production path):
      maintain `seen map[key]newestQuarter` (key = caseid else primaryid, ~1–2 GB total). Rows
      whose key is already in `seen` are superseded → append `DedupAuditRow{…, winning: seen[key]}`;
      other rows are kept. New keys are inserted **after** the quarter finishes (rows of the same
      quarter never supersede each other — exactly matches Python semantics where all rows of the
      winning quarter survive).
   d. Write the quarter's kept rows to a **scratch parquet run file** with the 11 public columns
      **+ `drug_seq`** (needed for merge ordering, dropped from outputs — today's `cases.parquet`
      already emits it empty). Runs are sorted (inherited from `BuildQuarterCases`).
   e. Drop all references to the quarter's tables/cases (`runtime.GC()` at the boundary is fine).
3. **K-way heap merge** of the run files ordered by `(primaryid, drug_seq, indication,
   run-index)` — run-index tiebreak = newer quarter first, which exactly reproduces
   `sort.SliceStable` over the current most-recent-first concat (=> **byte-identical**
   `faers_cases.tsv` vs. the golden/Python order). Stream the merged rows once into both:
   - `cases.parquet` via a small streaming variant of `WriteStringParquet` (append row-by-row,
     same 17-column layout with empty provenance columns as today), and
   - `faers_cases.tsv` via the existing `WriteCasesTSV` row logic (refactored to accept a row
     iterator so it never holds all cases).
4. **Audits & warnings unchanged**: `delete_audit` from the (tiny) DELETE tables; dedup audit
   sorted in memory by `(quarter, primaryid)` (bounded by superseded rows, ≪ kept rows);
   `warnings.parquet` empty as today. Register the same 5 refs in the same order.
5. **Safety net**: if `cfg.MemoryBudgetGB > 0`, call `debug.SetMemoryLimit` in the FAERS task
   (GC soft cap; the structural fix is what actually bounds memory).

Expected peak: per-quarter tables (~1–1.5 GB) + one quarter's cases (~1–2 GB) + `seen` map
(~1–2 GB) + dedup-audit buffer + parquet/merge buffers (~1 GB) ≈ **~8–15 GB** (was ≫ 188 GB).
Scratch disk: ~2–5 GB run files, removed by the existing `defer os.RemoveAll(scratch)`.

`faers.Extract` / `ReduceCases` / `BuildQuarterCases` stay as-is (unit tests + the dev-only
`cmd/dakp-worker faers` CLI keep using the batch path); the bundle task switches to the new
streaming orchestration.

#### Key sketches (Go)

Lazy source handles replace the bulk `[]byte` loads:

```go
// internal/airflow — one FAERS ASCII file, opened on demand (no content buffered up front).
type faersSourceHandle struct {
    Quarter, Family, Name string
    Open func() (io.ReadCloser, error) // loose .txt: os.Open; zip member: zipFile.Open()
}
// inventoryFAERSSources(inDir) -> map[quarter][]handle, newest-first quarter order.
// A zip.ReadCloser stays open only while its own quarter is parsed, then Close().
```

Streaming parser with tee'd hashing (`source_record_id` bytes unchanged):

```go
// internal/faers — ParseSource delegates here; tests keep working.
func ParseStream(r io.Reader, quarter, family, sourceName string, warn *Warnings) *Table {
    h := blake3.New()
    br := bufio.NewReaderSize(io.TeeReader(r, h), 1<<20)
    // ... same header/line/field logic as today's ParseSource ...
    short := b3Short(blake3store.IDPrefix + hex.EncodeToString(h.Sum(nil))) // == HashBytes(content) prefix
    // ... prepend quarter/source_file/source_record_id provenance exactly as today ...
}
```

Quarter loop + streaming dedup (replaces the all-in-memory `ReduceCases` path):

```go
deduper := &caseDeduper{seen: map[string]string{}} // dedup key -> newest quarter that owns it
var runs []string                                  // per-quarter kept-run parquet paths (newest-first)
var dedupAudit []faers.DedupAuditRow
var deleteTables []*faers.Table

for _, q := range quarters { // most-recent-first
    families := parseQuarter(ctx, handles[q], warn) // <=6 files concurrently, ~1 GB peak
    if del := families["DELETE"]; del != nil && len(del.Rows) > 0 {
        deleteTables = append(deleteTables, del)
    }
    cases := faers.BuildQuarterCases(families, q, faers.DeletedPrimaryIDs(families["DELETE"]), warn)

    kept := cases[:0:0]
    var newKeys []string
    for _, c := range cases {
        k := c.DedupKey() // export today's unexported dedupKey() — the deduper lives in internal/airflow
        if w, ok := deduper.seen[k]; ok { // a NEWER quarter already owns this caseid
            dedupAudit = append(dedupAudit, faers.DedupAuditRow{Quarter: q, PrimaryID: c.PrimaryID,
                CaseID: c.CaseID, DedupKey: k, WinningQuarter: w, SourceFile: c.SourceFile})
            continue
        }
        kept = append(kept, c)
        newKeys = append(newKeys, k)
    }
    for _, k := range newKeys { deduper.seen[k] = q } // insert AFTER the quarter: same-quarter
    // rows never supersede each other (== Python: all rows of the winning quarter survive)

    runs = append(runs, writeCaseRun(scratch, q, kept)) // 11 public cols + drug_seq, sorted
    families, cases, kept = nil, nil, nil               // quarter memory reclaimable now
}
```

K-way merge emitting both outputs in one streaming pass:

```go
// Runs are sorted by (primaryid, drug_seq, indication); tie-break toward the NEWER run
// (lower index) == sort.SliceStable over the most-recent-first concat => byte-identical output.
pq := heap.New(runCursors, less /* primaryid, drugSeq, indication, runIdx */)
pqw := airflow.NewStringParquetWriter(casesPath, faersCaseColumns) // 17 cols, provenance ""
tsv := bufio.NewWriterSize(tsvFile, 1<<20)                          // header = CasesTSVColumns
for pq.Len() > 0 {
    row := popNext(pq)               // one merged Case row
    pqw.Append(caseRow17(row))       // same empty-provenance layout as today
    writeTSVRow(tsv, row)            // same 11-column projection as WriteCasesTSV
}
```

Streaming parquet writer helper + memory safety net:

```go
// internal/airflow/parquet.go — reuse the existing schema/leaf-index logic.
type StringParquetWriter struct{ /* *parquet.Writer, cols, leafIdx, n */ }
func NewStringParquetWriter(path string, columns []string) (*StringParquetWriter, error)
func (w *StringParquetWriter) Append(row []string) error
func (w *StringParquetWriter) Close() (rows int, err error)
// WriteStringParquet becomes a thin Append loop (dailymed/drigsfda paths unchanged).

// in ExtractFAERS, before starting:
if cfg.MemoryBudgetGB > 0 { debug.SetMemoryLimit(int64(cfg.MemoryBudgetGB) << 30) }
```

### B. Python `shape_faers_use_tables`: aggregate in Polars, not Python rows

In `observed_uses.build_observed_use_rows` (keep the exact signature `(pl.DataFrame | None,
disease_map) -> list[dict[str, str]]` so unit tests keep working):

- Replace the `iter_rows(named=True)` + `pair_cases` dict with a lazy Polars aggregation:
  strip + filter empty `drugname`/`indication`, then
  `group_by("drugname", "indication").agg(n_unique(non-empty primaryid), count(empty primaryid))`;
  `case_count = uniques + anon`. This is exactly equivalent to today's semantics: distinct-case
  counting with the `_row{index}` per-row fallback for missing/empty primaryid, and row-count
  when the column is absent.
- Iterate only the aggregated pairs (millions, not tens of millions) for the
  `is_non_disease_indication` filter, `match_diseases`, and `row_for`; output rows and ordering
  (sorted by drug, indication) stay identical.
- `find_faers_cases`: add an optional `columns` projection and have observed-uses read only
  `drugname`/`indication`/`primaryid` (schema-peek first; missing-column behavior stays `None`).
  Cuts the eager frame ~5×.

```python
# observed_uses.build_observed_use_rows — same signature, same output rows/order.
has_primaryid = "primaryid" in faers_cases.columns
pairs = (
    faers_cases.lazy()
    .select(
        pl.col("drugname").fill_null("").str.strip_chars().alias("drugname"),
        pl.col("indication").fill_null("").str.strip_chars().alias("indication"),
        (pl.col("primaryid").fill_null("").str.strip_chars() if has_primaryid else pl.lit("")).alias("primaryid"),
    )
    .filter((pl.col("drugname") != "") & (pl.col("indication") != ""))
    .group_by("drugname", "indication")
    .agg(
        pl.col("primaryid").filter(pl.col("primaryid") != "").n_unique().alias("distinct_cases"),
        pl.col("primaryid").filter(pl.col("primaryid") == "").len().alias("anon_rows"),
    )
    .collect()
    .sort("drugname", "indication")
)
# case_count = distinct_cases + anon_rows  (== legacy set-of-ids + _row{index} fallback);
# iterate aggregated pairs only: is_non_disease_indication -> match_diseases -> row_for.
```

### C. FAERS acquisition: stop re-downloading cached quarters

In `sources/faers.py` `download_quarter`, check the content-addressed store BEFORE any network
call (the `acquire_ner_models` pattern), honoring the existing `force` param:

```python
def download_quarter(self, ctx: TaskContext, source: QuarterSource) -> ArtifactRef:
    wd = Workdir(ctx.workdir)
    store = ArtifactStore(wd)
    alias = f"faers/faers_ascii_{source.quarter}.zip"
    if not bool(ctx.params.get("force", False)):
        cached = store.cached_ref(alias)  # alias + .path pointer lookup (see below)
        if cached is not None and cached.uri.exists():
            logger.info("faers quarter cached", quarter=source.quarter, artifact_id=cached.blake3)
            return cached
    dest = wd.raw / "downloads" / f"faers_ascii_{source.quarter}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _http_download_stream(source.url, dest, timeout=_DEFAULT_TIMEOUT)  # shutil.copyfileobj, 1 MiB chunks
    ref, cache_hit = store.ingest(dest, alias=alias, source=SourceBlock(url=source.url, retrieved_at=_now()))
    dest.unlink(missing_ok=True)  # staged copy no longer needed once content-addressed
    return ref
```

Supporting changes:

- Promote DailyMed's private `_cached_ref` alias lookup to
  `ArtifactStore.cached_ref(alias) -> ArtifactRef | None` (`io/artifact_store.py`); DailyMed
  switches to the shared method (behavior identical).
- Replace `_http_download`'s whole-file `response.read()` with a streaming
  `shutil.copyfileobj(resp, handle, length=1 << 20)` (DailyMed's `_CHUNK` pattern) — a 63–160 MB
  zip no longer sits in Python memory.
- Rationale (verified 2026-08-06): `fis.fda.gov` sends no ETag/Last-Modified, so conditional GET
  is impossible; FAERS quarterly zips are immutable published snapshots, so alias/content reuse is
  safe. `force` (already plumbed through `dakp_config`) re-downloads everything.
- First run still downloads all 37 zips; every later run downloads only newly published quarters
  (~1 zip/run) instead of ~2.5 GB.

### Out of scope (flagged)

- `extract_dailymed` / `extract_drugsfda` share the "hold everything" style — much smaller
  inputs, not OOMing today; revisit separately if needed.
- The Python reference extractor (`extract/faers_ascii.py`) is test-only now; untouched.

## Files to modify

- `go/internal/faers/faers.go` — add `ParseStream` (ParseSource delegates), add streaming
  dedup/merge primitives (e.g. `CaseRun` writer/reader + `MergeCaseRuns` + a `Deduper` holding
  the `seen` map); `ReduceCases`/`BuildQuarterCases`/`Extract` unchanged.
- `go/internal/airflow/extract_faers.go` — rewrite `ExtractFAERS` orchestration: lazy handles
  (loose + zip), quarter loop, scratch run files, merge-emitting parquet+TSV, same refs.
- `go/internal/airflow/parquet.go` — small streaming writer helper (open → append rows → close)
  reusing the existing schema/leaf-index logic; row-reading helper for run files.
- `src/dakp_pipeline/assertions/observed_uses.py` — Polars aggregation rewrite.
- `src/dakp_pipeline/assertions/evidence.py` — optional column projection in `find_faers_cases`.
- `src/dakp_pipeline/sources/faers.py` — cached-quarter skip + `force`, streaming download,
  staged-file cleanup.
- `src/dakp_pipeline/io/artifact_store.py` — new `cached_ref(alias)` (DailyMed's `_cached_ref`
  promoted); `src/dakp_pipeline/sources/dailymed.py` switches to it.
- Tests: `go/internal/faers/faers_test.go`, `go/internal/airflow/extract_faers_test.go`,
  `tests/unit/test_assertions_observed_uses.py`, `tests/unit/test_assertions_evidence*.py`
  (100% branch coverage gate applies to the Python side).

## Reuse

- `faers.BuildQuarterCases`, `faers.DeletedPrimaryIDs`, `faers.FamilyAndQuarter`,
  `sortCases`/`dedupKey`/`dedupSubsetKey` (`go/internal/faers/faers.go`) — unchanged building blocks.
- `airflow.WriteStringParquet` schema/leaf mapping (`go/internal/airflow/parquet.go`).
- `airflow.StageInputs`, `Store.Register` (`go/internal/airflow/stage.go`, `store.go`).
- `blake3store.HashBytes` semantics via `blake3` streaming hash (`go/internal/blake3store`).
- `schemas.FAERS_CASES_COLUMNS`, `schemas.read_table` (`src/dakp_pipeline/io/schemas.py`).
- DailyMed's `_cached_ref` / `_CHUNK` streaming-download patterns (`sources/dailymed.py`) and the
  NER model-cache skip pattern (`ner/model_cache.py`) for the acquisition fix.

## Steps

- [x] **Go streaming parser**: `faers.ParseStream(io.Reader, …)` with tee'd BLAKE3 hashing;
      `ParseSource` delegates; existing parser tests green.
- [x] **Go streaming orchestration pieces**: per-quarter kept-run writer (11 public cols +
      `drug_seq`), run reader, k-way heap merge with quarter tiebreak, streaming deduper
      (`seen map[key]quarter`, insert-at-quarter-end); unit test **streaming-vs-batch byte parity**
      on the committed testdata (TSV bytes + parquet rows + audits).
- [x] **Rewrite `airflow.ExtractFAERS`**: lazy inventory (zip members streamed, `zip.ReadCloser`
      lifetime = one quarter), quarter loop (newest-first, families parsed concurrently),
      merge-emit `cases.parquet` + `faers_cases.tsv` in one pass, audits/warnings/refs unchanged;
      optional `debug.SetMemoryLimit(cfg.MemoryBudgetGB)`; `TestExtractFAERSParity` + zip-loading
      tests adapted/green.
- [x] **Python observed-uses**: Polars group-by aggregation replacing `iter_rows`; optional
      column projection in `find_faers_cases`; keep all existing unit tests green + add edge
      coverage for the new branches (100% branch gate).
- [x] **FAERS acquisition caching**: `ArtifactStore.cached_ref(alias)`; `download_quarter`
      skip-on-cache honoring `force`; streaming download + staged-file cleanup; DailyMed switched
      to the shared helper; unit tests for hit/miss/force/missing-file branches.
- [x] **Scale verification**: run the streaming extractor standalone over real downloaded quarter
      zips (start with the newest 3–5 quarters; cross-check byte parity vs. the batch path on the
      same subset), measure peak RSS (`/usr/bin/time -v`), confirm ≪ 50 GB and linear scaling.
- [ ] Local end-to-end: `uv run dakp up --small`; then the full build on the 188 GB box.

## Measured results (real FDA data, dev laptop)

- **Streaming vs batch byte parity** (newest 2 quarters, 2,044,738 cases): the 470,623,372-byte
  `faers_cases.tsv` is identical between the streaming path and the legacy all-in-memory path.
- **Peak RSS** (6 quarters 24Q1-26Q2, 5,808,374 cases, 1.32 GB TSV): **6.1 GB**
  (`Maximum resident set size`), ~60 s wall, live heap 0.04 GB after the run. Target was < 50 GB;
  the prior all-in-memory path held raw + parsed + ~5 copies of the case set (>> 188 GB at full
  scale, hence the OOM).
- **Determinism**: two back-to-back streaming runs produce byte-identical `faers_cases.tsv`
  (same SHA-256).
- Scale harness: `go test -tags scale ./internal/airflow/ -run TestExtractFAERSScale...` (skips
  unless `FAERS_SCALE_DIR` / `FAERS_SCALE_BATCH_DIR` point at real quarter zips).

## Verification

1. `cd go && go build ./... && go vet ./... && go test ./... && gofmt -l .` — all parity
   goldens (`TestExtractParityWithPythonTSV`, `TestExtractFAERSParity`, determinism) unchanged.
2. `uv run pytest -q --cov` (100% branch gate), `uv run ruff check`,
   `uv run ruff format --check`, `uv run pyright`.
3. Standalone scale run on real FAERS zips: peak RSS reported (target < 15 GB, hard requirement
   < 50 GB), TSV byte-identical to batch output on the same subset, deterministic across reruns.
4. Acquisition cache: run `acquire_faers` twice against a workdir — second run logs
   `cache_hit`/`cached` for every quarter and performs zero zip downloads (verify via logs /
   network counters); `force=True` re-downloads.
5. `uv run dakp up --small` end-to-end green locally; full `dakp up` on the production machine
   completes `extract_faers` + `shape_faers_use_tables` without OOM.
