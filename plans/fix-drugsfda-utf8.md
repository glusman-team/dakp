# Drugs@FDA extract: speed is legit — but outputs contain invalid UTF-8 (spec-violating parquet)

## Context

User observation: `extract_drugsfda` finishes "super fast" — suspicious for a production extract.

### Verdict on the speed: **no bug** (verified against the real 2026-08-06 run)

- Input: one real FDA data-files zip (6 MB compressed / 43 MB uncompressed, FDA-published 2026-08-04),
  acquired fresh the same minute. Extract wall time: **0.91 s** (`extract start` 13:40:00.875 →
  `extract done` 13:40:01.787, `tmp/airflow-home/logs/.../task_id=extract_drugsfda`).
- Row counts **exactly match the source files** — nothing skipped:
  - `Products.txt` 51,622 data rows → `products.parquet` 51,622 ✓
  - `Applications.txt` 29,256 → `applications.parquet` 29,256 ✓
  - `Submissions.txt` 193,261 → `submissions.parquet` 193,261 ✓ (manifest `table.rows`)
- All 6 output refs registered (products parquet + public TSV, applications, submissions, lookups,
  warnings). Real bytes on disk: `submissions.parquet` 19 MB, `products.parquet` 7 MB, TSV 8 MB.
- The stage is the native Go bundle (`go/internal/airflow/extract_drugsfda.go`); ~15 MB TSV /
  ~274 k rows in <1 s is expected for Go. Sibling stages same run: FAERS 9.8 s (37 quarters),
  DailyMed 79 s (14,988 XML inputs) — drugsfda is simply the smallest input.

### But the investigation surfaced a **real latent bug**: invalid UTF-8 in the outputs

The live FDA `Submissions.txt` contains **18 rows with Windows-1252 bytes** in
`SubmissionsPublicNotes` (e.g. `Men\x92s Rogaine` = Men’s, `Approval \x96 March 23` = en-dash).

- Go `drugsfda.ParseTSVReader` (`go/internal/drugsfda/drugsfda.go`) reads bytes as-is (no encoding
  handling), and `airflow.WriteStringParquet` (`go/internal/airflow/parquet.go`) writes them
  unvalidated into STRING columns → **`submissions.parquet` violates the Parquet spec**
  (BYTE_ARRAY + STRING must be valid UTF-8).
- Consequence today: `polars.read_parquet(.../submissions.parquet)` fails with
  `ComputeError: parquet: File out of specification: String data contained invalid UTF-8`.
  The build still goes green only because **nothing downstream reads `submissions.parquet` yet**
  (`build_drugsfda_ingredient_map` in `assertions/evidence.py` reads `products.parquet` only,
  which is clean ASCII). Any future consumer (or a simple verification read) breaks.
- Parity gap: the Python reference extractor would **crash** on the real zip
  (`_read_tsv_bytes` → `pl.read_csv` defaults to strict UTF-8). Go↔Python parity tests only use
  clean ASCII fixtures (`go/internal/drugsfda/testdata/*.tsv`), so neither path was ever exercised
  against dirty real-world bytes.

## Approach

**Windows-1252 fallback decode at parse time, applied identically in Go and Python.**

While reading each table, pass valid UTF-8 through unchanged and decode any invalid byte run as
Windows-1252 (the encoding the FDA feed actually uses; cp1252 == Latin-1 except 0x80–0x9F, which
hold the `’ “ ” – — •` punctuation seen in the data). The 5 cp1252-undefined bytes
(0x81, 0x8D, 0x8F, 0x90, 0x9D) map to U+FFFD. Chosen over plain U+FFFD replacement because the
affected bytes are meaningful punctuation in labels/notes; over fail-fast because the FDA feed is
upstream and unpurifiable.

Fixing it in the shared parsers (not the parquet writer) means every downstream artifact
(parquet **and** the Tablassert TSV handoff) is clean, and both production entry points — the
Airflow bundle (`ExtractDrugsFDA`) and the `dakp-worker drugsfda` CLI — share
`internal/drugsfda.ParseTSV*`, so one Go fix covers both. The Python mirror gets the same decoder
so byte-for-byte parity holds on real data too.

Determinism note: outputs for inputs already valid UTF-8 are **byte-identical** before/after, so
all existing goldens (`go/internal/drugsfda/testdata/golden/*`, computed with the Python
reference) stay untouched.

## Files to modify

- `go/internal/drugsfda/drugsfda.go` — add `toValidUTF8(string) string` (valid runs pass through;
  invalid bytes → cp1252 rune via `golang.org/x/text/encoding/charmap` `Windows1252`, undefined →
  U+FFFD); apply to header + every cell in `ParseTSVReader`. Promote `golang.org/x/text` from
  indirect to direct in `go/go.mod` (`go mod tidy`).
- `src/dakp_pipeline/extract/drugsfda_products.py` — same algorithm in `_read_tsv_bytes`: decode
  the raw bytes with the UTF-8/cp1252-fallback scheme first, hand polars a `StringIO` of valid
  UTF-8 (no new dependency; small lookup for 0x80–0x9F).
- Tests (new coverage; existing goldens untouched):
  - `go/internal/drugsfda/drugsfda_test.go` — unit tests for `toValidUTF8` and `ParseTSVReader`
    on cells containing `\x92`, `\x96`, a valid multibyte char, and an undefined cp1252 byte.
  - `go/internal/airflow/extract_drugsfda_test.go` — end-to-end: stage a temp-dir submissions
    fixture containing cp1252 bytes, run `ExtractDrugsFDA`, assert the resulting submissions
    parquet reads back (strict) with the decoded `’`/`–` characters. Do **not** drop the fixture
    into `testdata/*.tsv` (the glob-based golden test would pick it up and shift the submissions
    table — last-file-wins by sorted path).
  - `tests/unit/test_drugsfda_extract.py` — Python mirror: fixture bytes with `\x92`/`\x96` →
    extracted parquet + TSV contain `’`/`–`, and `pl.read_parquet` round-trips.
  - One parity test asserting Go and Python produce identical output for the dirty fixture
    (fits the existing parity-test style in `go/internal/drugsfda/drugsfda_test.go`).

## Reuse

- `golang.org/x/text` (already in `go/go.sum` as indirect) → `charmap.Windows1252`; no new dep.
- Existing parity harness: `go/internal/drugsfda/testdata/golden/*` + the glob-driven tests in
  `drugsfda_test.go` / `extract_drugsfda_test.go` already prove byte-parity; the new dirty-fixture
  tests slot into the same pattern.
- `utf8.ValidString` / `utf8.DecodeRuneInString` (stdlib) for the fast path.

## Steps

- [ ] Go: implement `toValidUTF8` in `internal/drugsfda/drugsfda.go`; apply in `ParseTSVReader`
      (header + rows); `go mod tidy` to promote `golang.org/x/text`.
- [ ] Python: implement the identical decoder in `_read_tsv_bytes`
      (`extract/drugsfda_products.py`), feeding polars a UTF-8-clean `StringIO`.
- [ ] Add Go unit tests (`toValidUTF8`, `ParseTSVReader` dirty cells) and a targeted
      `ExtractDrugsFDA` test with a temp-dir cp1252 submissions fixture + strict parquet read-back.
- [ ] Add Python unit tests for the dirty fixture (parquet + TSV contents, polars round-trip) and
      the cross-language parity assertion.
- [ ] Run full Go + Python suites to confirm existing goldens/behavior are unchanged.

## Verification

1. `cd go && go test ./...` and `uv run pytest tests/unit/test_drugsfda_extract.py
   tests/unit/test_drugsfda_products_edge.py tests/integration -k drugsfda` — all green; existing
   golden tests unchanged.
2. Re-run the real build (or just the extract stage) against the live FDA zip:
   `uv run python -c "import polars as pl; print(pl.read_parquet('<workdir>/data/interim/drugsfda/submissions.parquet').height)"`
   → must load and print **193261**.
3. Spot-check the 18 formerly-dirty rows: `submission_notes` contains `’` / `–`
   (e.g. "Label for Men’s Rogaine"); row counts for all four tables unchanged
   (51,622 / 29,256 / 193,261 / lookups).
4. Byte check: `grep -c $'\x92'` on the emitted `drugsfda_products.tsv`-style outputs is 0 for any
   table (no raw cp1252 bytes survive into artifacts).

## Out of scope (observed, not touched unless requested)

- FAERS / DailyMed parsers have no encoding handling either (no evidence of dirty bytes there yet;
  FAERS ASCII fields observed clean). Same one-line decoder could be reused later.
- Go `ExtractDrugsFDA` writes an always-empty `extract_warnings.jsonl` (warnings go to the log
  stream) while the Python reference writes real warning records — a pre-existing parity gap,
  separate concern.
