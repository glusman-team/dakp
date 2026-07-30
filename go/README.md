# DAKP Go workers

Native Go worker foundation for the DAKP pipeline rebuild (see `PLAN.md`, "Go worker
plan"). Heavy parsing/extraction runs here as subcommands of a single `dakp-worker` CLI;
Airflow tasks stay thin Python orchestrators that invoke this binary and stream its
output. This module is the **shared foundation** — content addressing, the artifact
manifest, shared pipeline types, and the CLI dispatcher — that the per-source extractors
(FAERS, DailyMed, Drugs@FDA, MEDI — a later milestone) plug into.

Module path: `github.com/glusman-team/dakp/go`. All commands below run from this `go/`
directory.

## Layout

```text
go/
  go.mod / go.sum
  cmd/dakp-worker/
    main.go        # STABLE entrypoint: calls registry.Main(os.Args). Never edit per-extractor.
    hash.go        # "hash" subcommand — self-registered via init(). The pattern to copy.
    hash_test.go
  internal/
    registry/      # self-registration command dispatcher (Register/Dispatch/Run/Main)
    blake3store/   # BLAKE3 file + tree hashing, SHA-256 SRI, artifact manifests
    pipeline/      # ArtifactRef/TaskContext, SourceRecordID, InferMediaType
```

## Build, test, acceptance

```bash
go build ./...        # compiles everything
go vet ./...          # static checks
go test ./...         # all tests (includes Python-parity fixtures)
gofmt -l .            # must print nothing

# Run the worker directly:
go run ./cmd/dakp-worker hash <path>        # file -> content hash; dir -> tree hash
go run ./cmd/dakp-worker hash -mode=tree <dir>
go run ./cmd/dakp-worker help               # list registered subcommands

# Build a binary for full runs:
go build -o dakp-worker ./cmd/dakp-worker   # ignored by go/.gitignore
```

## The self-registration pattern (how to add a new extractor subcommand)

Subcommands register themselves from `init()` in their **own file** under
`cmd/dakp-worker/` (all `package main`). Because every file in a package contributes its
`init()`, a new subcommand needs **no edits to `main.go`, the registry, or any existing
file** — so multiple extractor workers can land in parallel and merge cleanly.

To add an extractor (e.g. `faers`), create exactly one new file
`cmd/dakp-worker/faers.go`:

```go
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/glusman-team/dakp/go/internal/registry"
	// ... pipeline / blake3store as needed
)

func init() {
	registry.Register("faers", func(ctx context.Context, args []string) error {
		return runFAERS(ctx, args, os.Stdout)
	})
}

func runFAERS(ctx context.Context, args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("faers", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // keep stdout clean for machine-readable output
	// ... parse flags, do the work, write the artifact id / manifest ...
	return nil
}
```

That's it. `dakp-worker faers ...` now works, and `dakp-worker help` lists it. Rules:

- **One file per subcommand**, `package main`, with an `init()` calling
  `registry.Register(name, fn)`.
- The command func receives `args` **after** the subcommand name (flags/positionals only).
- Keep **stdout** machine-readable (the artifact `b3:<hex>` id and/or the JSON manifest);
  send logs to **stderr** as structured JSON via `log/slog`.
- Return an error on failure; the registry maps it to a non-zero exit code.
- Duplicate names panic at startup (a programming error that should fail loudly).

`main.go` is deliberately frozen — it only calls `registry.Main(os.Args)`.

## How Python invokes the worker

Airflow tasks (and the Python `workers/go_runner.py` wrapper) shell out to the worker and
stream its output line-by-line into the Airflow task logger:

```bash
# development:
go run ./cmd/dakp-worker hash data/raw/by-hash/<hex>/faers_ascii_2024q3.zip
# full runs (prebuilt binary):
/path/to/dakp-worker faers --quarter 2024q3 --in <manifest.json> --out data/interim/faers
```

Contract:

- **stdout** — machine-readable results: the `b3:<hex>` artifact id and/or the artifact
  manifest JSON. Python captures this.
- **stderr** — structured JSON log lines (`log/slog`) with the shared schema fields
  (`task_id`, `shard_id`, `artifact_id`, `input_hash`, `output_hash`, `rows`,
  `elapsed_ms`, `cache_hit`, ...). Python relays these so Airflow shows Go and Python logs
  uniformly.
- **exit code** — `0` success, `1` command error, `2` usage error (no/unknown command).

## Content addressing (BLAKE3)

BLAKE3 is DAKP's primary content hash (`PLAN.md`, "Nix-store-inspired artifact and
cryptography model"). Canonical artifact ids are `b3:<hex>` with a **32-byte / 64-hex**
digest.

- **Library:** [`github.com/zeebo/blake3`](https://pkg.go.dev/github.com/zeebo/blake3)
  v0.2.4 — pure Go + SIMD, no cgo. `blake3.New()` defaults to a 32-byte output, matching
  the Python `blake3` package's default `hexdigest()`, so Python and Go agree.
- **File hash** (`blake3store.HashFile`): streaming BLAKE3 of the file bytes (1 MiB
  window).
- **Tree hash** (`blake3store.HashTree`): deterministic, Nix-NAR-like, **byte-for-byte
  identical to Python's `content_hash.hash_tree`**. The exact algorithm (mirror this if
  reimplementing):
  1. Collect every regular file under the root (recursive); skip symlinks/non-regular
     files and empty directories.
  2. Sort by **relative POSIX path** (forward slashes, lexicographic by byte value — equal
     to Unicode code-point order for valid UTF-8, the same order Python's `sorted()` uses).
  3. Feed **one** BLAKE3 hasher; for each file in order write:
     `relPath(utf-8) | 0x00 | size(decimal ASCII) | 0x00 | fileBytes | 0x00`.
  4. Emit `b3:` + hex of the final 32-byte digest.

  Directory mtimes, traversal order, and empty dirs do not affect the result; an empty
  directory hashes to BLAKE3 of the empty input.
- **SHA-256 SRI** (`blake3store.SHA256SRI`): optional `sha256-<base64>` interoperability
  metadata only — never the primary key.
- **Bounded parallel hashing** (`blake3store.HashFiles`): hash many files concurrently via
  `golang.org/x/sync/errgroup` with `SetLimit`, cancelling on first error — use this to
  respect Airflow task concurrency / memory budgets.

### Cross-language parity (tested)

`internal/blake3store/testdata/` holds golden fixtures computed with the **Python**
reference (`blake3` 1.0.9, `pydantic` 2.13.4); the Go tests assert byte-for-byte equality:

- `testdata/tree/` — a small nested directory; Go `HashTree` must equal Python
  `hash_tree` → `b3:3efcf1d2ac7f501dda31fb970875d3a8a2d59852d09f55cf562af3ba3d029fb6`.
- `testdata/manifest_full.json`, `testdata/manifest_minimal.json` — Python-written
  manifests; Go reads them and re-marshals to **identical bytes** (2-space indent, no HTML
  escaping, no trailing newline, `inputs: []`, nullable fields as `null`, matching field
  order).
- `internal/pipeline` `SourceRecordID` vectors match `spl_xml._source_record_id`
  (`b3:` + BLAKE3 of the `\x1f`-joined `[source_id, kind, *local_keys]`).

The integration milestone's parity tests can rely on: **zeebo/blake3 v0.2.4, 32-byte
output, and the tree-hash algorithm above.**

## Artifact manifests

`blake3store.ArtifactManifest` mirrors `src/dakp_pipeline/io/manifests.py`
(`schema_version: dakp.artifact.v1`): `artifact_id`, `path`, `media_type`, `hash`
(`algorithm`/`file`/`tree`/`sha256_sri`), `inputs`, `operation`, `source`, `environment`,
`table`. `ReadManifest` / `WriteManifest` round-trip Python- and Go-written files; the
marshal is byte-compatible with pydantic's `model_dump_json(indent=2)`.

## Dependencies

Direct (both genuinely used by the foundation):

- `github.com/zeebo/blake3` — BLAKE3 hashing.
- `golang.org/x/sync` — `errgroup` for bounded concurrency (`HashFiles`).

**Parquet — deferred (TODO).** The per-source extractors will emit partitioned interim
tables; the Tablassert-facing tables are uncompressed TSV. The foundation writes no tables
yet, and `go mod tidy` strips unused deps, so the parquet writer is intentionally **not**
added here. When an extractor needs parquet, add
[`github.com/parquet-go/parquet-go`](https://github.com/parquet-go/parquet-go) with a
single `go get github.com/parquet-go/parquet-go` (a one-line `go.mod` change) and write TSV
for the Tablassert handoff regardless.

## FAERS extractor subcommand (`faers`)

The first per-source extractor, ported from `src/dakp_pipeline/extract/faers_ascii.py`
(Milestone 3). It lives in `internal/faers/` (parser + case-join library) and
`cmd/dakp-worker/faers.go` (self-registered `faers` subcommand — a NEW file; `main.go` and
the registry are untouched).

```bash
# Parse a directory of FAERS ASCII .txt files (one or more quarters, derived from each
# filename like DEMO24Q3.txt) and write the uncompressed source-section TSVs:
go run ./cmd/dakp-worker faers <quarter-dir> <out-dir>
go run ./cmd/dakp-worker faers -jobs 8 tests/fixtures/pipeline/faers /tmp/faers-out
```

Behavior (faithful Go port of the Python, including the `listCases.pl` join semantics):

- Parses `$`-delimited ASCII per family (DEMO, DRUG, INDI, REAC, RPSR, DELETE) with
  bufio streaming line reads, handling the trailing `$`, CRLF, UPPERCASE→lowercase
  headers, and the legacy `isr`→`primaryid` column. Files parse concurrently via
  `errgroup` with `SetLimit` (`-jobs`; `<=0` = unbounded).
- Normalized tables carry `quarter`, `source_file`, `source_record_id` provenance first;
  `source_record_id` is `<first-12-hex-of-file-b3>:<primaryid>[:<drug_seq|indi_drug_seq:indi_pt|pt>]`.
- Builds the INDI-driven per-quarter case join (INDI×DRUG on `primaryid|drug_seq`),
  left-joining DEMO reporter metadata, RPSR source, and REAC reactions (sorted-unique,
  `$`-joined), honoring DELETEd primaryids and intra-quarter exact-row dedup.
- Reduces across quarters with caseid dedup (most-recent-wins; `caseid` key, falling back
  to `primaryid`). `nda` is digits-only with leading zeroes stripped (`nda_raw` keeps them).

Outputs (uncompressed TSV; parquet is deferred — see Dependencies above):

- `<out-dir>/faers_cases.tsv` — the public Tablassert source-section contract, columns
  `schemas.FAERS_CASES_COLUMNS`:
  `quarter, primaryid, caseid, source, occp_cod, reporter_country, drugname, ingredient, nda, indication, effects`.
- `<out-dir>/delete_audit.tsv` — `quarter, primaryid, caseid, source_file, source_record_id`.
- `<out-dir>/dedup_audit.tsv` — `quarter, primaryid, caseid, dedup_key, winning_quarter, source_file`.

Contract: **stdout** is the single `b3:<hex>` content id of `faers_cases.tsv`; **stderr**
is a structured JSON `log/slog` summary (`quarters`, `cases`, `deleted`, `deduped`,
`warnings`, `artifact_id`, `elapsed_ms`). `internal/faers/faers_test.go` asserts
byte-for-byte parity of `faers_cases.tsv` against the Python extractor using the
byte-identical fixtures in `internal/faers/testdata/`.
## Drugs@FDA extractor subcommand (`drugsfda`)

Self-registered subcommand (`cmd/dakp-worker/drugsfda.go`, `init()` → `registry.Register`,
same pattern as `hash.go`) backed by `internal/drugsfda`. Parses the Drugs@FDA
tab-delimited tables (`Products.txt` / `Applications.txt` / `Submissions.txt`, or fixture
mirrors like `drugsfda_products.tsv`) into normalized **products / applications /
submissions / lookups** tables and writes the uncompressed TSV source-section tables for
Tablassert handoff. It is the Go mirror of
`src/dakp_pipeline/extract/drugsfda_products.py` and is **byte-for-byte compatible** with
it: the golden fixtures in `internal/drugsfda/testdata/golden/` are computed with the
Python reference (polars 1.43.1), and `TestParityGoldenTSV` asserts identical bytes.

```bash
# development:
go run ./cmd/dakp-worker drugsfda <input-dir> <out-dir>
# e.g. go run ./cmd/dakp-worker drugsfda internal/drugsfda/testdata /tmp/out
```

- **`<input-dir>`** — directory of loose Drugs@FDA TSV/TXT tables, classified by filename
  (`drugsfda.Classify`: stems ending in `products`/`applications`/`submissions`, or the
  singular `product`/`application`/`submission`; sub-tables like `SubmissionPropertyType.txt`
  are ignored). Recognized files are parsed concurrently (`errgroup` + `SetLimit`,
  `-limit=N`, default 4).
- **`<out-dir>`** — receives `drugsfda_products.tsv`, `drugsfda_applications.tsv`,
  `drugsfda_submissions.tsv`, `drugsfda_lookups.tsv` (each written only when its source
  table is present; lookups derive from products).
- **stdout** — one JSON summary: per-output `path`, `artifact_id` (`b3:<hex>`), `rows`,
  `media_type`, `schema_fingerprint`, plus the input `b3:<hex>` hashes, warning count, and
  `elapsed_ms`. **stderr** — `log/slog` JSON logs (`task_id=extract_drugsfda_products`).

Application numbers keep the raw `APPLICATIONNUMBER` **and** both normalized forms
(`appl_no` with leading zeroes, `appl_no_stripped` without), porting the legacy
`readNDAproducts` `s/^(NDA|BLA|ANDA)0*(.+)/`. `source_record_id` uses the Drugs@FDA string
form (`drugsfda:product:NDA12345:001`, `drugsfda:application:NDA12345`,
`drugsfda:submission:NDA12345:1`) to match the Python reference exactly — not a `b3:` hash;
`b3:<hex>` is used for artifact/content ids. Submissions inherit `appl_type` from
products/applications (the real `Submissions.txt` carries none).

TSV columns (ordered):

- **products** — `source_record_id source_file appl_no_raw appl_type appl_no
  appl_no_stripped product_no drug_name active_ingredient form route strength
  reference_drug reference_standard product_ndc marketing_status_name`
- **applications** — `source_record_id source_file appl_no_raw appl_type appl_no
  appl_no_stripped sponsor_name common_or_original_name submission_classification
  orphan_status`
- **submissions** — `source_record_id source_file appl_no_raw appl_type appl_no
  appl_no_stripped submission_type submission_no submission_status
  submission_status_date submission_notes`
- **lookups** — `lookup_type term appl_no appl_no_stripped appl_type`

Empty cells render as `""` (literal quotes) and fields containing a quote/tab/CR/LF are
quoted with doubled quotes — exactly polars `write_csv(separator="\t")` behavior, so Go and
Python TSV bytes match.
