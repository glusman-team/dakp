# Pipeline speed: content-hash fixes, NER mention cache, stage skip, scan rewrites

## Context

Iterative `dakp up` runs were slow because everything downstream of acquisition rebuilt from
scratch every run, and the most expensive rebuilds had no persisted shortcuts:

- Every artifact ingest/register read each file **twice** (BLAKE3, then SHA-256 SRI).
- The Go DailyMed extractor re-hashed every SPL input file even though the incoming
  `ArtifactRef`s already carried the BLAKE3 id from acquisition.
- The NER model tree (~GBs) was fully re-hashed on every cache hit — once per
  `DiseaseNER._load_model`, including every spawned GPU worker (~13–16×/run).
- GLiNER mention mining (the ~20+ min long pole) ran from scratch in all three shape tasks;
  results were never persisted.
- Extract/shape stages had no "inputs unchanged → skip" check; manifests recorded `inputs` +
  `operation` but nothing consulted them.
- `build_dailymed_evidence` row-scanned `spl_sections.parquet` twice per run;
  `_faers_candidates` iterated tens of millions of FAERS rows in Python;
  `check_assertion_tables` materialized every assertion row into one list.
- `acquire_dailymed` pushed one ArtifactRef dict per SPL member (tens of thousands) over XCom,
  which had already caused Execution API `ReadTimeout`s.

## What landed

### Hashing

- `content_hash.hash_file_with_sri` / `blake3store.HashFileWithSRI`: one streaming pass updates
  BLAKE3 and SHA-256 together; used by `ArtifactStore.ingest`/`register` and Go
  `Store.Register`.
- `dailymed.Extract` takes the refs' BLAKE3 ids (`Extract(ctx, paths, ids, limit)`), falling
  back to per-file hashing only when an id is missing — the FAERS pattern.
- Model-cache verification (`ner/model_cache.py`): manifests record `file_count`/`total_bytes`;
  cache hits verify with a fast stat walk instead of `hash_tree`. Legacy manifests get one full
  hash, then backfill. `DAKP_MODEL_VERIFY=full` restores full-tree verification. Trade-off:
  same-size, same-count content tampering is only caught under `=full`.

### NER mention cache (Pebble, Go server + Python client)

- `go/cmd/dakp-nercache` + `go/internal/nercache`: Pebble DB at `<workdir>/cache/ner/`,
  HTTP on `127.0.0.1:<ephemeral>`; discovery via `<workdir>/cache/ner/server.json`
  (pid + port, atomic write, removed on clean shutdown). Endpoints: `GET /health`,
  `POST /batch_get`, `POST /batch_put` (Pebble batch, Sync), `GET /stats`. A second server over
  a locked DB exits non-zero.
- Key = `blake3("<model_id>|<model_b3>|<config_fingerprint>|<normalized_text>")` (64-hex, no
  `b3:` prefix; text stripped + whitespace-collapsed). The key fully pins the HF model: the
  exact upstream id AND the checkpoint content tree hash from the model-cache manifest — so
  swapping or re-uploading a model invalidates every entry by construction. Key material comes
  from `ner_cache_material(ner)`, which reads the manifest without loading GLiNER; the offline
  gazetteer backend is never cached.
- Python client `ner/mention_cache.py` (`MentionCache`): reuses a live server from
  `server.json`, else spawns the binary (`DAKP_NERCACHE_BIN` → `<workdir>/bin/dakp-nercache` →
  PATH). Any failure degrades to a warn-once no-op cache — tests and CPU-only runs need zero
  setup. `Mention.to_dict/from_dict` round-trips losslessly; cache hits are byte-identical to
  re-mining.
- Dispatch: `mine_with_cache` in `assertions/ner_dispatch.py` is the single integration point —
  batch-get by text, mine only misses, batch-put, merge. Spawned GPU workers never touch the
  cache (Pebble stays single-owner). All three shapers route through it.
- CLI: `dakp up` builds the binary to `<workdir>/bin/dakp-nercache` (non-fatal on failure);
  `dakp cache clear` stops a live server and deletes the store.

### GPU model cap

- `DiseaseNER._load_model` takes a blocking exclusive `fcntl.flock` on
  `<lock_dir>/cuda-N.lock` before loading GLiNER on `cuda:N` (lock dir:
  `$DAKP_GPU_LOCK_DIR` → `<workdir>/cache/gpu-locks` → sibling of the model-cache root). Held
  for the life of the model; released on process exit (workers are short-lived). CPU and
  offline paths never lock. At most one model per GPU — 4 total — regardless of Airflow task
  concurrency.
- The three shape tasks also run in the `ner_mining` Airflow pool (1 slot) to serialize mining
  and avoid lock-wait churn.

### Stage skip + shared evidence

- Operation index `data/manifests/_index.json` (`{"version": 1, "entries": {...}}`), key =
  `hex(BLAKE3(operation + "|" + "|".join(sorted(input_ids))))`. Maintained on `register()` /
  `record_operation`, flock-serialized, atomic rename; lookups prune entries whose files
  vanished. Byte-compatible Go mirror in `go/internal/airflow/opindex.go`, pinned by shared
  test vectors on both sides.
- All three Go extract tasks skip when inputs are unchanged (`cfg.Force` bypasses).
- The three Python shape tasks skip via `cached_shape_outputs` with a
  `shape_config_fingerprint` (disease_map content, NER cache material, contraindication
  keywords) folded into the key.
- `load_or_build_dailymed_evidence` persists the built `DailyMedEvidence` as a pickled store
  artifact keyed by the consumed `spl_*.parquet` ids + `EVIDENCE_BUILDER_VERSION`;
  `approved_treats` and `contraindications` share it instead of rescanning twice.

### Scan rewrites + XCom

- `_faers_candidates`: Python row loop → Polars lazy `select/filter/unique(keep="first",
  maintain_order=True)`; semantics pinned by differential fuzzing against the old loop.
- `check_assertion_tables`: streams per table via `scan_csv/scan_parquet().collect_batches()`
  instead of materializing all rows; identical violation reports.
- `acquire_dailymed` XCom is now one sentinel ref (`application/vnd.dakp.refs+json`) pointing
  at a store JSON file holding the member refs (`write_refs_manifest`). `refs_from_xcom`
  (Python) and `DecodeArtifactRefs` (Go) transparently resolve the sentinel; inline lists still
  decode (FAERS/Drugs@FDA unchanged). The `EXECUTION_API_TIMEOUT=1000` workaround stays — FAERS
  still pushes inline lists.

## Verification

- `uv run pytest -q`: full suite passes with the coverage gate at 100%.
- `uv run ruff check` + `ruff format --check`: clean.
- `cd go && go test ./... && go vet ./...` + `gofmt -l`: clean.

## Known follow-ups (not done)

- `runExtract` (go/cmd/dakp-bundle/main.go) still emits 2–3 `Stat` log lines per input ref over
  the supervisor IPC — the next per-member cost if extract still feels slow.
- FAERS/Drugs@FDA acquire tasks could get the same XCom refs-file treatment if their payloads
  grow.
