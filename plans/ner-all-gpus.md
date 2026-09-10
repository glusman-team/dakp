# Ensure NER uses all GPUs concurrently and correct the production labels

## Context
The repository already has multi-GPU NER dispatch: production `DiseaseNER` instances can be pinned to `cuda:N`, and `ner_dispatch` shards work into one spawned worker per available GPU. The requested outcome is to ensure all four GPUs are actively used concurrently, with separate batches, to minimize NER wall time.

Initial findings:
- `src/dakp_pipeline/assertions/ner_dispatch.py` defines four build-host devices (`cuda:0`–`cuda:3`), resolves visible/supported devices, balances work by text length, and launches one process per GPU with `spawn`.
- `src/dakp_pipeline/ner/ner.py` supports per-worker device pinning and one model per GPU.
- Three assertion shapers call the dispatch layer; contraindications can dispatch multiple passes concurrently.
- The Airflow DAG serializes shape tasks in the `ner_mining` pool, but each shape task is intended to use all GPUs internally.
- This host currently reports only one visible GPU via `nvidia-smi -L`; code should still use all four when the runtime exposes them, and degrade safely when it does not.

## Approach
The four-GPU dispatch already exists end to end; the remaining speed lever is **batched GLiNER inference inside each per-GPU worker**. The published `SkyeAv/drug-approvals-gliner-small-v2.1` model card specifies the trained labels `disease` and `phenotype`, not the currently hardcoded fused `DiseaseOrPhenotype` label. Update production NER to request both trained labels, preserve the returned label through `canonical_type`, and remove/update the fused-label fallback assumptions and tests/docs. Model-only spans use the model's disease/phenotype label; gazetteer types remain authoritative on contested spans. Indication vs contraindication remains shaper context metadata, not an entity label. Today `_mine_shard` calls `ner.extract(text)` per text, and `_merge_model_spans` calls `model.predict_entities(window, ...)` one window at a time — batch size 1, so each P100 idles between kernel launches. GLiNER 0.2.28 ships `GLiNER.inference(texts, labels, ..., batch_size=...)` (the deprecated `batch_predict_entities` forwards to it) for real batched inference.

1. Add a batched extraction path on `DiseaseNER` (e.g. `extract_batch(texts)` or a `batch_size` path in `_mine_shard`): compute windows for a chunk of texts, run one `model.inference(all_windows, labels, batch_size=N)` call, remap spans per text/window, then run the unchanged per-text gazetteer merge. Output must remain byte-identical to the sequential path.
2. `_mine_shard` chunks its LPT shard into batches of texts (default batch size TBD, ~16–32 windows) so each of the 4 GPU workers runs back-to-back batched inference.
3. Keep the existing 4-GPU sharding/dispatch, device filtering, flock, and CPU/offline fallbacks untouched.
4. Tests: batch path equals sequential path on the same texts (production-mode backend with monkeypatched model), chunking behavior, and offline pass-through. Keep 100% branch coverage gate in mind.

### CI fix and delivery
`main` CI fails only on formatting: `ruff format --check` flags `tests/integration/test_kgx_end_to_end.py` (line 188 `_CONTRA` list should be one line) and `tests/unit/test_ner_edge.py` (blank-line placement around lines 204/224). Both `python-lint` and `pre-commit` jobs fail on exactly this; type/test/go jobs pass. Fix these files first, verify the CI-equivalent checks locally, then commit the CI fix directly to `main` before committing the GPU implementation/version bump directly to `main`.

## Files to modify
- `src/dakp_pipeline/ner/ner.py` — correct production labels (`disease`, `phenotype`) and add batched inference on `DiseaseNER` (windows flattened across texts, one `model.inference` call per chunk, per-text merge unchanged).
- `src/dakp_pipeline/assertions/ner_dispatch.py` — `_mine_shard` batches texts through the new path.
- `tests/unit/test_assertions_ner_dispatch.py`, `tests/unit/test_ner_edge.py` (or a new test file) — batch-equals-sequential and chunking tests.
- `tests/integration/test_kgx_end_to_end.py`, `tests/unit/test_ner_edge.py` — `ruff format` (CI fix).

## Reuse
- `DiseaseNER(device=...)` and `_config()` in `src/dakp_pipeline/ner/ner.py`.
- `_resolve_devices`, `_shard_by_text_length`, `_mine_shard`, `_mine_multi_gpu`, and `mine_passes_multi_gpu` in `src/dakp_pipeline/assertions/ner_dispatch.py`.
- Existing `mine_with_cache` to ensure only cache misses are dispatched.

## Decisions
- The observed issue is that only one GPU shows compute activity at a time even though all models are resident. Add true batched GLiNER inference within each GPU worker; preserve one independent worker/model per device.
- Padding is acceptable, so use GLiNER's `inference(..., batch_size=...)` and tune a practical default for the 16 GB cards. Span extraction remains deterministic; minor floating-point score differences from padded batches are acceptable.
- Discover devices from the runtime (`torch.cuda.device_count()` / visible CUDA ordinals), filter unsupported architectures, and use every visible supported device. Keep the per-device flock so there is never more than one model process per device. No fixed four-device cap.
- Fix the two Ruff formatting failures before committing the implementation. The CI fix is part of this change, and the version bump will be `1.11.1` (patch release).

## Steps
- [ ] Inspect all shaper call sites, tests, runtime configuration, and current GPU-related assumptions.
- [ ] Replace the stale fused-label production assumptions with the checkpoint's trained `disease`/`phenotype` labels and update related docs/tests.
- [ ] Decide the exact batching/concurrency change needed based on the call-path audit.
- [ ] Implement four-way concurrent dispatch with safe visibility and fallback behavior.
- [ ] Add regression tests for four-device scheduling, separate shards, multi-pass behavior, and output determinism.
- [ ] Run focused tests, lint/type checks, and a GPU-aware smoke/benchmark check where hardware is available.

## Verification
- Confirm logs show one worker/model per visible supported CUDA device for a sufficiently large production NER workload, with no device running more than one model.
- Confirm the production checkpoint is called with exactly its trained `disease` and `phenotype` labels and model-only output preserves those types.
- Confirm each device receives a distinct shard and all four workers overlap in execution.
- Confirm outputs match sequential extraction, cache hits are not redundantly mined, and one-/zero-GPU environments continue to work.
- Run the relevant pytest targets plus repository quality checks.
