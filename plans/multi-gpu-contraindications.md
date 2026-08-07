# Multi-GPU Contraindication Mining

## Context

The contraindication processing stage mines disease/phenotype mentions from DailyMed SPL
"Contraindications" sections (LOINC 34070-3) using a GLiNER zero-shot NER model. Currently
all inference runs on a single GPU: `_model_device()` in `ner.py` returns `"cuda"` (defaults
to GPU 0), and `build_contraindication_rows()` processes every document sequentially through
that one model instance. The host has 4 identical Tesla P100-PCIE-16GB GPUs, but 3 sit idle.

The workload is **embarrassingly parallel**: each `(set_id, doc_id, text)` triple is
independent — the NER extraction over one document has no dependency on any other. This
lends itself to data-parallel dispatch: shard the documents across N workers, each loading
its own GLiNER model on a different GPU, then merge the extracted mentions and feed them into
the existing aggregation logic.

## Approach

**Multi-process data parallelism** (not DataParallel / multi-threaded):

1. Add a `device` parameter to `DiseaseNER` so each worker can pin its GLiNER model to a
   specific GPU (`cuda:0`, `cuda:1`, ...). Default `None` preserves current auto-detect
   behavior (backward compatible for tests / offline mode).
2. Refactor `build_contraindication_rows` to collect work items first, then dispatch NER
   extraction either sequentially (`num_gpus=1`, unchanged behavior) or across N
   `ProcessPoolExecutor` workers (`num_gpus>1`), each pinned to one GPU.
3. Use the `spawn` start method (CUDA + `fork` is unsafe; PyTorch requires `spawn` for
   multi-process GPU usage).
4. Balance load with LPT (longest-processing-time-first) scheduling: sort items by text
   length descending, assign each to the shard with the least total text.
5. **Hardcode the 4 GPUs** on this build host (4× Tesla P100-PCIE-16GB). The device list
   `("cuda:0", "cuda:1", "cuda:2", "cuda:3")` is a module-level constant in
   `contraindications.py` — no runtime auto-detection. When CUDA is unavailable (CI,
   tests, machines without GPUs), the shaper falls back to sequential single-device
   processing so nothing breaks.
6. Results from all workers merge into a single `(set_id, doc_id) → list[Mention]` map, then
   the existing `_accumulate` / `_finalize_row` aggregation runs unchanged — guaranteeing
   identical output regardless of GPU count.

### CUDA pin compatibility

torch is already pinned to `>=2.9,<2.12` from the `pytorch-cu128` wheel index in
`pyproject.toml` (the default PyPI `+cu130` wheels refuse to initialize on the CUDA 12.8
driver: "driver too old, found version 12080"). The multi-GPU code uses only standard
PyTorch CUDA APIs (`torch.cuda.is_available()`, `map_location="cuda:N"`) that are fully
supported by the cu128 builds. **No changes to `pyproject.toml`, the torch pin, or the
wheel index are needed.**

## Files to Modify

### 1. `src/dakp_pipeline/ner/ner.py`

- **`_model_device()`** — unchanged (still returns `"cuda"` / `"cpu"` as before).
- **`DiseaseNER.__init__`** — add `device: str | None = None` parameter; store as
  `self._device`. When `None`, `_load_model` falls back to `_model_device()` (current
  behavior). When set (e.g. `"cuda:2"`), that device is used directly.
- **`DiseaseNER._load_model`** — change `map_location=_model_device()` to
  `map_location=self._device or _model_device()`.
- **New `DiseaseNER._config() -> dict`** — returns serializable construction kwargs
  (`offline`, `gazetteer`, `model_id`, `threshold`, `chunk_words`, `cache_dir`, `workdir`)
  so worker processes can reconstruct an equivalent `DiseaseNER` pinned to a specific GPU.
- No changes to `_model_device()` or the torch/CUDA pinning in `pyproject.toml`.

### 2. `src/dakp_pipeline/assertions/contraindications.py`

- **New module-level constant** `CONTRAINDICATION_GPUS: tuple[str, ...] =
  ("cuda:0", "cuda:1", "cuda:2", "cuda:3")` — the 4× Tesla P100-PCIE-16GB GPUs on the
  build host (wenceslaus). Hardcoded; not auto-detected.
- **New module-level `_mine_shard(shard, ner_config, device)`** — the `ProcessPoolExecutor`
  worker: reconstructs a `DiseaseNER(device=device, **ner_config)` and calls `extract()` on
  each text in its shard. Returns `list[(set_id, doc_id, list[Mention])]`.
- **New `_shard_by_text_length(items, n)`** — LPT scheduling: sorts items by text length
  descending, greedily assigns each to the least-loaded shard.
- **New `_mine_multi_gpu(work_items, ner, devices)`** — shards work across `len(devices)`
  workers via `ProcessPoolExecutor(mp_context=mp.get_context("spawn"))`, each pinned to
  `devices[i]`. Collects results into a `dict[(set_id, doc_id), list[Mention]]`.
- **Refactor `build_contraindication_rows`** — add `devices: Sequence[str] | None = None`
  keyword param (defaults to `None` = sequential single-device, current behavior):
  - Collect `(set_id, doc_id, text)` work items (same iteration order as today).
  - When `devices` is provided AND `len(work_items) > 1` AND the NER is production-mode
    (`offline=False`): call `_mine_multi_gpu(work_items, ner, devices)`.
  - Otherwise: sequential extraction (current behavior, unchanged).
  - Feed the resulting `(set_id, doc_id) → mentions` map into the existing
    `_accumulate` / `_finalize_row` aggregation (no changes to aggregation logic).
- **`ContraindicationsShaper.transform`** — after resolving `ner`, when it is production
  mode (`offline=False`) and CUDA is available (`torch.cuda.is_available()` via lazy import),
  pass `devices=CONTRAINDICATION_GPUS` to `build_contraindication_rows`. Otherwise pass
  `devices=None` (sequential fallback for tests / CI / non-GPU hosts).

### 3. `src/dakp_pipeline/dags/dakp_build.py`

- **No structural changes.** The `shape_contraindication_tables` task already creates
  `DiseaseNER(offline=False, workdir=ctx.workdir)`. The shaper checks
  `torch.cuda.is_available()` and uses the hardcoded `CONTRAINDICATION_GPUS` when CUDA is
  present, so the DAG wiring works as-is.

## Reuse

- **`build_dailymed_evidence`** (`assertions/evidence.py`) — unchanged; still builds the
  evidence index in the main process before dispatching.
- **`_accumulate` / `_finalize_row`** (`contraindications.py`) — unchanged; run in the main
  process after all workers return.
- **`Mention`** (`ner/lexical.py`) — frozen dataclass, already picklable across processes.
- **`ensure_model`** (`ner/model_cache.py`) — each worker calls it independently; the
  content-addressed cache ensures the model is downloaded once and shared from disk.

## Steps

- [x] **`ner.py`: Add `device` parameter + `_config()`**
  - Add `device: str | None = None` to `DiseaseNER.__init__`, store as `self._device`.
  - Change `_load_model` to use `self._device or _model_device()`.
  - Add `_config()` method returning serializable construction kwargs.
  - Verify the existing `torch>=2.9,<2.12` cu128 pin supports `map_location="cuda:N"`
    (it does — cu128 wheels are full CUDA builds).

- [x] **`contraindications.py`: Add multi-GPU dispatch infrastructure**
  - Add hardcoded `CONTRAINDICATION_GPUS = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")` constant.
  - Add `_mine_shard()` worker function (module-level, picklable for `spawn`).
  - Add `_shard_by_text_length()` LPT sharder.
  - Add `_mine_multi_gpu()` orchestrator using `ProcessPoolExecutor(spawn)`.

- [x] **`contraindications.py`: Refactor `build_contraindication_rows`**
  - Add `devices: Sequence[str] | None = None` keyword parameter.
  - Split into "collect work items" → "extract mentions (seq or multi-GPU)" → "aggregate".
  - Guard: only use multi-GPU when `devices` is given, `len(items) > 1`, and `ner` is production.

- [x] **`contraindications.py`: Wire hardcoded GPU list in the shaper**
  - `ContraindicationsShaper.transform` passes `devices=CONTRAINDICATION_GPUS` when the NER
    is production mode and `torch.cuda.is_available()`, else `devices=None` (sequential).

- [x] **Tests: `test_ner.py` / `test_ner_edge.py`**
  - Test `device` parameter is stored and passed to `map_location` (using fake GLiNER).
  - Test `_config()` returns correct kwargs.

- [x] **Tests: `test_assertions_contraindications.py` / `_edge.py`**
  - Test `_shard_by_text_length` distributes work evenly (LPT).
  - Test `_mine_shard` with a fake offline NER (no GPU needed).
  - Test `build_contraindication_rows(devices=None)` still produces identical output (regression).
  - Test multi-GPU path produces identical output to sequential (monkeypatch
    `_mine_multi_gpu` to use sequential extraction, assert same rows).

## Verification

1. **Unit tests**: `uv run pytest tests/unit/test_ner.py tests/unit/test_ner_edge.py tests/unit/test_assertions_contraindications.py tests/unit/test_assertions_contraindications_edge.py -v`
2. **Regression**: confirm `build_contraindication_rows(inputs, ner, devices=None)` output is
   byte-identical to the pre-change output (same row order, same values).
3. **On the 4-GPU host**: run the full pipeline (`uv run dakp up`) and observe `nvidia-smi`
   showing all 4 GPUs active during the `shape_contraindication_tables` task.
4. **Determinism**: run twice, confirm identical `contraindication_assertions.tsv` output.
