# Fix: shape_contraindication_tables crashes — torch 2.11+cu128 cannot run on P100 (sm_60)

## Context

The Airflow task `shape_contraindication_tables` (DAG `dakp_build`) fails with:

```
RuntimeError: cuDNN version 91900 is not compatible with devices with SM < 7.5.
```

preceded by warnings that the installed PyTorch only supports
`sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, while the build host **wenceslaus has
4× Tesla P100-PCIE-16GB = compute capability 6.0 (sm_60)**.

**Chain of causes:**

1. The shaper (`contraindications.py`) dispatches GLiNER NER mining across the 4 P100s
   (`CONTRAINDICATION_GPUS`, two-pass multi-GPU pool, commit history in
   `plans/multi-gpu-contraindications.md`).
2. Commit `c70666a` (Aug 6) pinned `torch>=2.9,<2.12` from the **cu128** wheel index to fix a
   *different* bug: default PyPI torch 2.13 wheels are +cu130 and refused the CUDA 12.8 driver
   ("driver too old, found version 12080").
3. But per PyTorch's July 2025 packaging policy, **starting with release 2.8, Maxwell and Pascal
   kernels were removed from the CUDA-12.8/12.9 binaries** (kept only in the cu126 builds). So
   every torch ≥2.8 from the cu128 index — including the locked `2.11.0+cu128` with
   `nvidia-cudnn-cu12 9.19.0.56` — can never compute on a P100. The first CUDA call inside a
   mining pool worker raises, the future fails, and the task dies at
   `_mine_two_passes_multi_gpu` → `future.result()`.

**Verified external facts**

- `uv.lock`: `torch 2.11.0+cu128` + `nvidia-cudnn-cu12 9.19.0.56` — matches the error exactly.
- PyTorch announcement (dev-discuss, 2025-07-14): Pascal removed from cu128/cu129 binaries
  starting with 2.8; **cu126 builds retain Maxwell/Pascal**.
- `torch-2.8.0+cu126-cp312-manylinux_2_28_x86_64.whl` exists on the official cu126 index.
- torch 2.8.0 depends on `nvidia-cudnn-cu12==9.10.2.21` (predates the SM<7.5 block).
- CUDA 12.6 runtime wheels run fine on the 570.211.01 driver (supports CUDA ≤12.8 runtimes).
- No other dependency constrains torch upward: gliner 0.2.28, transformers, sentence-transformers
  5.6.1 (via `tablassert[qc]`) all accept torch 2.x; tablassert has no direct torch dep.
- CI runs on `ubuntu-latest` only → a Linux-only wheel index (cu126) is safe.
- CUDA touchpoints in the codebase are exactly two: `ner.py::_model_device()` and
  `contraindications.py::_resolve_devices()`.

## Approach (user-approved: full recommended fix)

**A. Dependency fix — restore GPU mining:** pin `torch>=2.8,<2.9` sourced from the **cu126**
index (`https://download.pytorch.org/whl/cu126`). torch 2.8.0+cu126 is the newest official wheel
line that still contains sm_60 kernels and ships a P100-compatible cuDNN. This un-breaks the
existing 4-GPU dispatch with zero code changes to the mining path.

**B. Defensive code gate — never crash on incompatible GPUs again:** both CUDA entry points
currently trust `torch.cuda.is_available()`, which returns True even when the installed torch
has no kernels for the present GPU (exactly this bug). Gate on the installed torch's compiled
architecture list instead.

New helper in `ner.py` (imported by `contraindications.py`, which already imports from there):

```python
def _cuda_device_supported(torch_mod: Any, device_index: int) -> bool:
    """True when the installed torch has kernels compiled for ``cuda:device_index``.

    ``torch.cuda.is_available()`` only checks driver + runtime — it lies when torch was built
    without kernels for the present GPU arch (torch >=2.8 cu128 wheels have no sm_60 code, so
    the P100s pass is_available() but raise on first use). A device is usable iff its compute
    capability appears in the arch list torch was compiled for.
    """
    try:
        major, minor = torch_mod.cuda.get_device_capability(device_index)
    except Exception:  # defensive: driver/runtime errors surface as unusable, not a crash
        return False
    return f"sm_{major}{minor}" in torch_mod.cuda.get_arch_list()
```

- `_model_device()`: after `is_available()`, return `"cuda"` only if
  `_cuda_device_supported(torch, 0)`, else log `logger.warning("ner_device_unsupported: ...")`
  (loguru `{}` style, cf. `translator.py:167`) and return `"cpu"`.
- `_resolve_devices()`: keep only the visible devices (capped at `len(CONTRAINDICATION_GPUS)`)
  whose arch is compiled into torch; return the matching `CONTRAINDICATION_GPUS` entries. When
  none qualify, log a warning and return `None` (existing sequential CPU fallback).

Result: the next torch bump that drops the P100s degrades to (slow but correct) CPU mining with
a clear warning instead of killing the DAG run mid-pool.

## Files to modify

- `pyproject.toml` — torch pin `>=2.9,<2.12` → `>=2.8,<2.9`; `[tool.uv.sources]` index
  `pytorch-cu128` → `pytorch-cu126` (URL `.../whl/cu126`); rewrite both comments to record the
  P100/sm_60 constraint (cu128/cu129 binaries lack Pascal from torch 2.8 on).
- `uv.lock` — regenerate via `uv lock` (torch 2.11.0+cu128 → 2.8.0+cu126, cuDNN 9.19 → 9.10.2.21).
- `src/dakp_pipeline/ner/ner.py` — new `_cuda_device_supported()` helper; `_model_device()`
  arch gate.
- `src/dakp_pipeline/assertions/contraindications.py` — `_resolve_devices()` arch gate (import
  the helper from `ner.ner`).
- `tests/unit/test_ner_edge.py` — update `test_model_device_selects_cuda_when_available` to
  also stub `get_device_capability` → `(6, 0)` and `get_arch_list` → `[..., "sm_60", ...]`;
  new: CPU fallback when sm_60 absent from arch list; CPU fallback when
  `get_device_capability` raises.
- `tests/unit/test_assertions_contraindications_edge.py` — existing `_resolve_devices` tests
  (`..._returns_gpus_when_cuda_available`, `..._caps_at_visible_device_count`) need the same
  new stubs; new: returns `None` when no visible GPU's arch is compiled in (warning logged);
  mixed fleet keeps only supported devices; `get_device_capability` raising counts as
  unsupported.

Reuse: existing `CONTRAINDICATION_GPUS`, `_mine_multi_gpu` / `_mine_two_passes_multi_gpu`
machinery, and the sequential CPU fallback path in `build_contraindication_rows` all stay
untouched. Coverage gate is `fail_under = 100` — new branches must be fully tested.

## Steps

- [ ] Edit `pyproject.toml`: `"torch>=2.8,<2.9"`; rename index `pytorch-cu128` →
      `pytorch-cu126` with URL `https://download.pytorch.org/whl/cu126`; rewrite both comments
      to record the P100 constraint (PyPI +cu130 wheels refuse the CUDA 12.8 driver; torch ≥2.8
      cu128/cu129 binaries have no Maxwell/Pascal kernels, so P100/sm_60 needs the cu126 line).
- [ ] `uv lock` and sanity-check the lock diff (torch 2.11.0+cu128 → 2.8.0+cu126,
      `nvidia-cudnn-cu12` 9.19.0.56 → 9.10.2.21). If universal resolution complains about a
      platform with no cu126 wheel, add a `[tool.uv] environments` restriction matching the
      Linux-only deployment — but the current cu128 setup resolves without one, so expect none.
- [ ] Add `_cuda_device_supported()` + gate in `ner.py::_model_device`.
- [ ] Gate `contraindications.py::_resolve_devices` per visible device.
- [ ] Add/adjust unit tests (both test files); `uv run pytest -q --cov` green, coverage 100%
      (every new branch — including the `except Exception` path — needs a test).
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pyright`.
- [ ] Deploy on wenceslaus: pull branch, `uv sync`, re-run the `shape_contraindication_tables`
      task; confirm logs show `dispatch_gpus = 4`, no sm_60 warnings, and the task completes.
      (Airflow spawns a fresh Python process per task instance, so no scheduler restart is
      needed beyond the venv update — llama-server stays as-is; ~3–5 GB free per GPU, GLiNER
      workers need ~1 GB each.)

## Verification

1. Local: `uv sync && uv run pytest -q --cov` + ruff + pyright.
2. On wenceslaus after `uv sync`:
   `uv run python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); x = torch.zeros(1).cuda(); (x @ x).sum(); print(torch.backends.cudnn.version())"`
   → expect `2.8.0+cu126`, `sm_60` in the arch list, cuDNN `91002`.
3. Re-run the DAG task; expect `shape_contraindications: dispatch_gpus = 4`, no
   `sm_60 is not compatible` warnings, `failed = false`, and a non-empty
   `contraindication_assertions.tsv`.

## Decision record

User approved the full recommended fix: torch downgrade to 2.8.0+cu126 **and** the defensive
arch gate. llama-server coexistence on the P100s is accepted (adequate memory headroom).
