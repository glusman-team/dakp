# Debug: DAG "succeeds" despite tablassert exiting 1

## Context

The `dakp_build` Airflow DAG was marked **success**, but the `run_tablassert` task
actually **failed** — tablassert exited with code 1 (crashed during Stage 5 "BUILDING
SUBGRAPHS" → `resolve` for `approved_treats`). Three issues caused this false success
and masked the real error.

---

## Bug 1 (DAKP code): Failure is silently swallowed

### Root cause

`TablassertRunner.run()` (`src/dakp_pipeline/tablassert.py:500-504`) deliberately
**does not raise** when the tablassert subprocess exits non-zero. Instead it:

1. Logs an error (`logger.error("Tablassert exited {}: ...")`)
2. Sets `status = "failed"` in the handoff report JSON
3. Writes the report to disk
4. **Returns normally** — no exception

The docstring explicitly documents this design choice:
> A non-zero exit is captured as `status: failed` (logged loudly), not raised — the report
> is the artifact the pipeline surfaces.

This made sense in the old pure-Python pipeline (retired), where downstream stages would
inspect the report's `status` field. But under Airflow:

- The DAG's `run_tablassert` task calls `tablassert.run(...)` and returns
  `refs_to_xcom(...)` — a valid artifact-ref list. Airflow sees a normal return → **marks
  the task `success`**.
- The downstream `write_build_summary` task runs, records the handoff ref's *path* in the
  summary, but **never reads the JSON** to check `status == "failed"`.
- The entire DAG turns green.

Nothing in the pipeline ever checks `status`.

### Fix

**Raise after writing the report** in `TablassertRunner.run()`. The report is still
written to disk (available for post-mortem debugging), but the exception propagates to
Airflow so the task fails correctly.

```python
# tablassert.py, in TablassertRunner.run(), after report.update({...}):
refs = [_write_report(report, assertion_refs, ctx)]
if completed.returncode != 0:
    raise TablassertError(f"Tablassert exited {completed.returncode}; see handoff report: {refs[0].uri}\n{(completed.stderr or '').strip()[:500]}")
return refs
```

Define a small `TablassertError(RuntimeError)` exception class at module level.

### Files to modify

- `src/dakp_pipeline/tablassert.py` — add `TablassertError`; raise in `TablassertRunner.run()` after writing report; update the class docstring (remove "not raised"); add to `__all__`
- `tests/unit/test_tablassert_configs.py` — update `test_real_runner_records_failure` (currently asserts no raise + checks report); it should now `pytest.raises(TablassertError)` AND still verify the report was written with `status: "failed"`

---

## Bug 2 (data / environment): relative fullmap path resolves from the wrong cwd

### Root cause

tablassert crashed at Stage 5 during `compile_subgraph` → `resolve` for `approved_treats`.
The fullmap path is the cause — not because the fullmap doesn't exist, but because the
**relative path resolves from the wrong working directory**.

The `--fullmap ./../../DBSTORE/FULLMAP/fullmap` value is passed **verbatim** from the
`--fullmap` CLI flag through the Airflow Variable to tablassert. tablassert reads the
fullmap path from `graph.yaml`'s `fullmap` field and resolves it **relative to its cwd**
(the workdir root).

The same relative path resolves differently depending on where you stand:

| CWD | Resolves to | Exists? |
|---|---|---|
| Repo root (where `dakp up` runs) | `/local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap` | ✅ (`data  downloads`) |
| tablassert's cwd (`…/tmp/`) | `/local_raid1/sgoetz/CODE/DAKP/DBSTORE/FULLMAP/fullmap` | ❌ |

Confirmed with `pathlib.resolve()`:
```
/local_raid1/sgoetz/CODE/DAKP + ../../DBSTORE/FULLMAP/fullmap
  → /local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap        ← correct

/local_raid1/sgoetz/CODE/DAKP/tmp + ../../DBSTORE/FULLMAP/fullmap
  → /local_raid1/sgoetz/CODE/DAKP/DBSTORE/FULLMAP/fullmap  ← does not exist
```

When `fullmap_db_path()` can't find the `.redb` at the resolved path, `resolve_batch()`
crashes trying to open a non-existent redb database — exactly the Stage 5 `resolve` crash.

### Fix: resolve fullmap to absolute at `dakp up` time

Anchor relative `--fullmap` values to the **user's CWD** when `dakp up` runs, not to
tablassert's CWD at task-run time. In `build_context_from_config()`
(`src/dakp_pipeline/runtime.py:42`), resolve the fullmap to absolute:

```python
fullmap_raw = cfg.get("fullmap")
fullmap = str(Path(fullmap_raw).resolve()) if fullmap_raw else None
```

This way `--fullmap ./../../DBSTORE/FULLMAP/fullmap` becomes
`/local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap` at config-write time, and tablassert always
sees an absolute path regardless of its own cwd.

For the immediate re-run, you can also just pass an absolute path:
```bash
dakp up --fullmap /local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap
```

### Files to modify

- `src/dakp_pipeline/runtime.py` — resolve `fullmap` to absolute in `build_context_from_config()`
- `tests/unit/test_runtime.py` (or equivalent) — add a test that a relative fullmap is resolved to absolute

---

## Bug 3 (DAKP code): stderr truncated in logs

### Root cause

The error log in `TablassertRunner.run()` truncates stderr to 2000 chars:
```python
logger.error("Tablassert exited {}: {}", completed.returncode, (completed.stderr or "").strip()[:2000])
```
This cuts off the end of the traceback — exactly the part that names the actual exception.
The full stderr IS captured in the handoff report JSON (`report["stderr"]`), but the
Airflow task log (what you see in the UI) is truncated.

### Fix

Remove the `[:2000]` truncation. The full stderr is the diagnostic the user needs when a
build fails. (The report JSON already stores it untruncated, so this just makes the log
match.)

```python
logger.error("Tablassert exited {}: {}", completed.returncode, (completed.stderr or "").strip())
```

### Files to modify

- `src/dakp_pipeline/tablassert.py` — remove `[:2000]` from the error log in `TablassertRunner.run()`

---

## Steps

- [ ] **Bug 1 fix:** Add `TablassertError` exception class to `tablassert.py`
- [ ] **Bug 1 fix:** Raise `TablassertError` in `TablassertRunner.run()` after writing the report when `returncode != 0`
- [ ] **Bug 1 fix:** Update `TablassertRunner` class docstring (remove "not raised" language)
- [ ] **Bug 1 test:** Update `test_real_runner_records_failure` to assert both the exception AND the report on disk
- [ ] **Bug 2 fix:** Resolve fullmap to absolute in `build_context_from_config()` (`runtime.py`)
- [ ] **Bug 2 test:** Test that a relative fullmap is resolved to absolute
- [ ] **Bug 3 fix:** Remove `[:2000]` stderr truncation in `TablassertRunner.run()` error log

## Verification

```bash
# Unit tests pass (including updated failure test + fullmap resolution test)
uv run pytest tests/unit/test_tablassert_configs.py tests/unit/test_tablassert_run_edge.py tests/unit/test_runtime*.py -v

# Re-run the DAG — the task should now FAIL (red) if tablassert crashes, and the full
# traceback should be visible in the Airflow task log (no more truncation).
# The relative fullmap path should now resolve correctly (absolute) at dakp up time.
dakp up --fullmap ./../../DBSTORE/FULLMAP/fullmap
```
