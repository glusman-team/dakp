# Fix: `dakp up` crashes on a corrupt Airflow install — preflight probe can't see data-file corruption

## Context

`dakp up -f ./PATH/ -d` on the deployment machine (`wenceslaus:/local_raid1/sgoetz/CODE/DAKP`) dies with:

```
yaml.reader.ReaderError: unacceptable character #x0080: control characters are not allowed
  in ".venv/.../airflow/config_templates/config.yml", position 111403
```

Three stacked problems:

1. **Corrupt installed file on the deploy machine.** A healthy `config.yml` has the UTF-8 em-dash
   `e2 80 94` at offsets **111401–111403** ("…for that instance — including…") — verified against
   our local pristine copy; the whole file has only 12 non-ASCII bytes, all dash glyphs. On the
   deploy machine that spot now contains a lone `0x80` control byte. The file is owned by
   `apache_airflow_core-3.3.0` (verified via its `RECORD`). The deploy machine is a plain
   `git clone` + `uv sync` (no copy path to mangle bytes), so the suspects are bit-rot on the
   RAID-backed disk or a poisoned uv-cache wheel.
2. **The self-heal probe is too weak to see it.** `_preflight()` (`src/dakp_pipeline/cli.py:176`)
   probes with `airflow_importable()` = `importlib.util.find_spec("airflow") is not None`.
   `find_spec` only *locates* the package, it never executes it — verified empirically it returns
   `True` even for a package whose `__init__.py` raises. So preflight declares the install healthy,
   and the deferred `from dakp_pipeline.dags.dakp_build import …` (`cli.py:223`) does the real
   import and dies inside YAML parsing — exactly the reported traceback. The docstring promise
   "a corrupt venv is healed rather than crashing on import" does not hold for this corruption class.
3. **Aggravating: the deploy machine runs a stale checkout** — `984590d` (still has
   `run_up(profile=…)`, matching traceback lines 223/383), while HEAD is `1df53a0`. The old
   checkout's run-wait is **15 min** (`_RUN_WAIT_ROUNDS=300 × 3s`); HEAD raised it to **60 min**
   (`282cddc`) because GLiNER contraindication mining + KG build take 20+ min. A full build on the
   stale checkout would time out mid-run even with a healthy venv.

## Approach

Make the preflight probe a **real import check in a subprocess**, routed through the existing
`run_subprocess` boundary. Keep the name/signature of `airflow_importable()` so every existing test
that monkeypatches it keeps working unchanged:

```python
def airflow_importable() -> bool:
    """True when ``import airflow`` really succeeds here (the preflight health probe).

    Runs a real import in a subprocess: ``find_spec`` only locates the package and passes even when
    shipped data files are corrupt (a bit-rotted ``config_templates/config.yml`` kills
    ``import airflow`` deep inside YAML parsing). Subprocess so a broken install never poisons this
    process and a just-healed install is re-tested against fresh bytes.
    """
    return run_subprocess([sys.executable, "-c", "import airflow"]).returncode == 0
```

- **Subprocess, not in-process try/except** — a failed import must not leave partial modules in the
  CLI process, and the post-heal re-probe must see the freshly reinstalled bytes.
- **`sys.executable`** — same interpreter/venv the CLI runs under; no uv overhead per probe.
- **Via `run_subprocess`** — the project's monkeypatch boundary; orchestration tests unchanged.
- `_preflight`'s heal ladder stays: reinstall `apache-airflow-core` → if still broken,
  `uv cache clean apache-airflow-core` + reinstall → else bail with guidance.

### Deployment-machine unblock (ops steps, run there — no code change needed)

```
cd /local_raid1/sgoetz/CODE/DAKP
uv sync --reinstall-package apache-airflow-core
uv run python -c "import airflow; print(airflow.__version__)"    # must print 3.3.0
# if it STILL fails (poisoned uv cache re-supplies the bad bytes):
uv cache clean apache-airflow-core && uv sync --reinstall-package apache-airflow-core
```

Then before the full build: `git pull && uv sync` to get HEAD (60-min run-wait + all fixes since);
the `dakp up -f ./PATH/ -d` flags are unchanged. (User confirmed: will pull before the full build.)
If the file corrupts **again** after a clean reinstall, suspect the RAID — run a scrub/fsck.

## Files to modify

- `src/dakp_pipeline/cli.py` — rewrite `airflow_importable()` body (subprocess import probe);
  replace now-unused `import importlib.util` with `import sys`; refresh its docstring. Nothing else
  changes (`_preflight`, `run_up`, heal ladder all stay).

> **Explicitly out of scope** (user deferred): moving `DAG_ID`/pool/`CONFIG_VARIABLE` constants out
> of the airflow-importing DAG module so `dakp up` never imports airflow in-process. With a real
> probe, corruption is caught and healed before that import anyway.
- `tests/unit/test_cli_edge.py` — `test_airflow_importable_true_when_installed` exercises the real
  body; it now spawns `[sys.executable, "-c", "import airflow"]` and still asserts `True`. Update
  its comment; no other test changes (all orchestration tests monkeypatch `airflow_importable`).

## Reuse

- `run_subprocess` (`src/dakp_pipeline/cli.py:66`) — existing captured-subprocess boundary.
- `_preflight` heal ladder (`src/dakp_pipeline/cli.py:176`) — unchanged.
- Test helpers `_bools` / `_patch_happy` / `FakeSubprocess` (`tests/unit/test_cli.py`) — unchanged.

## Steps

- [ ] Rewrite `airflow_importable()` as the subprocess import probe; update docstring.
- [ ] Swap `import importlib.util` → `import sys`; leave `__all__` unchanged.
- [ ] `uv run ruff check src tests` and `uv run pyright` clean.
- [ ] `uv run pytest` — full suite incl. the 100% branch-coverage gate.
- [ ] Corruption drill (Verification §2) and restore the local venv afterwards.
- [ ] Commit: `fix(cli): preflight probe does a real airflow import (catches corrupt installs)`.

## Verification

1. `uv run pytest` green with the 100% branch gate (preflight branches:
   `tests/unit/test_cli.py::test_up_preflight_reinstall_heals / _cache_clean_heals / _still_broken`).
2. **Corruption drill on this machine** (backup → corrupt → probe → restore):
   - Back up and flip offset 111403 of
     `.venv/lib/python3.12/site-packages/airflow/config_templates/config.yml` to `0x80`.
   - `uv run python -c "from dakp_pipeline.cli import airflow_importable; print(airflow_importable())"`
     → must print `False` (old code prints `True` — that's the bug).
   - `uv sync --reinstall-package apache-airflow-core`; probe prints `True` again.
3. Deploy machine: run the unblock commands, then `dakp up -f ./PATH/ -d`.
