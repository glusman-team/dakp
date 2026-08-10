# Optimize the Airflow UI for DAKP (config-level, no fork)

## Context

The repo runs the **stock Apache Airflow 3 UI** via `airflow standalone`. Nothing in the repo
customizes appearance — `_airflow_env()` (`src/dakp_pipeline/cli.py`) sets executor/coordinator/queue
plumbing but **no UI keys**, and there is no `webserver_config.py` or `plugins/` dir. The generated
`tmp/airflow-home/airflow.cfg` is gitignored and rebuilt every run, so the only durable place to set
UI behavior is `_airflow_env()`.

Research (Airflow `customize-ui` docs + the live `airflow.cfg`) surfaced exactly three config-driven
wins that are high-value for *this* pipeline, all env-var driven and zero-risk:

1. **`instance_name = "DAKP"`** — browser tab title + homepage heading. Instantly distinguishes this
   orchestrator from any other Airflow on the host. Trivial, pure gain.
2. **`theme` (custom brand palette)** — recolors buttons/active states/accents away from Airflow's
   default blue to a biomedical **teal**. Cosmetic but improves orientation/ownership of the UI.
3. **`default_wrap = True`** — **functional.** The native Go workers emit one `slog` record per line
   as wide raw JSON (`plans/increase-airflow-logging.md`). The task-log view currently does not wrap,
   forcing horizontal scroll per line. Wrapping makes the log view readable.

### Explicitly NOT recommended (and why)

- **Nav-bar `icon` / `icon_dark_mode` SVG** — exists in the theme spec, but Airflow only serves the
  SVG from an app-relative `/static/...` path or absolute URL. `airflow standalone` has no committed
  static dir in this repo and no plugin loader wired, so it needs a `--static`/plugin path that
  doesn't exist today — added complexity for marginal value. Skipped.
- **`expose_config = True`** — handy for dev but a judgment call on what to surface; left at default
  `False`. (Easy to flip later if wanted.)
- `grid_view_sorting_order`, `auto_refresh_interval`, `hide_paused_dags_by_default` — already optimal
  or irrelevant for a single-DAG install.

## Approach

Add the three UI env keys inside the existing `_airflow_env()` `env.update({...})` block. Define the
theme JSON as a module-level constant (`json.dumps`-ed into the env var) so it stays readable and
the values are grep-able, matching how `COORDINATORS`/`QUEUE_TO_COORDINATOR` are already handled.

## Files to modify

- `src/dakp_pipeline/cli.py`
  - new module constant `_DAKP_UI_THEME: dict[str, Any]` (teal OKLCH 50–950 ramp, hue ~190–200,
    chroma ≤ 0.12, all within the documented `0 ≤ l ≤ 1`, `0 ≤ c ≤ 0.5`, `0 ≤ h ≤ 360` bounds).
  - `_airflow_env()`: add three keys to the `env.update({...})` dict:
    - `"AIRFLOW__API__INSTANCE_NAME": "DAKP"`
    - `"AIRFLOW__API__THEME": json.dumps(_DAKP_UI_THEME)`
    - `"AIRFLOW__API__DEFAULT_WRAP": "True"`
- `tests/unit/test_cli.py`
  - add one test asserting `_airflow_env(...)` carries the three UI keys with correct values, plus
    that the theme parses as valid JSON with a complete `brand` 50–950 ramp. Follows the existing
    boundary convention (call `cli._airflow_env` directly with tmp paths; no Airflow/Go needed).

## Reuse

- `json` — already imported in `cli.py` (used for `COORDINATORS`/`CONFIG_VARIABLE`).
- `_airflow_env()` — the single existing config-injection point; no new mechanism introduced.

## Steps

- [x] 1. In `src/dakp_pipeline/cli.py`, add `_DAKP_UI_THEME` constant (teal OKLCH scale, 11 stops).
- [x] 2. Add the three `AIRFLOW__API__*` keys to `_airflow_env()`'s `env.update({...})`.
- [x] 3. Add `test_airflow_env_carries_ui_customization` to `tests/unit/test_cli.py`.
- [x] 4. Run `uv run pytest tests/unit/test_cli.py -q` and `uv run ruff check src tests`.

## Verification

- Unit: `uv run pytest tests/unit/test_cli.py -q` passes (including the new test).
- Lint: `uv run ruff check` clean.
- Live (optional, manual): `uv run dakp down && uv run dakp up --small --detach`, then open
  `http://127.0.0.1:8090` and confirm: tab title/heading say **DAKP**, accents are teal, and a Go
  extract task's log wraps instead of horizontal-scrolling.
