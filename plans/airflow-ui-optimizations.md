# Optimize the Airflow UI for DAKP (config-level, no fork)

## Context

The repo runs the **stock Apache Airflow 3 UI** via `airflow standalone`. `_airflow_env()`
(`src/dakp_pipeline/cli.py`) sets executor/coordinator/queue plumbing and the retained
`AIRFLOW__API__INSTANCE_NAME` and `AIRFLOW__API__DEFAULT_WRAP` settings; it does not inject a
custom color theme. There is no `webserver_config.py` or `plugins/` dir. The generated
`tmp/airflow-home/airflow.cfg` is gitignored and rebuilt every run, so the only durable place to set
UI behavior is `_airflow_env()`.

> **Current implementation note:** The custom `AIRFLOW__API__THEME` teal palette described below was
> intentionally removed. The instance name and default log wrapping remain unchanged.

Research (Airflow `customize-ui` docs + the live `airflow.cfg`) originally surfaced exactly three
config-driven wins that are high-value for *this* pipeline, all env-var driven and zero-risk:

1. **`instance_name = "DAKP"`** — browser tab title + homepage heading. Instantly distinguishes this
   orchestrator from any other Airflow on the host. Trivial, pure gain.
2. **`theme` (custom brand palette; removed)** — a custom teal palette was considered for buttons,
   active states, and accents, but unique colors are intentionally not retained.
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

Keep the two retained UI env keys inside the existing `_airflow_env()` `env.update({...})` block.
The custom theme constant and `AIRFLOW__API__THEME` env key are intentionally omitted.

## Files to modify

- `src/dakp_pipeline/cli.py`
  - `_airflow_env()`: retain the two non-palette keys in the `env.update({...})` dict:
    - `"AIRFLOW__API__INSTANCE_NAME": "DAKP"`
    - `"AIRFLOW__API__DEFAULT_WRAP": "True"`
  - remove the `_DAKP_UI_THEME` constant and `"AIRFLOW__API__THEME"` assignment.
- `tests/unit/test_cli.py`
  - keep assertions for the two retained UI keys and assert that the theme env key is absent. Follows
    the existing boundary convention (call `cli._airflow_env` directly with tmp paths; no Airflow/Go
    needed).

## Reuse

- `json` — already imported in `cli.py` (used for `COORDINATORS`/`CONFIG_VARIABLE`).
- `_airflow_env()` — the single existing config-injection point; no new mechanism introduced.

## Steps

- [x] 1. Keep the `INSTANCE_NAME` and `DEFAULT_WRAP` settings in `_airflow_env()`.
- [x] 2. Remove the `_DAKP_UI_THEME` constant and `AIRFLOW__API__THEME` assignment.
- [x] 3. Update `test_airflow_env_carries_ui_customization` to pin the retained keys and theme removal.
- [x] 4. Run `uv run pytest tests/unit/test_cli.py -q -o addopts=""` and `uv run ruff check src tests`.

## Verification

- Unit: `uv run pytest tests/unit/test_cli.py -q -o addopts=""` passes, including the theme-absence
  assertion.
- Lint: `uv run ruff check` clean.
- Live (optional, manual): `uv run dakp down && uv run dakp up --small --detach`, then open
  `http://127.0.0.1:8090` and confirm: tab title/heading say **DAKP**, Airflow's default colors remain,
  and a Go extract task's log wraps instead of horizontal-scrolling.
