# Fix: transient DNS/network failure kills `acquire_dailymed` (and all acquire tasks)

## Context

`acquire_dailymed` failed on 2026-08-10 with:

```
URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
gaierror: [Errno -3] Temporary failure in name resolution
```

Trace through the failure:

1. Index fetch succeeded at 19:35:25 (`dailymed.nlm.nih.gov` resolved fine).
2. Releases 1–6 (`dm_spl_release_human_rx_part1..6.zip`) hit the 7-day freshness gate
   (`fresh_skip = true`) — **zero network I/O** — and expanded ~46k SPL docs from cache.
3. Release 7 (`dm_spl_release_human_otc_part1.zip`) was past the freshness window, so
   `_download_one` fell through to a real download. The very first `urlopen` in
   `_conditional_download` hit a **transient DNS failure** (`EAI_AGAIN`, errno -3:
   the resolver got no answer from its upstream nameserver).
4. There is **no retry anywhere**:
   - `sources/dailymed.py::_conditional_download` — one bare `urllib.request.urlopen`
   - `sources/faers.py::_http_get_text` / `_http_download` — same
   - `sources/drugsfda.py::download_drugsfda_zip` — same
   - the Airflow `@task` definitions in `dags/dakp_build.py` set no `retries`

A single DNS hiccup (~seconds) destroyed a whole DAG run after 3.5 min of work.

The pipeline is designed to be idempotent and content-addressed, so retrying is always safe:
stale staging files are removed before each request (`_conditional_download` unlinks `dest`
first), and re-ingesting identical bytes is a cache hit.

## Approach

**Shared stdlib-only transient-retry helper + wire it into all three fetchers.**

1. New module `src/dakp_pipeline/sources/http_retry.py` (stdlib only — the codebase
   deliberately avoids `requests`):
   - `retry_transient(fn, *, attempts=_RETRY_ATTEMPTS, base_delay=_RETRY_BASE_DELAY, ...)`
     calls `fn()`, retrying on transient failures with exponential backoff + jitter.
   - Transient = `URLError` whose wrapped reason is `socket.gaierror` (DNS),
     `socket.timeout`/`TimeoutError`, `ConnectionError`/`ConnectionResetError`,
     `OSError` network errnos, `http.client.RemoteDisconnected`/`IncompleteRead`;
     plus `HTTPError` with status 408/429/500/502/503/504.
     Non-transient errors (HTTP 404 etc.) propagate immediately.
   - Defaults (TBD with user): ~5 attempts, delays 1s→2s→4s→8s→16s (± jitter) ≈ ≤ 1 min extra.
   - Sleep function injectable for tests; retry attempts logged via `stats(logger, ...)`
     with the fetcher's event prefix.
2. Wire into `dailymed._conditional_download` — wrap the urlopen+stream body in a retry
   closure. Each attempt starts clean (dest unlinked), 304 handling unchanged.
3. Wire into `faers._http_get_text` / `_http_download` and
   `drugsfda.download_drugsfda_zip` — identical failure class, one-line wrapping.
4. Unit tests for the helper (succeeds-after-N-failures, non-transient propagates, attempt
   cap) + a dailymed test that `_conditional_download` survives a `gaierror` then succeeds
   (monkeypatch `urllib.request.urlopen`, matching existing seam conventions).

### Open questions (see chat)

- Scope: dailymed-only vs all three fetchers via the shared helper?
- Retry budget: attempts/backoff defaults?
- Airflow task-level `retries` on acquire tasks as defense-in-depth?
- Range-based resume for multi-GB ZIPs: include now or defer?

## Files to modify

- `src/dakp_pipeline/sources/http_retry.py` — **new** shared retry helper
- `src/dakp_pipeline/sources/dailymed.py` — wrap `_conditional_download` request body
- `src/dakp_pipeline/sources/faers.py` — wrap `_http_get_text` / `_http_download`
- `src/dakp_pipeline/sources/drugsfda.py` — wrap `download_drugsfda_zip`
- `tests/unit/test_http_retry.py` — **new** helper tests
- `tests/unit/test_dailymed_source.py` — retry-survives-gaierror test
- `src/dakp_pipeline/dags/dakp_build.py` — (if approved) `retries=` on acquire tasks

## Reuse

- `dakp_pipeline.logging_setup.stats` / `logger` — retry narration (existing one-stat-per-line style)
- Existing monkeypatch seams: tests already patch `urllib.request.urlopen` and module-level
  `fetch` / `_download_full_release` (see `tests/unit/test_dailymed_source.py`)
- `_conditional_download` already unlinks `dest` before each request → retries need no
  partial-file cleanup

## Steps

- [ ] Add `http_retry.py` with `retry_transient` + transient-error classification
- [ ] Wrap `dailymed._conditional_download` (preserve 304 semantics)
- [ ] Wrap `faers` and `drugsfda` download paths
- [ ] Log each retry attempt (`attempt`, `delay_s`, `error`) under each fetcher's event
- [ ] Unit tests: helper behavior + dailymed gaierror recovery
- [ ] (if approved) Airflow `retries`/`retry_delay` on the four acquire tasks

## Verification

- `uv run pytest tests/unit/test_http_retry.py tests/unit/test_dailymed_source.py -q`
- `uv run pytest tests/unit -q` (no regressions in faers/drugsfda tests)
- Ruff/lint per `.pre-commit-config.yaml`
- Manual: re-run the DAG (`acquire_dailymed`) on the machine that failed; transient DNS
  should now log `retry attempt=N` and proceed.
