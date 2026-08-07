# DailyMed: skip re-download while the stored release is < 1 week old

## Context

Every time NLM cuts a new DailyMed full release, `acquire_dailymed` re-downloads
**all** of it (~40 min). Root cause (verified live 2026-08-07):

- The "Full Releases" section uses **fixed filenames**
  (`dm_spl_release_human_rx_part1..6.zip`, OTC parts, …) and NLM **replaces those
  S3 objects in place** for every new release (`part1.zip`: 3.2 GB,
  `Last-Modified: Fri, 07 Aug 2026`). The conditional-GET logic in
  `sources/dailymed.py` works correctly — the bytes genuinely changed, so every
  part returns 200 and the whole snapshot streams down again.

Requested behavior: tolerate up to a week of staleness. **If the release already
in the store was fetched within the last week, keep using it — zero ZIP
downloads.** No periodic-update/delta machinery (earlier draft dropped per user
direction).

Note on the earlier dedup ask: with one release held at a time — and each release
being a snapshot (exactly one document version per `setId`) — the
`listCases.pl`-style latest-wins dedup is a no-op here: the stored dump *is* the
latest entry for every key. Extractors stay untouched.

## Approach

Add a **freshness gate** to `_download_full_release` in
`src/dakp_pipeline/sources/dailymed.py`:

1. Discover releases exactly as today (index conditional GET + `_parse_release_zips`
   + `release_limit`). The index page fetch stays (small, 304-cached).
2. Per release URL, resolve the stored alias (`dailymed/<name>`) via
   `ArtifactStore.cached_ref` + `read_manifest` and read
   `manifest.source.retrieved_at` — **already recorded at ingest** by
   `_download_one` (`SourceBlock.retrieved_at`). No new state files.
3. Gate, per release:
   - cached ZIP present **and** `retrieved_at` within `max_age` (default **7
     days**, run param `dailymed_max_age_days`) **and** not `force` →
     **skip the network entirely** for that release: reuse the cached ZIP through
     the *existing* 304-style path (`_expand_release_zip(cached.uri, …)`), which
     re-emits the member refs with no download. Log loudly:
     `release fresh (age Xd < 7d), skipping download`.
   - missing from store, unparseable/absent `retrieved_at`, older than `max_age`,
     or `force` → existing conditional-GET download flow (unchanged).
4. Freshness is keyed on **our `retrieved_at`** (when we fetched it), not the
   server publish date: no extra HEAD requests, and it guarantees "the data in
   hand is at most a week old" even if NLM publishes mid-week.
5. `force=True` bypasses the gate (same convention as FAERS `download_quarter` /
   NER model cache).

Net effect: runs within a week of the last acquisition download **nothing**
(beyond a usually-304 index fetch); the ~40 GB re-download happens at most once
per week — or whenever `force` / a fresh workdir demands it.

**Out of scope (deliberately):** release-part scope (rx vs otc vs all), periodic
updates, extract-side changes, Go bundle.

## Files to modify

- `src/dakp_pipeline/sources/dailymed.py` — the gate + fresh-reuse path.
- `src/dakp_pipeline/runtime.py` — forward `dailymed_max_age_days` from the
  `dakp_config` Variable into `ctx.params` (like `quarter_limit`/`release_limit`).
- `src/dakp_pipeline/cli.py` — add `dailymed_max_age_days: 7` to the config dict
  written by `dakp up`.
- `tests/unit/test_sources_edge.py` — new gate coverage (the existing
  conditional-GET/304 tests remain the *stale* path).
- `tests/unit/test_dag_downloads.py` — `build_context_from_config` forwarding.
- `tests/unit/test_cli.py` — config dict carries the new key.
- `README.md` — one line in the acquire description.

## Reuse

- `ArtifactStore.cached_ref` / `read_manifest` (`io/artifact_store.py`) —
  alias → stored ZIP path + manifest `source.retrieved_at`.
- The existing HTTP-304 branch of `_download_one` (`_expand_release_zip` on the
  cached ZIP) — the fresh path reuses exactly this expansion, no new code path
  for member refs.
- `force` run-param convention from `sources/faers.py` (`download_quarter`).

## Steps

- [ ] `_release_age_days(store, alias)` helper: resolve cached ZIP + manifest,
      parse `retrieved_at` (`datetime.fromisoformat`), return age; missing file /
      missing or malformed timestamp → `None` (= stale).
- [ ] Gate in `_download_full_release` / `_download_one`: fresh ⇒ cached-ZIP
      re-expansion, no `_conditional_download` call; structured log event with age.
- [ ] Param plumbing: `runtime.build_context_from_config` + `cli.py` config dict
      (`dailymed_max_age_days`, default 7; `None`/<=0 ⇒ always re-check).
- [ ] Tests: fresh skip makes **zero** ZIP HTTP calls; stale ⇒ conditional GET as
      today; one-stale-one-fresh ⇒ only the stale one hits the network; `force`
      bypasses; absent/malformed `retrieved_at` treated as stale; missing cached
      ZIP treated as stale; param forwarding (runtime) + config key (cli).
- [ ] README touch-up.

## Verification

- `uv run pytest -q --cov` (100% branch coverage gate), `uv run ruff check`,
  `uv run ruff format --check`, `uv run pyright`; `cd go && go test ./...`
  (untouched, sanity).
- Live two-run check: `uv run dakp up --small` twice within the same week. The
  second run's `acquire_dailymed` log must show the fresh-skip event with **no**
  release downloads, and the task should drop from minutes to seconds (cached-ZIP
  re-expansion only).
