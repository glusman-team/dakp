# Foreman Prompt — Ruthless Simplification of DAKP

## Context

DAKP was rebuilt end-to-end with AI agents and accumulated dead code + overengineering. The goal is
a **steering prompt for the `foreman` skill** that reduces the repo to its simplest honest form:

- **One pipeline**, no `mock`/`sample`/`prod` profiles.
- **No machine naming** (`wenceslaus` / "laptop" / "workstation" / `/local_raid1` / thread-memory sizing).
- **One run surface** + one clear "how to run" README; delete the `docs/` sprawl.
- **Flatten** the directory structure, delete dead code, question every abstraction.
- **Runs on this laptop, fast** — drive the refactor from a live end-to-end run here; big downloads happen once.

## Feasibility & performance on this laptop (honest assessment)

**It runs here.** This machine is capable and the environment is already provisioned: AMD Ryzen AI 9 HX 370
(**12 cores / 24 threads**, up to 5.1 GHz), **30 GiB RAM** (+41 GiB swap), **208 GiB free** on NVMe, and
`uv sync` is done — `airflow 3.3.0`, `polars`, `gliner`, `tablassert`, `cyclopts` all import. The foreman
should **actually run the pipeline on this laptop and use the live run as the refactor feedback loop**
(run → observe → simplify → re-run), not edit blind.

**Not a bad idea — but scope it honestly** (the requested pushback):
- Comfortably fast here: bounded real acquisition, the **Go extraction** (CPU-bound — parallelize across all
  24 threads), NER, aggregation, config generation, and the mock pipeline.
- **Constraint 1 — RAM (30 GiB) vs the 49 GiB fullmap.** The old "laptop can't do it" claim was about
  *building* the fullmap (~120 GiB). A **prebuilt** fullmap now exists on the Desktop, so the laptop only
  *runs against* it — but a real `build-kg --fullmap` may still pressure 30 GiB. Keep dev runs **bounded** so
  the working set stays small; let the fullmap memory-map/swap.
- **Constraint 2 — disk (208 GiB free) vs unbounded downloads.** A full multi-TB prod build is server-scale,
  not laptop-scale. Bounded real runs fit fine, and the BLAKE3 content-addressed store means **big downloads
  happen once** and are reused — preserve that idempotency.
- **"As fast as possible" is mostly an I/O game**, not CPU: download once (idempotent cache), fan extraction
  out over all cores, avoid redundant parquet reads, and keep dev iterations bounded. Profile before
  optimizing; never trade correctness or simplicity for a micro-optimization.
- **The elegant fix: one scoped-build flag.** Add a single CLI flag (e.g. `uv run dakp up --small`) that runs
  the **full** pipeline on a tiny data subset — download only ~1 FAERS quarter + 1 DailyMed release (bound the
  *download*, not just the processing), run every stage, and build a **minimal KG** (far fewer edges) against
  the Desktop fullmap. This is the dev feedback loop: fast, low-RAM, low-disk, and it exercises the real
  end-to-end path. It bounds **scope only** — it is NOT a profile (no fixture-switching, no deferred handoff,
  no thread/memory sizing); the pipeline behaves identically, only the data volume changes.

## ⚠ Another agent is already working in this worktree (COORDINATE, don't collide)

An active pi session **`DAKP-rebuild`** (`0825a214977b454d`) is editing the **main worktree right now**
(uncommitted changes to `config.py`, `dags/dakp_build.py`, `acquire.py`, `runtime.py`, `tablassert/*`,
tests). It is executing **`plans/cli-replaces-scripts.md`**:

- **Building a single `uv run dakp` CLI (cyclopts)** that replaces the `Makefile` + `scripts/dakp_up.sh`
  / `dakp_down.sh` + `.envrc`/direnv. → This IS the "one way to run it." **Keep it; build on it; do not
  recreate a competing run surface.**
- **Stripping the vestigial fullmap/ontology download path** (`acquire_ontologies`, `fullmap_source`,
  `ontology_sources`, the `.fullmap` stub) — fullmap becomes an explicit `--fullmap <path>` the caller supplies.

**Conflicts with the user's new directives (the foreman must finish the job the CLI plan leaves short):**
- The CLI plan **keeps profiles** (`uv run dakp up [profile]`, default `mock`) → user wants profiles GONE.
- The CLI plan **keeps machine naming** (`wenceslaus`, `/local_raid1`) → user wants it GONE.
- The CLI plan **keeps the pure-Python `run_pipeline` harness** → user wants ONLY the Airflow DAG harness.

**Sequencing hazard:** the other agent has uncommitted edits in the shared tree. Do NOT run a concurrent
foreman/Ralph that touches the same files. **Resolved (decision 1a):** let `DAKP-rebuild` commit its CLI
work first, then run the foreman on top — the prompt's PRECONDITION enforces this.

## The inviolable core (must survive)

`acquire → extract (native Go workers) → NER → aggregate → Tablassert KGX handoff`, producing 3 assertion
TSV tables (treats / observed-use / contraindication), semantically equivalent to legacy DAKP
(`tests/integration/test_semantic_equivalence.py`). The **Airflow DAG `dakp_build`** + **native Go workers**
are core and stay.

## Overengineering map (what the foreman targets)

| # | Target | Action |
|---|--------|--------|
| 1 | **Profiles** (`config.py` `Profile`/`load_profile`/`_PROFILE_TABLE`; 161 tokens / 26 files; the CLI `[profile]` positional; 3 make targets) | DELETE. One pipeline. Offline/fixtures becomes a *test* concern (monkeypatched fetchers), not a user-facing profile. Real handoff is triggered by passing `--fullmap`; a new `--small` flag bounds *scope* for dev (not a profile). |
| 2 | **Machine naming / sizing** (32 `wenceslaus` hits; `threads`/`memory_budget_gb`; `/local_raid1`; `docs/wenceslaus-runbook.md`) | DELETE all host-specific sizing + runbooks. One pipeline, one "how to run." |
| 3 | **Duplicate harness** (`pipeline.py::run_pipeline` duplicates the DAG wiring) | DELETE `run_pipeline`; the **Airflow DAG is the single harness**. Migrate tests that used `run_pipeline` (test_mock_pipeline, test_prod_smoke, semantic_equivalence) to drive the DAG or call stage functions directly. |
| 4 | **`docs/` sprawl** (8 files) + `plans/` (3 files) | DELETE `docs/`; fold everything a user needs into **one clear README** ("what it is" + "how to run"). |
| 5 | **`ref/legacy`** (192K / 20 files "for audit") | DELETE (lives in git history). |
| 6 | **Directory nesting / single-file packages** | FLATTEN. Collapse needless packages. |
| 7 | **BLAKE3 store / ArtifactRef / manifests / xcom** | QUESTION: do stages need content-addressed handles, or just files in a workdir? Prefer the simpler thing that still passes tests. |
| 8 | **100% branch-coverage gate** | **KEEP** `fail_under=100` (user decision). Delete tests + covered branches together with dead code; never drop the gate. |
| — | **Native Go workers** (`go/`, 7.3k LOC) | **KEEP** (user decision — they're core). |
| — | **Airflow DAG** | **KEEP** as the sole orchestrator/harness. |
| — | **`uv run dakp` CLI** (in-flight) | **KEEP** as the single run surface; extend it (drop the profile positional). |

## Decisions (resolved with the user)

1. **Sequencing = (a):** let `DAKP-rebuild` finish + **commit** the CLI first, then run the foreman on top
   of it. → Hand this prompt to the user only once the CLI work is committed; the prompt assumes the `dakp`
   CLI already exists on the branch.
2. **Offline = (a):** no offline flag and no profile. The one pipeline runs **real**; offline execution is
   purely a test-suite concern (monkeypatched fetchers + fixtures). No `--fixtures` affordance.
3. **Coverage = keep:** maintain `fail_under = 100` branch coverage. Delete tests + covered branches together
   with dead code (net-neutral); never drop the gate.

## Available for real end-to-end testing

A real prebuilt fullmap lives at **`/home/skyeav/Desktop/fullmap/data/fullmap.redb`** (49G; sharded
`.s0`–`.s15.redb` alongside). Passing it via `--fullmap` lets the pipeline run a **real** Tablassert
handoff (full KGX build) end-to-end — not just the deferred path — so the foreman can verify the complete
path during development.

## The foreman prompt (finalized — ready to paste)

```
foreman — in /home/skyeav/Code/ISB/DAKP, finish simplifying the pipeline to its simplest honest form.

PRECONDITION: the `uv run dakp` CLI from plans/cli-replaces-scripts.md has already landed (a prior agent
built it: a single cyclopts CLI replacing the Makefile + scripts/*.sh + .envrc/direnv; the fullmap is now a
caller-supplied `--fullmap <path>`). Confirm that work is committed on the branch before starting; if any of
it is still uncommitted, STOP and report instead of editing concurrently. That CLI is the single run
surface — BUILD ON IT, do not recreate it.

Dispatch ONE Ralph sub-orchestrator for this (coherent single-repo build), not a fan-out of shallow workers.

THE CORE (must survive, semantically equivalent to legacy — test_semantic_equivalence.py stays green):
acquire → extract (native Go workers) → NER → aggregate → Tablassert KGX handoff → 3 assertion TSVs
(treats / observed-use / contraindication). KEEP the Airflow DAG (dakp_build) and the native Go workers.

DEV FULLMAP: a real prebuilt fullmap is available at /home/skyeav/Desktop/fullmap/data/fullmap.redb.
Use it to exercise a REAL Tablassert handoff end-to-end during development (uv run dakp up --fullmap
/home/skyeav/Desktop/fullmap/data/fullmap.redb) so you verify the full KGX build path, not just the
deferred handoff. It is large (~49G) — a bounded verification run is fine.

RUN ON THIS LAPTOP + DRIVE THE REFACTOR FROM A LIVE RUN:
- The environment here is ready (deps installed, airflow 3.3.0; 24-thread Ryzen, 30G RAM, NVMe). FIRST get
  the pipeline running end-to-end on this laptop to establish a green baseline, THEN refactor against that
  live feedback loop (run -> observe -> simplify -> re-run). Do not edit blind.
- Big downloads must happen ONCE: keep acquisition content-addressed/idempotent (BLAKE3) so reruns reuse the
  cache. There is disk space for the downloads.
- Make the pipeline run AS FAST AS POSSIBLE within these constraints: parallelize the Go extraction across
  all cores, avoid redundant parquet reads / repeated I/O, keep dev iterations bounded. The bottleneck is
  I/O + fullmap resolution, not CPU — profile before optimizing; never sacrifice correctness or simplicity
  for a micro-opt.
- Laptop limits to respect: 30G RAM (the 49G fullmap may not fully fit — keep real runs bounded so the
  working set stays small) and 208G free disk (a full multi-TB prod build is server-scale; bounded real runs
  are the laptop target).

DELETE / SIMPLIFY:
- Profiles: remove mock/sample/prod entirely — config.py Profile/load_profile/_PROFILE_TABLE, every
  profile branch across the ~26 src files, AND the CLI `[profile]` positional. One pipeline that runs
  REAL. There is NO offline flag and NO profile: offline execution is purely a test-suite concern
  (monkeypatched fetchers + fixtures). A real Tablassert handoff is triggered simply by passing
  `--fullmap <path>`; without it the handoff is deferred. A simple integer scope bound (quarters/releases)
  may survive ONLY as the internal knob the new `--small` flag sets — that is scope, not a profile.
- ADD an elegant single CLI flag (e.g. `uv run dakp up --small`) that turns on a SCOPED build for feedback:
  run the FULL pipeline on a tiny subset — download only ~1 FAERS quarter + 1 DailyMed release (bound the
  download itself, not just the processing), run every stage, and build a MINIMAL KG (far fewer edges than
  the full build) against the Desktop fullmap. Use THIS as the fast dev feedback loop. The flag bounds SCOPE
  only — it is NOT a profile: it does not switch to fixtures, does not defer the handoff, and does not size
  threads/memory; pipeline behavior is identical, only the data volume changes.
- Machine naming: remove wenceslaus / laptop / workstation / /local_raid1 / threads / memory_budget_gb
  and docs/wenceslaus-runbook.md. One pipeline, one "how to run."
- Duplicate harness: delete pipeline.py::run_pipeline. The Airflow DAG is the SINGLE harness. Migrate
  tests that used run_pipeline (test_mock_pipeline, test_prod_smoke, semantic_equivalence) to drive the
  DAG or call the stage functions directly, holding 100% branch coverage.
- Docs: delete docs/ entirely; fold what a user needs into ONE clear README (what it is + how to run).
- ref/legacy: delete (it's in git history).
- Flatten the directory structure; collapse single-file packages / needless nesting.
- QUESTION + simplify: the BLAKE3 content-addressed store / ArtifactRef / manifests / xcom — do stages
  need content-addressed handles, or just files in a workdir? Prefer the simpler thing that keeps the
  tests green.
- KEEP the 100% branch-coverage gate (fail_under=100). When you delete code, delete its tests + covered
  branches in the same step (net-neutral); never drop the gate.

APPROACH (two phases):
1. PROPOSE: a written simplification plan. For each item: current state, simpler alternative, cost to
   keep, recommendation. Gate the CRITICAL architectural calls to me before executing.
2. EXECUTE: small green-test increments. End with one README "how to run this one pipeline."

ACCEPTANCE / DoD:
- grep -rniE "wenceslaus|/local_raid1|\bprofile\b|\bmock\b|\bsample\b|\bprod\b|run_pipeline" src/ README.md
  → no profile/machine/harness leftovers (fixtures-only test code aside).
- One way to run: `uv run dakp up [--fullmap PATH]`, documented in the README.
- docs/ and ref/legacy gone; directory structure flattened.
- uv run pytest green; uv run ruff check + uv run pyright clean; `uv run dakp up` produces the 3 TSVs +
  handoff; test_semantic_equivalence.py still passes; no extractor implemented twice; net LOC way down.
- Real handoff verified: `uv run dakp up --fullmap /home/skyeav/Desktop/fullmap/data/fullmap.redb`
  produces KGX output (or a bounded equivalent), confirming the full Tablassert path works.
- The pipeline actually runs end-to-end on this laptop (mock run green; bounded real run with the Desktop
  fullmap produces KGX); reruns reuse cached downloads (no re-download); extraction is parallelized across cores.
- Scoped dev build works: `uv run dakp up --fullmap /home/skyeav/Desktop/fullmap/data/fullmap.redb --small`
  downloads only a small portion, runs every stage, and produces a minimal KG (far fewer edges than the full
  build) — the fast feedback loop.

Conventions: uv project; ruff lint+format; pyright; tests meaningful not exhaustive.
```

## How to deliver

Wait until `DAKP-rebuild` has **committed** the CLI work (decision 1a), then paste the prompt above as a
steering message to the foreman (`foreman` / `/skill:foreman`). The prompt's PRECONDITION tells the foreman
to verify the CLI is committed and to stop rather than collide with uncommitted work.

## Verification (of the foreman run)

- `grep -rniE "wenceslaus|/local_raid1|\bprofile\b|run_pipeline" src README.md Makefile` → empty.
- `uv run pytest -q --cov` green at 100% branch coverage; `uv run ruff check` + `uv run pyright` clean.
- One documented run command produces the 3 TSVs + handoff; `test_semantic_equivalence.py` passes.
- `docs/` and `ref/legacy` removed; Go workers + Airflow DAG intact.
