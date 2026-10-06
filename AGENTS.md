# DAKP Agent Instructions

Directory-local instructions for Pi agents in this repository. They supplement
`~/.pi/agent/AGENTS.md` (global); follow the global instructions everywhere they
are not overridden here. Detailed runbooks live in `.pi/skills/`:

- `.pi/skills/dakp-verify/SKILL.md`: running tests, gates, and builds on wenceslaus.
- `.pi/skills/dakp-release/SKILL.md`: cutting a release and publishing the dataset.

These files are tracked in git, so worktrees pick them up after pulling main.

## Continuity

Before substantial work, recall relevant project context from Hindsight. DAKP
memories exist under both `project:dakp` and `project:DAKP`; recall with
`tags: ["project:dakp", "project:DAKP"]` and `tags_match: "any_strict"`, or
untagged for general hits. When a workflow depends on precedent (releases,
uploads, historical builds), also read the relevant Pi session logs under
`~/.pi/agent/sessions/` and `git log` before acting. Current code, CI, and this
file override stale memories; several retired rebuild-era decisions (optional
dependency extras, mock/sample/prod profiles, old fullmap schema versions) are
obsolete. Retain durable decisions and gotchas with `project:dakp` plus topical
tags. Never retain secrets or transient logs.

## Shared checkout safety

Another agent may be working in this checkout at the same time. Treat the tree
as shared:

- Check `git status`, `git diff`, `git stash list`, and recent `git log` before
  editing, and re-read target files immediately before writing them.
- Before any commit, re-inspect status and stage ONLY files this task owns.
  Never `git add -A` or `git add .`; untracked and modified files may belong to
  the other agent.
- Never stash, reset, clean, checkout-overwrite, rebase, or force-push to make
  room for your own edits. Park conflicting work and ask.
- Leave the other agent's stashes, untracked plans, and runtime state alone.

## Where work runs

This laptop has 30GB RAM and has been OOM-killed by heavy workloads (exit 137).
Per the user's standing instruction, ALL DAKP testing and heavy work runs on
wenceslaus over SSH, never locally:

- Use the `ssh_bash` tool with `host: "wenceslaus"` (and `ssh_read`, `ssh_write`,
  `ssh_edit`, `ssh_copy`) for every test run, quality gate, build, and dataset
  validation. Long jobs go in remote tmux.
- Follow `.pi/skills/dakp-verify/SKILL.md` for the exact procedure: isolated
  remote snapshot, source-identity manifest, tool preflight, bounded workers,
  captured logs and exit status.
- Local work is limited to reading code, editing, and cheap inspections. Do not
  run `pytest`, `pyright`, `make check`, `make test`, or pipeline commands
  locally, and never fall back to local execution because a remote step failed;
  fix the remote step or stop and report.
- Heavy production builds (`dakp up --fullmap`) run on wenceslaus against the
  production workdir, only with explicit scope approval.

## Quality gates and QC metrics

- Coverage: `make test` accumulates coverage across the unit + integration phases with `--cov-append`; the `fail_under = 90` gate (pyproject.toml) evaluates the combined data.  (evidence: ci, pyproject)
- NER accuracy gate: backend/model/inference-config changes must keep the composite backend at
  P/R/F1 1.000 on the gold fixture (`tests/eval/ner_gold.json`, 34 cases / 42 spans). Rerun
  `uv run python tests/eval/benchmark_ner.py` per dakp-verify; record in
  `tests/eval/benchmark_results.json` + `src/dakp_pipeline/ner/BENCHMARK.md`. `tests/eval/*.py`
  scripts are evaluation artifacts: never pytest-collected, never coverage-gated.  (evidence: history, docs)
- Inference-config changes (`chunk_words`, `inference_batch_size`, `compute_dtype`) flip
  knife-edge mentions (fp16: 28/2000 texts). Prove a clean diff with `tests/eval/ab_ner_diff.py`
  before adopting; `compute_dtype` defaults to fp32 (v1.21.0 revert) and both cache tiers key it.  (evidence: history, docs)
- `tablassert build-kg --qc` is always requested but dropped with a warning when the QC runtime (`tablassert[qc]`
  sentence-transformers) is not importable; a green build does not prove QC ran, so check the build log's qc stats event.  (evidence: code: tablassert.py, cli.py)
- `tests/eval/approval_status_audit.py` re-derives `clinical_approval_status` with the shipped
  rule over real production tables, exiting non-zero on any demotion; run it when
  approved-treats or FAERS shaping changes.  (evidence: docs)
- `tests/eval/build_manifest.py` captures per-table row counts + blake3 hashes; diff manifests
  across builds to prove speed/refactor work changed nothing observable downstream of NER.  (evidence: docs, history)
- The three edge families' semantic-equivalence guardrails live in
  `tests/integration/test_semantic_equivalence.py`; keep them green when touching `translator.py`
  or assertion shapers.  (evidence: ci)
- Data-quality fixes use the `quality(<scope>):` commit prefix.  (evidence: history x4)

## Commits

- Conventional commits; stage deliberately; re-check status first (shared tree).
- Use `git commit --no-verify` for DAKP commits: the user authorized bypassing
  hooks so committing can never launch a local workload that OOMs this laptop.
  This waives the hooks, not verification: equivalent checks still run remotely
  per `dakp-verify`.
- The pre-push stage would run pyright plus the full pytest suite locally. If
  hooks are installed when pushing, use `git push --no-verify` for the same
  reason, and say so in the report.
- Do not touch version files, `uv.lock`, or `CITATION.cff` outside an approved
  release flow.

## Repository invariants

- Python package in `src/dakp_pipeline`, native Go extractors in `go/`. Airflow 3
  orchestrates; `uv run dakp` is the CLI entry point.
- `tables/*.yaml` are generated snapshots of the Tablassert configs. Change the
  generators in `src/dakp_pipeline/tablassert.py`, regenerate, and keep them
  byte-equal (tests enforce parity). Never hand-edit the YAML.
- NER emits mention text and offsets only; ontology resolution and KGX
  compilation belong to Tablassert/fullmap. No local KGX fallback compiler.
- Preserve source provenance, deterministic edge identity, clinical approval
  logic, and product-level FAERS case attribution.
- Dev workflows go through the Makefile; CI (`.github/workflows/ci.yml`) runs
  parallel unit/integration shards and a `python-test` combined coverage gate;
  other jobs run `make lint fmt-check typecheck test-go vet precommit`. The
  coverage gate lives in `pyproject.toml` (`fail_under`, currently 90); older
  README/history claims of 100% are stale. Read current config rather than
  trusting remembered test counts.
- Releases are more than a version bump: every release publishes its
  matching-version KGX artifacts to the Hugging Face dataset
  `SkyeAv/drug-approvals-kp`. Use `.pi/skills/dakp-release/SKILL.md`; GitHub
  `v<version>` tags are created by CI (`tag-version.yml`), never by hand.
- On the dataset card, the LATEST release is what visitors see first: its
  `<version>_edges` config carries `default: true` (the HF viewer orders subsets
  "default first, then alphabetical"; card YAML order does not control this),
  and `configs:` entries plus the Releases/checksums tables are kept newest
  first. When adding a version, insert its configs at the TOP of the list and
  MOVE `default: true` from the previous latest's `_edges` config to the new
  one; older subsets stay one click away in the subset picker.
