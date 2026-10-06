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
