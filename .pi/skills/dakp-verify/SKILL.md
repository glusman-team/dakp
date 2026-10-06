---
name: dakp-verify
description: "Run DAKP tests, quality gates, and builds on wenceslaus over SSH. Use before claiming any DAKP code change is verified, whenever tests or `make check`/`make test` are requested, and instead of running anything heavy on the laptop. Never run pytest, pyright, or pipeline commands locally; this laptop OOMs on them."
---

# DAKP remote verification on wenceslaus

## Hard rules

- Every pytest, pyright, ruff-format, Go test/vet, build, or pipeline run
  executes on wenceslaus via the `ssh_bash` tool (`host: "wenceslaus"`), or in a
  remote tmux session for long jobs. Never on the laptop, not even one file.
- Never edit or overwrite the existing remote checkouts `~/Code/dakp` or
  `/local_raid1/sgoetz/CODE/DAKP`: they hold production artifacts, real data,
  and possibly another session's work. Verify in an isolated snapshot directory.
- A remote infrastructure failure (missing tool, broken uv, full disk) is a STOP:
  report it. It is never a reason to run the workload locally instead.
- Evidence or silence: report commands, exit codes, and captured output. A
  dropped SSH connection proves nothing; re-check the recorded exit status.

## Procedure

1. Snapshot locally without disturbing the tree: record `git rev-parse HEAD`,
   `git status --porcelain`, and the diff of owned changes. Assemble the exact
   source set for verification (tracked files at HEAD plus owned uncommitted
   changes). Do not copy `.venv/`, `tmp/`, caches, coverage output, or unrelated
   untracked files.
2. Transfer the snapshot to a unique remote directory, e.g.
   `~/dakp-verify-<date>-<slug>/` under the wenceslaus home, via `ssh_copy` or
   `tar` over `ssh_bash`. Never rsync with `--delete` into any shared path.
3. Verify source identity on the remote copy: compare a hash manifest
   (`find . -name '*.py' -o -name '*.go' -o ... | sort | xargs sha256sum`) and
   `git rev-parse HEAD` if `.git` was copied, against the local values. Record
   both; HEAD alone does not identify uncommitted changes.
4. Preflight remote tools and capacity before running anything:
   - `free -g; df -h .; nproc`
   - `uv --version` (a snap-installed uv has failed before with "cannot create
     user data directory"; try `bash -lc 'command -v uv'` for a login-shell
     PATH, and STOP and report if no working uv exists)
   - `python3 --version` and the repo's `.python-version`
   - `go version; gofmt --help` (Go has been absent from PATH; `make test-go`,
     `make vet`, and `make fmt-check` need it)
   - `locale`: wenceslaus defaults to `LANG=en_US` (ISO-8859-1). Export
     `LC_ALL=C.UTF-8` (or `PYTHONUTF8=1`) for every command; without it
     `import airflow` dies on `yaml ReaderError: unacceptable character
     #x0080 ... position 111403` (latin-1 decode of the UTF-8 em dash in
     airflow's shipped `config.yml`). Not a broken file or filesystem.
   - `tmux -V` for long jobs
   Stop and report on any missing or broken tool. Do not install tools without
   explicit approval.
5. Environment: in the snapshot directory run `uv sync --frozen --group dev`
   (matches CI's dev-group sync; never reuse the laptop's venv). Confirm
   `uv lock --check` passes.
6. Run the gate. Prefer the CI-identical target:
   - `make check` (lint, fmt-check, typecheck, test, test-go, vet)
   - `make precommit` when hook wiring itself matters
   For targeted iteration, run a narrow slice first, e.g.
   `uv run pytest -o addopts='' -q -n 0 --no-cov tests/unit/test_x.py::test_y`,
   then still run the full gate before declaring done.
7. Bounded parallelism: wenceslaus has 80 cores; `-n auto` from addopts would
   fan out to all of them while integration tests also spawn subprocesses. If
   `make test` wedges or oversubscribes, rerun the two phases with explicit
   caps and report the deviation, e.g.:
   - `uv run pytest tests/unit -n 24`
   - `uv run pytest tests/integration -n 0 --cov-append`
   Known trap: the Makefile comment claims integration runs sequentially, but
   the recipe does not pass `-n 0` while addopts defaults to `-n auto`. If you
   deviate from `make test`, say so explicitly; never present a bounded rerun
   as the unmodified CI gate. Never weaken tests, timeouts, or the coverage
   gate to get green.
8. Long runs: launch in a unique remote tmux session via `ssh_bash`, redirect
   output to a log file, and write the exit status to a file
   (`cd <dir> && ... ; echo $? > <dir>/exit-status`). Poll the log and status
   file; the jump path may idle-kill SSH masters, so never rely on a live
   connection surviving.
9. Re-collect evidence at the end: exit codes, pass/fail/skip counts, coverage
   number if the gate ran, and the log path. If remote fixes (format, lint)
   were applied, apply the same fix deliberately to owned local files and note
   it; never copy the remote tree back wholesale.
10. Leave the snapshot directory in place until cleanup is agreed; do not delete
    remote data you did not create.

## Reporting

State: remote host and directory, source identity (HEAD + manifest match),
environment versions, exact commands, exit codes, test counts, coverage result,
and log locations. List anything unrun and why. `git commit --no-verify` (the
user-authorized hook bypass) does not replace any of this evidence.
