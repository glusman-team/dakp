---
name: dakp-release
description: "Cut a DAKP release: version sync, remote gates on wenceslaus, commit with the authorized --no-verify, CI-owned GitHub tag, and publication of the matching KGX artifacts to the Hugging Face dataset SkyeAv/drug-approvals-kp. Use for any DAKP release request alongside the global release skill; this runbook adds the dataset step and the local hook-bypass overrides."
---

# DAKP release and Hugging Face publication

Load the global `release` skill and `.pi/skills/dakp-verify` first. This skill
supplements them; where they conflict, these DAKP-specific rules win, and the
user has explicitly authorized the deviations marked OVERRIDE.

## Overrides of the global release skill

- OVERRIDE (hooks): once the user approves the release commit, commit with
  `git commit --no-verify`, and push with `--no-verify` if a pre-push hook is
  installed. Reason: the hooks would run pyright plus the full pytest suite on
  a 30GB laptop that OOMs on them. This waives local hooks only; all gates run
  remotely on wenceslaus first, with recorded evidence.
- OVERRIDE (location): every gate, build, and validation runs on wenceslaus via
  `dakp-verify`. Never locally.
- ADDITION (scope): a DAKP release is not done when the code version is tagged.
  It is done when the matching KGX artifacts are published to
  `SkyeAv/drug-approvals-kp` and verified. Include the dataset step in the plan
  and approval; never publish without explicit approval.

Shared-tree caution: if the checkout is dirty with another agent's work, or
main has moved, stop before checkout, fast-forward, staging, or committing.
Never stash or commit someone else's changes as part of a release.

## Code release

1. Release model: detect from CI, do not assume. Currently
   `.github/workflows/tag-version.yml` watches `pyproject.toml` on pushes to
   main and creates the GitHub `v<version>` tag itself (version-push model).
   NEVER create, move, or delete the GitHub tag by hand; after the approved
   push, verify the tag appeared (`git ls-remote --tags origin v<version>`).
   There is no PyPI publish.
2. Version carriers: bump every live carrier in one commit, or none:
   `pyproject.toml` `[project] version`, the `dakp-pipeline` entry in
   `uv.lock` (re-lock and confirm the lock diff touches only the self entry),
   `src/dakp_pipeline/__init__.py` `__version__`, `CITATION.cff` `version:`,
   and the generated `tables/graph.yaml` (regenerate it; it embeds the version
   and the `RIG_ARTIFACT_BASE_URL` `<version>` path). Leave historical version
   strings in changelogs, benchmarks, and docs untouched. The repo keeps no
   CHANGELOG.md; do not invent one.
3. Gates: run the full `make check` plus `make precommit` equivalent on
   wenceslaus per `dakp-verify`, on the exact release tree. A red gate stops
   the release.
4. Approval: show the complete diff (all version carriers), the commit message,
   and the full dataset publication plan (below) for explicit approval.
5. Act: stage exactly the release files, `git commit --no-verify` (OVERRIDE),
   push to main (the push IS the release act in this model), then verify CI
   tagged: `gh run list` plus `git ls-remote --tags origin "refs/tags/v<version>"`.

## Matching dataset build

The HF dataset publishes one directory per released version with exactly three
files: `DRUG_APPROVALS_KP_<version>.nodes.ndjson`,
`DRUG_APPROVALS_KP_<version>.edges.ndjson`, and
`DRUG_APPROVALS_KP_<version>.RIG.yaml`. No TSV views, no foldreports, no NER
bundles unless separately requested.

- Reuse finished artifacts ONLY after verifying they were produced from the
  release revision: check the build's source revision, embedded graph-config
  version, RIG URLs, and validation report match `<version>`. File names alone
  are not provenance. Never relabel old artifacts as a new version.
- If no matching artifacts exist, add the full production build on wenceslaus
  to the approval scope: production workdir, `dakp up --fullmap
  <absolute-path-to-fullmap.redb>`, the tablassert version pinned by this
  checkout, and the fullmap schema it requires. Building without approval is a
  STOP; report "no matching-version artifacts exist" and wait.
- KGX/Translator contract validation and RIG URL checks run remotely
  (`dakp-verify` rules apply). Collect real numbers: node/edge counts, bytes,
  SHA256 or MD5, Biolink version, validation status.

## Hugging Face publication (order matters)

Target `SkyeAv/drug-approvals-kp`, `--repo-type dataset`. Auth check first
(`hf auth whoami`); never print tokens. Keep heavy statistics on wenceslaus;
upload from there or from a small local staging dir, never by pulling the
whole dataset to the laptop.

1. Verify the three artifacts (contract validation, counts, hashes, RIG). If a
   `<version>/` path or `v<version>` tag already exists on the dataset, compare
   content; any mismatch is a STOP. Published versions are immutable: never
   overwrite, move, or delete existing version files or tags.
2. Stage exactly the three files in a staging directory (hardlink or copy; a
   directory upload avoids glob-filter ambiguity), then upload the version
   directory:
   `hf upload SkyeAv/drug-approvals-kp <staging>/<version> <version> --repo-type dataset --commit-message "Add DAKP <version> KGX (<biolink>, <N> nodes / <M> edges)"`
   with real counts, matching the established per-release commit message style.
3. Update the dataset README card from its CURRENT content (download it first):
   - Replace the card's TWO `configs:` entries with the new release:
     `<version>_edges` (carrying `default: true`) and `<version>_nodes`, with
     `data_files` pointing at the new `<version>/` NDJSON pair and both feature
     lists declared inline. Only the latest release is browsable, so nothing
     aliases them and no YAML anchors are needed.
   - Do NOT keep configs for earlier releases: their directories stay in the repo
     as path downloads, never parquet-converted and never in the subset dropdown,
     which is what keeps the dropdown reading newest-edges then newest-nodes. The
     HF viewer orders subsets "default first, then alphabetical" (per HF's
     data-files-config docs) and `_edges` sorts before `_nodes`, so two configs
     need no rank prefix; card YAML order alone does NOT control the order.
   - `default: true` on `<version>_edges` also makes plain `load_dataset` load
     that config. Exactly one config carries the flag at any time.
   - Update the "Latest release" line, and add the `<version>` row at the TOP of
     the Releases table and the checksums table. Extend Schema/Loading only if
     the schema actually changed. Match the card's existing structure and ASCII
     style; keep all historical releases and their URLs intact. Then upload it:
   `hf upload SkyeAv/drug-approvals-kp <staging>/README.md README.md --repo-type dataset --commit-message "..."`
   Verify afterwards that the viewer opens `<version>_edges`:
   `datasets-server.huggingface.co/splits?dataset=SkyeAv/drug-approvals-kp` must
   return exactly `["<version>_edges", "<version>_nodes"]` in that order, and
   `first-rows?dataset=...&config=<version>_edges&split=train` must return 200
   with the declared features. Renaming configs makes the datasets-server
   re-convert, so `/splits`, `/is-valid`, `/parquet` and `first-rows` can answer
   HTTP 500 `server is busier than usual` for a few minutes, and `/is-valid` can
   report all-false until conversion finishes; retry instead of reading a 500 or
   a false flag as a card error.
4. Tag last, pinning the completed release, via the Python API (the `hf` CLI
   has no tag command):
   `HfApi().create_tag(repo_id="SkyeAv/drug-approvals-kp", tag="v<version>", repo_type="dataset", tag_message=...)`.
   If the tag exists, verify it points where expected; never delete or move it.
5. Verify everything, with evidence: the three files resolve under
   `https://huggingface.co/datasets/SkyeAv/drug-approvals-kp/tree/main/<version>`
   and match the `artifact_base_url` the release's RIG advertises; spot-check
   one downloaded file hash against the source; the card renders the new
   configs; `v<version>` exists on the dataset; the code tag exists on GitHub.

## Failure handling

If any dataset step fails after the code is tagged, report the partial state
explicitly: which steps completed, what is missing. Resume only the remaining
approved steps; do not cut a new version to work around a partial upload, and
do not mark the release complete while its dataset is absent or unverified.
