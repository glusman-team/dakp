---
name: Bug report
about: Report a reproducible problem in the DAKP pipeline
title: "[Bug]: "
labels: bug
assignees: ""
---

## Description
<!-- Briefly describe the problem and its impact. -->

## Steps to Reproduce
1. <!-- First step, including any setup or fixture details. -->
2. <!-- Next step leading to the failure. -->
3. <!-- First observable failure or incorrect result. -->

## Command or Run Configuration
<!-- Include the exact command, such as `uv run dakp up --small`, and relevant flags. -->

## Pipeline Stage
<!-- Check the stage where the problem appears, if known. -->
- [ ] Acquisition (DailyMed, Drugs@FDA, or FAERS)
- [ ] Go extraction worker
- [ ] Disease NER
- [ ] Aggregation
- [ ] Tablassert KGX handoff
- [ ] Airflow or `dakp up` orchestration
- [ ] Other (describe below):

## Data Source
<!-- Include the source, release date, FAERS quarter, or other input identifier if known. -->

## Expected Behavior
<!-- What should happen? -->

## Actual Behavior
<!-- What actually happens? Include the complete error message if available. -->

## Environment
- OS: <!-- e.g., Ubuntu 24.04, macOS 15.3 -->
- Python version: <!-- e.g., 3.12.x -->
- Go version: <!-- e.g., 1.24.x -->
- `uv` version: <!-- e.g., output of `uv --version` -->
- DAKP version or commit: <!-- e.g., `git rev-parse HEAD` or `uv pip show dakp-pipeline` -->
- Tablassert version: <!-- e.g., output of `tablassert --version` or the installed package version -->
- Airflow version: <!-- e.g., output of `airflow version` -->
- `DAKP_ARIA2`: <!-- unset, 0, or enabled -->
- Airflow run/task ID: <!-- if applicable -->

## Logs and Screenshots
<!-- Attach relevant logs, task output, manifests, or screenshots. Remove secrets and credentials. -->
