"""Extraction stage: parse raw source artifacts into normalized interim tables (stubs).

Real streaming/partitioned extraction lands in **Milestone 3**. The mock parsers here
handle the tiny fixture shapes and write interim parquet tables under ``data/interim/``.
"""

from __future__ import annotations
