"""Extraction stage: parse raw source artifacts into normalized interim tables.

Streaming and partitioned extractors handle both tiny fixtures and real source artifacts,
writing parquet tables under ``interim/``.
"""

from __future__ import annotations
