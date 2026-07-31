"""Source fetchers for DailyMed, FAERS, and Drugs@FDA.

Each source module exposes a module-level :func:`fetch` (the default instance method) so tests
can ``monkeypatch.setattr(dailymed, "fetch", ...)`` and the pipeline can call
``dailymed.fetch(ctx)``. Fetchers always run their real (network) acquisition; offline test
runs monkeypatch the module-level ``fetch`` (or the stdlib download seam) to serve fixtures.
"""

from __future__ import annotations
