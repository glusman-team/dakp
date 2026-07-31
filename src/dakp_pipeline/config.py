"""Pipeline profiles and configuration.

Profiles are defined in Python (not parsed from YAML) so the base install needs no
``pyyaml`` dependency. ``configs/pipeline.yaml`` and ``configs/local.example.yaml``
ship as human-readable declarations of the same knobs.

Three profiles (per ``PLAN.md`` "Resolved planning decisions" + performance tiers):

* ``mock``   — tiny fixtures, all external calls monkeypatchable, CI-friendly.
* ``sample`` — laptop-safe bounded sample, real acquisition but limited scope.
* ``prod``   — real full build on the 80-thread / 187 GiB workstation (the ``wenceslaus`` host).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PROFILES: frozenset[str] = frozenset({"mock", "sample", "prod"})


class DownloadConfig(BaseModel):
    """Acquisition/download tuning: concurrency bound + source overrides.

    Download-only knobs owned by the acquisition layer (:mod:`dakp_pipeline.acquire` and the
    Airflow acquisition tasks). Kept as one nested field on :class:`Profile` so it is fully
    isolated from unrelated profile concerns (e.g. the Go-worker integration field). Empty /
    ``None`` values mean "use the acquisition layer's documented defaults" — the mock profile
    never touches the network regardless of these values.
    """

    concurrency: int = Field(default=4, ge=1, description="Max concurrent source downloads (sizes the Airflow download pool / acquire_all workers).")
    ner_model_ids: tuple[str, ...] = Field(default_factory=tuple, description="NER model ids to cache; empty = backend default (mock acquires none).")
    drugsfda_url: str | None = Field(default=None, description="Override the Drugs@FDA data-files ZIP URL (forwarded to the fetcher).")


class Profile(BaseModel):
    """Execution profile: concurrency budgets and source-acquisition behavior."""

    name: str = Field(description="Profile name (mock | sample | prod).")
    threads: int = Field(ge=1, description="Worker threads/processes per task.")
    memory_budget_gb: int = Field(ge=1, description="Soft per-task memory budget in GiB.")
    quarter_limit: int | None = Field(default=None, description="Cap FAERS quarters processed (dev/sample); None = all available.")
    release_limit: int | None = Field(default=None, description="Cap DailyMed full releases processed (dev/sample/smoke); None = all available.")
    mock_sources: bool = Field(description="If True, fetchers load fixtures instead of the network (mock profile).")
    force: bool = Field(default=False, description="Ignore cached artifacts and rerun every stage.")
    run_tablassert: bool = Field(default=False, description="Invoke real ../Tablassert at the handoff stage (deferred in mock profile).")
    download: DownloadConfig = Field(default_factory=DownloadConfig, description="Acquisition/download tuning (concurrency + source overrides).")


_MOCK = Profile(name="mock", threads=1, memory_budget_gb=1, quarter_limit=1, mock_sources=True, force=False, run_tablassert=False)

_SAMPLE = Profile(name="sample", threads=4, memory_budget_gb=8, quarter_limit=1, mock_sources=False, force=False, run_tablassert=False)

_PROD = Profile(name="prod", threads=64, memory_budget_gb=128, quarter_limit=None, mock_sources=False, force=False, run_tablassert=True)

_PROFILE_TABLE: dict[str, Profile] = {"mock": _MOCK, "sample": _SAMPLE, "prod": _PROD}


def load_profile(name: str, **overrides: object) -> Profile:
    """Return the named profile with optional field overrides applied.

    Raises ``KeyError`` for an unknown profile name rather than silently defaulting,
    so typos fail loudly at startup.
    """
    if name not in _PROFILE_TABLE:
        known = ", ".join(sorted(PROFILES))
        msg = f"Unknown profile {name!r}; expected one of: {known}"
        raise KeyError(msg)
    base = _PROFILE_TABLE[name]
    if not overrides:
        return base
    data = base.model_dump()
    data.update(overrides)
    return Profile(**data)


__all__ = ["PROFILES", "DownloadConfig", "Profile", "load_profile"]
