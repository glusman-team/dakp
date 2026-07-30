"""Edge-case tests for ``dakp_pipeline.paths`` (the ``default_workdir`` fallback)."""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.paths import Workdir, default_workdir


def test_default_workdir_is_repo_local_data() -> None:
    wd = default_workdir()
    assert isinstance(wd, Workdir)
    assert wd.root == Path("data")
