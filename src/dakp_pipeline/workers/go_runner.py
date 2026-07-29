"""Thin wrapper around compiled Go workers (stub).

In later milestones this shells out to ``dakp-worker`` (or ``go run ./go/cmd/dakp-worker``),
streaming its JSON log lines into the task logger. For Milestone 1 it raises a clear
``NotImplementedError`` so accidental invocation fails loudly.
"""

from __future__ import annotations

from collections.abc import Sequence

from dakp_pipeline.io.contracts import TaskContext


def run_worker(args: Sequence[str], ctx: TaskContext) -> None:
    """Invoke a Go worker. Stub: the Go worker tree lands in a later milestone."""
    msg = "Go workers are not implemented in Milestone 1; the go/ tree lands later."
    raise NotImplementedError(msg)


__all__ = ["run_worker"]
