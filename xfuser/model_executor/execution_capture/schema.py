"""Versioned event schema primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

SCHEMA_NAME: Final = "xdit.execution-capture"
SCHEMA_VERSION: Final = "1.1.0"
GRAPH_SCHEMA_NAME: Final = "xdit.execution-capture.graph"
GRAPH_SCHEMA_VERSION: Final = "1.1.0"


@dataclass
class Completeness:
    complete: bool = True
    replayable: bool = True
    losses: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        kind: str,
        message: str,
        severity: str = "error",
        replayable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        loss = {
            "kind": kind,
            "message": message,
            "severity": severity,
            "replayable": replayable,
            "context": context or {},
        }
        self.losses.append(loss)
        if severity in {"error", "fatal"} or not replayable:
            self.complete = False
        if not replayable:
            self.replayable = False
        return loss


def manifest_status(
    *,
    execution_succeeded: bool,
    completeness: Completeness,
    writer_failed: bool,
    active_workers: int,
) -> tuple[str, bool]:
    """Central capture invariant: COMPLETE always implies strictly replayable."""

    replayable = (
        execution_succeeded
        and completeness.complete
        and completeness.replayable
        and not writer_failed
        and active_workers == 0
    )
    return ("complete" if replayable else "incomplete"), replayable


class CaptureError(RuntimeError):
    """Base class for recorder failures."""


class CaptureConflictError(CaptureError):
    """Raised when process-global instrumentation is already owned."""


class CaptureFinalizationError(CaptureError):
    """Raised by strict capture when artifacts are incomplete."""


class ReplayPreflightError(CaptureError):
    """Raised when a capture cannot be replayed safely."""


def event(kind: str, sequence: int, rank: int, **payload: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "sequence": sequence,
        "rank": rank,
        **payload,
    }
