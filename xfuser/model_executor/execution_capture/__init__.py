"""xDiT semantic execution capture and replay.

Imports are lazy so the disabled feature adds no recorder hooks or heavyweight
capture imports to normal inference startup.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "CaptureConflictError",
    "CaptureFinalizationError",
    "CapturePolicy",
    "CapturePolicyError",
    "ExecutionCaptureSession",
    "GraphMaterializationError",
    "RankMergeError",
    "ReplayPreflightError",
    "materialize_invocation",
    "materialize_operation_range",
    "merge_rank_captures",
    "replay_graph",
    "replay_orchestration",
    "replay_preflight",
    "validate_policy_json",
]

_EXPORT_MODULES = {
    "CapturePolicy": "policy",
    "CapturePolicyError": "policy",
    "validate_policy_json": "policy",
    "ExecutionCaptureSession": "recorder",
    "GraphMaterializationError": "graph",
    "materialize_invocation": "graph",
    "materialize_operation_range": "graph",
    "RankMergeError": "merge",
    "merge_rank_captures": "merge",
    "replay_graph": "replay",
    "replay_orchestration": "replay",
    "replay_preflight": "replay",
    "SCHEMA_NAME": "schema",
    "SCHEMA_VERSION": "schema",
    "CaptureConflictError": "schema",
    "CaptureFinalizationError": "schema",
    "ReplayPreflightError": "schema",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
