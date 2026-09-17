"""Validated configuration for execution capture.

The command-line representation is deliberately JSON so that additions remain
backward compatible and capture behavior can be reproduced from the manifest.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class CapturePolicyError(ValueError):
    """Raised when an execution-capture policy is invalid."""


@dataclass(frozen=True)
class CapturePolicy:
    strict: bool = True
    fail_on_profiler_conflict: bool = True
    capture_root_boundary: bool = True
    capture_first_invocation_per_signature: bool = True
    capture_parameters: bool = True
    snapshot_operations: bool = True
    mutation_fingerprints: bool = False
    max_snapshot_bytes: int = 1 << 30
    max_snapshot_tensors: int = 256
    max_snapshot_objects: int = 1024
    max_jsonl_bytes: int = 4 << 30
    max_index_entries: int = 1_000_000
    graph_candidates: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_snapshot_bytes",
            "max_snapshot_tensors",
            "max_snapshot_objects",
            "max_jsonl_bytes",
            "max_index_entries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CapturePolicyError(f"{name} must be a positive integer")

    @classmethod
    def from_json(cls, value: str | Mapping[str, Any] | None) -> CapturePolicy:
        if value is None or value == "":
            return cls()
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise CapturePolicyError(
                    f"capture execution policy is not valid JSON: {exc.msg}"
                ) from exc
        else:
            decoded = dict(value)
        if not isinstance(decoded, dict):
            raise CapturePolicyError("capture execution policy must be a JSON object")

        fields = {field.name: field.type for field in dataclasses.fields(cls)}
        unknown = sorted(set(decoded) - set(fields))
        if unknown:
            raise CapturePolicyError(
                "unknown capture execution policy field(s): " + ", ".join(unknown)
            )

        bool_fields = {
            "strict",
            "fail_on_profiler_conflict",
            "capture_root_boundary",
            "capture_first_invocation_per_signature",
            "capture_parameters",
            "snapshot_operations",
            "mutation_fingerprints",
            "graph_candidates",
        }
        int_fields = set(fields) - bool_fields
        for name, item in decoded.items():
            if name in bool_fields and not isinstance(item, bool):
                raise CapturePolicyError(f"{name} must be a JSON boolean")
            if name in int_fields and (
                isinstance(item, bool) or not isinstance(item, int)
            ):
                raise CapturePolicyError(f"{name} must be a JSON integer")
        return cls(**decoded)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate_policy_json(value: str | None) -> str | None:
    """Argparse-compatible validator that preserves the original JSON string."""

    CapturePolicy.from_json(value)
    return value
