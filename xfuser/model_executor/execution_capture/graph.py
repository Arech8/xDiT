"""Offline graph materialization from dispatcher operation ranges."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import GRAPH_SCHEMA_NAME, GRAPH_SCHEMA_VERSION
from .writer import atomic_write_json, canonical_json


class GraphMaterializationError(RuntimeError):
    pass


def read_events(capture_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(capture_dir) / "events.jsonl"
    if not path.exists():
        raise GraphMaterializationError(f"event stream does not exist: {path}")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GraphMaterializationError(
                    f"invalid JSON event at {path}:{line_number}"
                ) from exc
    return events


def materialize_operation_range(
    capture_dir: str | Path,
    operation_ids: Iterable[str],
    *,
    strict: bool = True,
) -> Path:
    root = Path(capture_dir)
    requested = list(operation_ids)
    by_id = {
        item["operation_id"]: item
        for item in read_events(root)
        if item.get("kind") == "operation"
    }
    missing = [operation_id for operation_id in requested if operation_id not in by_id]
    if missing:
        raise GraphMaterializationError(
            "operation(s) missing from event stream: " + ", ".join(missing)
        )
    operations = [by_id[operation_id] for operation_id in requested]
    unsupported = []
    for item in operations:
        if item.get("status") != "ok":
            unsupported.append(
                {"operation_id": item["operation_id"], "reason": "operation failed"}
            )
        if item.get("argument_snapshot") is None:
            unsupported.append(
                {
                    "operation_id": item["operation_id"],
                    "reason": "operation input snapshot was not captured",
                }
            )
        if item.get("opaque_custom_op"):
            unsupported.append(
                {
                    "operation_id": item["operation_id"],
                    "reason": "custom operator is opaque",
                }
            )
        collective = item.get("collective")
        if collective is not None and collective.get("group") is None:
            unsupported.append(
                {
                    "operation_id": item["operation_id"],
                    "reason": "collective process group is unavailable",
                }
            )
    graph_key = {
        "operations": [
            {
                "op": item["op"],
                "schema": item.get("schema"),
                "arguments": item.get("arguments"),
                "declared_alias_mutation": item.get("declared_alias_mutation"),
            }
            for item in operations
        ]
    }
    digest = hashlib.sha256(canonical_json(graph_key).encode()).hexdigest()
    graph = {
        "schema": GRAPH_SCHEMA_NAME,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "graph_id": f"sha256:{digest}",
        "supported": not unsupported,
        "unsupported": unsupported,
        "operation_ids": requested,
        "operations": operations,
    }
    destination = root / "graphs" / f"sha256-{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        atomic_write_json(destination, graph)
    if strict and unsupported:
        reasons = "; ".join(
            f"{item['operation_id']}: {item['reason']}" for item in unsupported
        )
        raise GraphMaterializationError(f"operation range is not replayable: {reasons}")
    return destination


def materialize_invocation(
    capture_dir: str | Path,
    invocation_id: str,
    *,
    strict: bool = True,
) -> Path:
    for item in read_events(capture_dir):
        if (
            item.get("kind") == "invocation_binding"
            and item.get("invocation_id") == invocation_id
        ):
            return materialize_operation_range(
                capture_dir, item["subtree_operation_ids"], strict=strict
            )
    raise GraphMaterializationError(
        f"invocation {invocation_id!r} has no materializable operation graph"
    )
