"""Offline multi-rank validation and partial-order construction."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .graph import read_events
from .schema import SCHEMA_NAME, SCHEMA_VERSION
from .writer import atomic_write_json, canonical_json


class RankMergeError(RuntimeError):
    pass


def _group_key(collective: dict[str, Any]) -> str:
    group = collective.get("group") or {}
    return canonical_json(
        {
            "group_id": collective.get("group_id"),
            "registered_name": group.get("registered_name"),
            "backend": group.get("backend"),
            "size": group.get("size"),
            "ranks": group.get("ranks"),
        }
    )


def merge_rank_captures(
    rank_directories: Iterable[str | Path],
    output: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    roots = [Path(path) for path in rank_directories]
    if not roots:
        raise RankMergeError("at least one rank capture is required")
    operations: list[dict[str, Any]] = []
    ranks: dict[int, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    for root in roots:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            if allow_partial:
                mismatches.append({"kind": "missing_manifest", "path": str(root)})
                continue
            raise RankMergeError(f"rank manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != SCHEMA_NAME
            or manifest.get("schema_version") != SCHEMA_VERSION
        ):
            raise RankMergeError(f"rank schema mismatch: {root}")
        rank = int(manifest["rank"])
        if rank in ranks:
            raise RankMergeError(f"duplicate rank capture: {rank}")
        ranks[rank] = manifest
        if (
            manifest.get("status") != "complete"
            or manifest.get("replayable") is not True
            or not (root / "COMPLETE").exists()
        ):
            mismatch = {"kind": "incomplete_rank", "rank": rank}
            mismatches.append(mismatch)
            if not allow_partial:
                raise RankMergeError(f"rank {rank} capture is incomplete")
        operations.extend(
            item for item in read_events(root) if item.get("kind") == "operation"
        )

    world_sizes = {manifest.get("world_size") for manifest in ranks.values()}
    if len(world_sizes) != 1:
        raise RankMergeError(f"rank captures disagree on world size: {world_sizes}")
    expected_world_size = int(next(iter(world_sizes))) if world_sizes else 0
    capture_uuids = {manifest.get("capture_uuid") for manifest in ranks.values()}
    configuration_ids = {
        manifest.get("configuration_id") for manifest in ranks.values()
    }
    if len(capture_uuids) != 1 or None in capture_uuids:
        raise RankMergeError("rank captures disagree on capture UUID")
    if len(configuration_ids) != 1 or None in configuration_ids:
        raise RankMergeError("rank captures disagree on configuration identity")
    missing_ranks = sorted(set(range(expected_world_size)) - set(ranks))
    if missing_ranks:
        mismatches.append({"kind": "missing_ranks", "ranks": missing_ranks})
        if not allow_partial:
            raise RankMergeError(f"missing rank captures: {missing_ranks}")

    by_collective: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for operation in operations:
        collective = operation.get("collective")
        if collective is not None:
            ordinal = collective.get("local_ordinal")
            if ordinal is None or not collective.get("resolved", False):
                mismatches.append(
                    {
                        "kind": "unresolved_collective_group",
                        "rank": operation.get("rank"),
                        "operation_id": operation.get("operation_id"),
                    }
                )
                ordinal = -1
            by_collective[
                (_group_key(collective), int(ordinal))
            ].append(operation)

    collective_nodes: list[dict[str, Any]] = []
    collective_node_for_op: dict[str, str] = {}
    for (group_key, ordinal), members in sorted(by_collective.items()):
        names = {member["op"] for member in members}
        declared_ranks = {
            tuple(member["collective"].get("ranks") or []) for member in members
        }
        observed_ranks = {int(member["rank"]) for member in members}
        participant_ranks = [int(member["rank"]) for member in members]
        expected_ranks = (
            set(next(iter(declared_ranks)))
            if len(declared_ranks) == 1 and next(iter(declared_ranks))
            else set(ranks)
        )
        duplicate_ranks = sorted(
            rank for rank in observed_ranks if participant_ranks.count(rank) > 1
        )
        signatures = {
            canonical_json(
                {
                    "op": member["op"],
                    "schema": member.get("schema"),
                    "tensor_types": [
                        (item.get("dtype"), item.get("shape"))
                        for item in member.get("pre_tensor_metadata", [])
                    ],
                }
            )
            for member in members
        }
        if (
            len(names) != 1
            or len(signatures) != 1
            or observed_ranks != expected_ranks
            or duplicate_ranks
        ):
            mismatches.append(
                {
                    "kind": "collective_sequence_mismatch",
                    "group": json.loads(group_key),
                    "ordinal": ordinal,
                    "operators": sorted(names),
                    "expected_ranks": sorted(expected_ranks),
                    "observed_ranks": sorted(observed_ranks),
                    "duplicate_ranks": duplicate_ranks,
                    "signatures": sorted(signatures),
                }
            )
        node_id = f"collective:{len(collective_nodes)}"
        node = {
            "node_id": node_id,
            "group": json.loads(group_key),
            "ordinal": ordinal,
            "operator": next(iter(names)) if len(names) == 1 else None,
            "members": sorted(member["operation_id"] for member in members),
        }
        collective_nodes.append(node)
        for member in members:
            collective_node_for_op[member["operation_id"]] = node_id

    operation_nodes = [
        {
            "node_id": item["operation_id"],
            "rank": item["rank"],
            "thread_id": item["thread_id"],
            "thread_sequence": item["thread_sequence"],
            "op": item["op"],
        }
        for item in operations
    ]
    edges: list[dict[str, str]] = []
    per_thread: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for operation in operations:
        per_thread[(int(operation["rank"]), operation["thread_id"])].append(operation)
    for thread_operations in per_thread.values():
        thread_operations.sort(key=lambda item: int(item["thread_sequence"]))
        for index, operation in enumerate(thread_operations):
            source = operation["operation_id"]
            collective_node = collective_node_for_op.get(source)
            if collective_node is not None:
                edges.append(
                    {"source": source, "target": collective_node, "kind": "collective"}
                )
                if index + 1 < len(thread_operations):
                    edges.append(
                        {
                            "source": collective_node,
                            "target": thread_operations[index + 1]["operation_id"],
                            "kind": "happens_before",
                        }
                    )
            elif index + 1 < len(thread_operations):
                edges.append(
                    {
                        "source": source,
                        "target": thread_operations[index + 1]["operation_id"],
                        "kind": "thread_order",
                    }
                )

    merged = {
        "schema": "xdit.execution-capture.rank-merge",
        "schema_version": "1.0.0",
        "world_size": expected_world_size,
        "ranks": sorted(ranks),
        "status": "complete" if not mismatches else "partial",
        "mismatches": mismatches,
        "dag": {
            "operation_nodes": operation_nodes,
            "collective_nodes": collective_nodes,
            "edges": edges,
        },
    }
    if mismatches and not allow_partial:
        raise RankMergeError(
            "rank merge validation failed: "
            + ", ".join(item["kind"] for item in mismatches)
        )
    atomic_write_json(Path(output), merged)
    return merged
