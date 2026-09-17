"""Strict eager replay for fully snapshotted dispatcher ranges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .graph import GraphMaterializationError, materialize_operation_range, read_events
from .schema import (
    GRAPH_SCHEMA_NAME,
    GRAPH_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ReplayPreflightError,
)
from .writer import canonical_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_preflight(
    capture_dir: str | Path,
    *,
    expected_rank: int | None = None,
    device_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(capture_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ReplayPreflightError(f"capture manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_NAME:
        raise ReplayPreflightError(
            f"unsupported capture schema {manifest.get('schema')!r}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReplayPreflightError(
            f"unsupported capture version {manifest.get('schema_version')!r}"
        )
    if (
        manifest.get("status") != "complete"
        or manifest.get("replayable") is not True
        or not (root / "COMPLETE").exists()
    ):
        raise ReplayPreflightError("capture is incomplete")
    fatal = [
        loss
        for loss in manifest.get("losses", [])
        if loss.get("severity") in {"error", "fatal"}
        or loss.get("replayable") is not True
    ]
    if fatal:
        raise ReplayPreflightError(
            "capture contains fatal loss events: "
            + "; ".join(item.get("kind", "unknown") for item in fatal)
        )
    events_meta = manifest.get("events")
    events_path = root / "events.jsonl"
    if not isinstance(events_meta, dict) or not events_path.is_file():
        raise ReplayPreflightError("event artifact metadata is missing")
    if _sha256(events_path) != events_meta.get("sha256"):
        raise ReplayPreflightError("event artifact checksum mismatch")
    snapshots_by_id = {
        snapshot["snapshot_id"]: snapshot
        for snapshot in manifest.get("snapshots", [])
    }
    for snapshot in snapshots_by_id.values():
        path = root / "snapshots" / snapshot["file"]
        if not path.exists():
            raise ReplayPreflightError(
                f"snapshot is missing: {snapshot['snapshot_id']}"
            )
        if _sha256(path) != snapshot.get("sha256"):
            raise ReplayPreflightError(
                f"snapshot checksum mismatch: {snapshot['snapshot_id']}"
            )
        if snapshot.get("snapshot_id") != f"sha256:{snapshot.get('sha256')}":
            raise ReplayPreflightError(
                f"snapshot content identity mismatch: {snapshot['snapshot_id']}"
            )
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ReplayPreflightError(
                f"snapshot is corrupt: {snapshot['snapshot_id']}"
            ) from exc
    try:
        events = read_events(root)
    except GraphMaterializationError as exc:
        raise ReplayPreflightError(str(exc)) from exc
    sequences = [item.get("sequence") for item in events]
    if sequences != list(range(len(events))):
        raise ReplayPreflightError("event sequence is not contiguous and monotonic")
    if not events or events[-1].get("kind") != "session_end":
        raise ReplayPreflightError("session_end is not the final event")
    for operation in (item for item in events if item.get("kind") == "operation"):
        if operation.get("status") != "ok":
            raise ReplayPreflightError(
                f"failed operation is not replayable: {operation.get('operation_id')}"
            )
        snapshot = operation.get("argument_snapshot")
        if snapshot is None:
            raise ReplayPreflightError(
                f"operation snapshot missing: {operation.get('operation_id')}"
            )
        if snapshot.get("snapshot_id") not in snapshots_by_id:
            raise ReplayPreflightError(
                f"operation snapshot is unreferenced: {operation.get('operation_id')}"
            )
        if operation.get("opaque_custom_op"):
            raise ReplayPreflightError(
                f"unresolved custom/HOP operator: {operation.get('op')}"
            )
        collective = operation.get("collective")
        if collective is not None:
            if not collective.get("resolved"):
                raise ReplayPreflightError("collective process group is unresolved")
            group_id = collective.get("group_id")
            groups = {
                group["group_id"]: group
                for group in manifest.get("process_groups", [])
            }
            if group_id not in groups:
                raise ReplayPreflightError(
                    f"collective group definition missing: {group_id}"
                )
            group = groups[group_id]
            if (
                not group.get("registered_names")
                or len(group.get("ranks") or []) != group.get("size")
            ):
                raise ReplayPreflightError(
                    f"collective group topology is incomplete: {group_id}"
                )
            if collective.get("async_requested"):
                raise ReplayPreflightError(
                    "escaping asynchronous collective Work is unsupported"
                )
        for device in operation.get("device", {}).get("devices", []):
            if device == "cpu":
                continue
            if device_mapping is None or device not in device_mapping:
                raise ReplayPreflightError(
                    f"device mapping missing for captured device {device}"
                )
    if expected_rank is not None and manifest.get("rank") != expected_rank:
        raise ReplayPreflightError("capture rank does not match requested rank")
    if int(manifest.get("world_size", 1)) > 1:
        expected_ranks = set(range(int(manifest["world_size"])))
        siblings: dict[int, tuple[Path, dict[str, Any]]] = {}
        for sibling in root.parent.glob("execution_capture_rank_*"):
            sibling_manifest = sibling / "manifest.json"
            if sibling_manifest.is_file():
                candidate = json.loads(sibling_manifest.read_text(encoding="utf-8"))
                siblings[int(candidate["rank"])] = (sibling, candidate)
        if set(siblings) != expected_ranks:
            raise ReplayPreflightError("distributed capture rank set is incomplete")

        def topology(candidate: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                {
                    key: value
                    for key, value in group.items()
                    if key not in {"source_object_id", "world_rank", "local_rank"}
                }
                for group in candidate.get("process_groups", [])
            ]

        reference_topology = topology(manifest)
        sequences = []
        for rank, (sibling, candidate) in siblings.items():
            if (
                candidate.get("status") != "complete"
                or candidate.get("replayable") is not True
                or not (sibling / "COMPLETE").is_file()
            ):
                raise ReplayPreflightError(
                    f"distributed rank {rank} capture is incomplete"
                )
            if (
                candidate.get("capture_uuid") != manifest.get("capture_uuid")
                or candidate.get("configuration_id")
                != manifest.get("configuration_id")
            ):
                raise ReplayPreflightError("distributed capture identity mismatch")
            if topology(candidate) != reference_topology:
                raise ReplayPreflightError(
                    "distributed process-group topology mismatch"
                )
            sequences.append(
                [
                    {
                        "op": item["op"],
                        "group_id": item["collective"].get("group_id"),
                        "ordinal": item["collective"].get("local_ordinal"),
                        "tensor_contract": item["collective"].get(
                            "tensor_contract"
                        ),
                    }
                    for item in read_events(sibling)
                    if item.get("kind") == "operation"
                    and item.get("collective") is not None
                ]
            )
        if any(sequence != sequences[0] for sequence in sequences[1:]):
            raise ReplayPreflightError("distributed collective sequence mismatch")
    captured_world_size = int(manifest.get("world_size", 1))
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() != captured_world_size
    ):
        raise ReplayPreflightError(
            "initialized distributed world size does not match capture"
        )
    return manifest


def _create_replay_groups(
    manifest: dict[str, Any],
) -> dict[str, torch.distributed.ProcessGroup]:
    records = sorted(
        manifest.get("process_groups", []),
        key=lambda item: item["creation_ordinal"],
    )
    if not records:
        return {}
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise ReplayPreflightError(
            "distributed replay requires an initialized process group"
        )
    if torch.distributed.get_world_size() != manifest["world_size"]:
        raise ReplayPreflightError("distributed replay world size mismatch")
    groups: dict[str, torch.distributed.ProcessGroup] = {}
    for record in records:
        backend = record["backend"]
        if record.get("is_world"):
            group = torch.distributed.group.WORLD
            if str(group._get_backend_name()) != backend:
                raise ReplayPreflightError("WORLD backend does not match capture")
        else:
            # All world ranks execute this loop in identical creation order.
            group = torch.distributed.new_group(
                ranks=record["ranks"],
                backend=backend,
            )
        groups[record["group_id"]] = group
        try:
            from torch.distributed.distributed_c10d import (
                _get_process_group_name,
                _register_process_group,
            )

            for alias in record["registered_names"]:
                if _get_process_group_name(group) != alias:
                    _register_process_group(alias, group)
        except Exception as exc:
            raise ReplayPreflightError(
                f"failed to register process-group aliases for {record['group_id']}"
            ) from exc
    return groups


def _execute_collective(
    operation: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    groups: dict[str, torch.distributed.ProcessGroup],
) -> Any:
    collective = operation["collective"]
    if collective.get("is_wait"):
        if not args or not isinstance(args[0], torch.Tensor):
            raise ReplayPreflightError("functional wait tensor input is missing")
        return args[0]
    group = groups.get(collective["group_id"])
    if group is None:
        raise ReplayPreflightError("collective replay group was not created")
    op = operation["op"]
    tensors = [item for item in torch.utils._pytree.tree_leaves(args) if isinstance(item, torch.Tensor)]
    if "allreduce" in op or "all_reduce" in op:
        if not tensors:
            raise ReplayPreflightError("all-reduce tensor is missing")
        tensor = (
            tensors[0].clone()
            if "_c10d_functional" in op
            else tensors[0]
        )
        torch.distributed.all_reduce(tensor, group=group)
        return tensor if "_c10d_functional" in op else ([tensor], None)
    if "all_gather_into_tensor" in op or "_allgather_base" in op:
        if "_c10d_functional" in op:
            if not tensors:
                raise ReplayPreflightError("functional all-gather input is missing")
            input_tensor = tensors[0]
            shape = list(input_tensor.shape)
            shape[0] *= torch.distributed.get_world_size(group)
            output = torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)
        else:
            if len(tensors) < 2:
                raise ReplayPreflightError(
                    "all-gather input/output tensors are missing"
                )
            output, input_tensor = tensors[0], tensors[1]
        torch.distributed.all_gather_into_tensor(output, input_tensor, group=group)
        return output if "_c10d_functional" in op else (output, None)
    if "all_to_all_single" in op or "alltoall_base" in op:
        if "_c10d_functional" in op:
            if not tensors:
                raise ReplayPreflightError("functional all-to-all input is missing")
            input_tensor = tensors[0]
            output = torch.empty_like(input_tensor)
        else:
            if len(tensors) < 2:
                raise ReplayPreflightError(
                    "all-to-all input/output tensors are missing"
                )
            output, input_tensor = tensors[0], tensors[1]
        split_sizes = collective.get("split_sizes") or []
        torch.distributed.all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=split_sizes[0] if split_sizes else None,
            input_split_sizes=split_sizes[1] if len(split_sizes) > 1 else None,
            group=group,
        )
        return output if "_c10d_functional" in op else (output, None)
    raise ReplayPreflightError(f"unsupported collective replay operation: {op}")


def _resolve_operator(name: str, expected_schema: str | None) -> Any:
    parts = name.split(".")
    if len(parts) < 2:
        raise ReplayPreflightError(f"operator name is not resolvable: {name!r}")
    namespace, packet_name, *overload_parts = parts
    try:
        packet = getattr(getattr(torch.ops, namespace), packet_name)
        operator = (
            getattr(packet, ".".join(overload_parts)) if overload_parts else packet.default
        )
    except AttributeError as exc:
        raise ReplayPreflightError(f"operator is not registered: {name}") from exc
    if expected_schema is not None and str(operator._schema) != expected_schema:
        raise ReplayPreflightError(
            f"operator schema mismatch for {name}: expected {expected_schema}, "
            f"found {operator._schema}"
        )
    return operator


class _TensorEnvironment:
    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.storages: dict[str, torch.Tensor] = {}

    def resolve(self, metadata: dict[str, Any], template: Any) -> torch.Tensor:
        tensor_id = metadata["tensor_id"]
        if tensor_id in self.tensors:
            return self.tensors[tensor_id]
        if not isinstance(template, torch.Tensor):
            raise ReplayPreflightError(
                f"tensor template missing for external input {tensor_id}"
            )
        storage_id = metadata.get("storage_id")
        if storage_id is not None and storage_id in self.storages:
            owner = self.storages[storage_id]
            try:
                tensor = torch.empty(
                    0, dtype=template.dtype, device=owner.device
                ).set_(
                    owner.untyped_storage(),
                    int(metadata.get("storage_offset") or 0),
                    tuple(metadata["shape"]),
                    tuple(metadata["stride"]),
                )
            except Exception as exc:
                raise ReplayPreflightError(
                    f"cannot reconstruct storage alias for {tensor_id}"
                ) from exc
        else:
            tensor = template
            if storage_id is not None:
                self.storages[storage_id] = tensor
        self.tensors[tensor_id] = tensor
        return tensor

    def bind(self, metadata: dict[str, Any], value: torch.Tensor) -> None:
        self.tensors[metadata["tensor_id"]] = value
        storage_id = metadata.get("storage_id")
        if storage_id is not None:
            prior = self.storages.get(storage_id)
            if prior is not None:
                try:
                    if prior.untyped_storage().data_ptr() != value.untyped_storage().data_ptr():
                        raise ReplayPreflightError(
                            f"operator violated captured storage alias {storage_id}"
                        )
                except RuntimeError as exc:
                    raise ReplayPreflightError(
                        f"opaque storage cannot be validated for {storage_id}"
                    ) from exc
            else:
                self.storages[storage_id] = value


def _decode_with_template(
    metadata: Any, template: Any, env: _TensorEnvironment
) -> Any:
    if not isinstance(metadata, dict):
        return template
    kind = metadata.get("type")
    if kind == "tensor_ref":
        return env.resolve(metadata, template)
    if kind == "sequence":
        items = metadata.get("items", [])
        if not isinstance(template, (tuple, list)) or len(template) != len(items):
            raise ReplayPreflightError("sequence snapshot does not match metadata")
        decoded = [
            _decode_with_template(item, source, env)
            for item, source in zip(items, template, strict=True)
        ]
        return tuple(decoded) if isinstance(template, tuple) else decoded
    if kind == "namedtuple":
        decoded = [
            _decode_with_template(item, source, env)
            for item, source in zip(metadata.get("items", []), template, strict=True)
        ]
        return type(template)(*decoded)
    if kind == "mapping":
        if not isinstance(template, dict):
            raise ReplayPreflightError("mapping snapshot does not match metadata")
        decoded: dict[Any, Any] = {}
        template_items = list(template.items())
        metadata_items = metadata.get("items", [])
        if len(template_items) != len(metadata_items):
            raise ReplayPreflightError("mapping snapshot length does not match metadata")
        for (key_meta, value_meta), (key, value) in zip(
            metadata_items, template_items, strict=True
        ):
            decoded[_decode_with_template(key_meta, key, env)] = _decode_with_template(
                value_meta, value, env
            )
        return decoded
    return template


def _bind_results(metadata: Any, value: Any, env: _TensorEnvironment) -> None:
    if not isinstance(metadata, dict):
        return
    kind = metadata.get("type")
    if kind == "tensor_ref":
        if not isinstance(value, torch.Tensor):
            raise ReplayPreflightError("operator result does not match tensor metadata")
        env.bind(metadata, value)
    elif kind in {"sequence", "namedtuple"}:
        for child, item in zip(metadata.get("items", []), value, strict=True):
            _bind_results(child, item, env)
    elif kind == "mapping":
        for (_, child), (_, item) in zip(
            metadata.get("items", []), value.items(), strict=True
        ):
            _bind_results(child, item, env)


def replay_graph(
    capture_dir: str | Path,
    graph: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> Any:
    root = Path(capture_dir)
    manifest = replay_preflight(root)
    graph_path = Path(graph)
    try:
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayPreflightError("materialized graph is missing or corrupt") from exc
    if not graph_data.get("supported"):
        raise ReplayPreflightError("materialized graph is marked unsupported")
    graph_key = {
        "operations": [
            {
                "op": item["op"],
                "schema": item.get("schema"),
                "arguments": item.get("arguments"),
                "declared_alias_mutation": item.get("declared_alias_mutation"),
            }
            for item in graph_data.get("operations", [])
        ]
    }
    digest = hashlib.sha256(canonical_json(graph_key).encode()).hexdigest()
    if graph_data.get("graph_id") != f"sha256:{digest}":
        raise ReplayPreflightError("materialized graph checksum mismatch")
    if (
        graph_data.get("schema") != GRAPH_SCHEMA_NAME
        or graph_data.get("schema_version") != GRAPH_SCHEMA_VERSION
    ):
        raise ReplayPreflightError("materialized graph schema is incompatible")
    env = _TensorEnvironment()
    groups = _create_replay_groups(manifest)
    result: Any = None
    for operation in graph_data["operations"]:
        snapshot = operation.get("argument_snapshot")
        if snapshot is None:
            raise ReplayPreflightError(
                f"operation {operation['operation_id']} has no argument snapshot"
            )
        template = torch.load(
            root / snapshot["file"],
            map_location=map_location,
            weights_only=False,
        )
        decoded = _decode_with_template(operation["arguments"], template, env)
        if (
            not isinstance(decoded, tuple)
            or len(decoded) != 2
            or not isinstance(decoded[0], tuple)
            or not isinstance(decoded[1], dict)
        ):
            raise ReplayPreflightError("operation argument boundary is malformed")
        args, kwargs = decoded
        if operation.get("collective") is not None:
            result = _execute_collective(operation, args, kwargs, groups)
        else:
            operator = _resolve_operator(operation["op"], operation.get("schema"))
            result = operator(*args, **kwargs)
        _bind_results(operation["result"], result, env)
    return result


def replay_orchestration(
    capture_dir: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> Any:
    """Replay all supported dispatcher operations and return the executed result.

    Plain Python control is not re-executed. Captures containing required native
    transitions are therefore marked nonreplayable before this function runs.
    """

    root = Path(capture_dir)
    replay_preflight(root)
    operations = [
        item["operation_id"]
        for item in read_events(root)
        if item.get("kind") == "operation"
    ]
    try:
        graph = materialize_operation_range(root, operations, strict=True)
    except GraphMaterializationError as exc:
        raise ReplayPreflightError(str(exc)) from exc
    return replay_graph(root, graph, map_location=map_location)
