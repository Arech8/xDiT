"""Quota-bounded content-addressed snapshots."""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import torch


class SnapshotQuotaExceeded(RuntimeError):
    pass


def _estimate(value: Any, seen: set[int] | None = None) -> tuple[int, int, int]:
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0, 0, 0
    seen.add(object_id)
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size(), 1, 1
    if isinstance(value, dict):
        totals = [_estimate(item, seen) for pair in value.items() for item in pair]
    elif isinstance(value, (tuple, list, set, frozenset)):
        totals = [_estimate(item, seen) for item in value]
    else:
        return 0, 0, 1
    if not totals:
        return 0, 0, 1
    byte_count, tensor_count, object_count = (
        sum(parts) for parts in zip(*totals, strict=True)
    )
    return byte_count, tensor_count, object_count + 1


class SnapshotStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        max_tensors: int,
        max_objects: int,
        suppress: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_tensors = max_tensors
        self.max_objects = max_objects
        self.bytes_reserved = 0
        self.tensor_count = 0
        self.object_count = 0
        self._ordinal = 0
        self._lock = threading.RLock()
        self._suppress = suppress or nullcontext
        self.entries: dict[str, dict[str, Any]] = {}

    def save(self, value: Any, purpose: str) -> dict[str, Any]:
        _, tensors, objects = _estimate(value)
        with self._lock:
            temporary = self.root / f"snapshot-{self._ordinal:08d}.pt.partial"
            self._ordinal += 1
            try:
                with self._suppress():
                    torch.save(
                        value,
                        temporary,
                        _use_new_zipfile_serialization=False,
                    )
                digest = hashlib.sha256()
                with temporary.open("rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        digest.update(block)
                content_hash = digest.hexdigest()
                destination = self.root / f"sha256-{content_hash}.pt"
                actual_bytes = temporary.stat().st_size
                existing = self.entries.get(content_hash)
                if existing is not None:
                    temporary.unlink()
                    if purpose not in existing["purposes"]:
                        existing["purposes"].append(purpose)
                        existing["purposes"].sort()
                    return {
                        "snapshot_id": existing["snapshot_id"],
                        "file": f"snapshots/{existing['file']}",
                        "sha256": content_hash,
                    }
                if self.bytes_reserved + actual_bytes > self.max_bytes:
                    raise SnapshotQuotaExceeded(
                        f"snapshot byte quota exceeded: requested {actual_bytes}, "
                        f"remaining {self.max_bytes - self.bytes_reserved}"
                    )
                if self.tensor_count + tensors > self.max_tensors:
                    raise SnapshotQuotaExceeded(
                        f"snapshot tensor quota exceeded: requested {tensors}, "
                        f"remaining {self.max_tensors - self.tensor_count}"
                    )
                if self.object_count + objects > self.max_objects:
                    raise SnapshotQuotaExceeded(
                        f"snapshot object quota exceeded: requested {objects}, "
                        f"remaining {self.max_objects - self.object_count}"
                    )
                os.replace(temporary, destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

            self.bytes_reserved += actual_bytes
            self.tensor_count += tensors
            self.object_count += objects
            entry = {
                "snapshot_id": f"sha256:{content_hash}",
                "file": destination.name,
                "sha256": content_hash,
                "serialized_bytes": actual_bytes,
                "tensor_count": tensors,
                "object_count": objects,
                "purposes": [purpose],
            }
            self.entries[content_hash] = entry
        return {
            "snapshot_id": entry["snapshot_id"],
            "file": f"snapshots/{entry['file']}",
            "sha256": content_hash,
        }
