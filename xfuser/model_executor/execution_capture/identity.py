"""Rank-qualified identity tracking with Python-id generation safety."""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _Entry:
    reference: Any
    generation: int
    stable_id: str
    weak: bool

    def value(self) -> Any:
        return self.reference() if self.weak else self.reference


class IdentityRegistry:
    """Assign stable IDs without conflating views, storages, or reused Python IDs."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._entries: dict[str, dict[int, _Entry]] = {}
        self._counters: dict[str, int] = {}
        self._generations: dict[tuple[str, int], int] = {}
        self._lock = threading.RLock()

    def identify(self, value: Any, kind: str) -> str:
        python_id = id(value)
        with self._lock:
            entries = self._entries.setdefault(kind, {})
            existing = entries.get(python_id)
            if existing is not None and existing.value() is value:
                return existing.stable_id

            generation_key = (kind, python_id)
            generation = self._generations.get(generation_key, -1) + 1
            self._generations[generation_key] = generation
            ordinal = self._counters.get(kind, 0)
            self._counters[kind] = ordinal + 1
            stable_id = f"r{self.rank}:{kind}:{ordinal}:g{generation}"
            try:
                reference = weakref.ref(value)
                weak = True
            except TypeError:
                # Keeping a strong reference is the only safe way to prevent id reuse
                # for objects that do not support weak references.
                reference = value
                weak = False
            entries[python_id] = _Entry(reference, generation, stable_id, weak)
            return stable_id

    def tensor(self, value: torch.Tensor) -> str:
        return self.identify(value, "tensor")

    def storage(self, value: torch.Tensor) -> str | None:
        try:
            storage = value.untyped_storage()
        except (RuntimeError, NotImplementedError):
            return None
        return self.identify(storage, "storage")

    def module(self, value: torch.nn.Module) -> str:
        return self.identify(value, "module")

    def callable(self, value: Any) -> str:
        return self.identify(value, "callable")

    def object(self, value: Any) -> str:
        return self.identify(value, "object")
