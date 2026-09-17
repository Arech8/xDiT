"""Structured, identity-preserving value descriptions."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from .identity import IdentityRegistry


def tensor_metadata(value: torch.Tensor, identities: IdentityRegistry) -> dict[str, Any]:
    try:
        version = value._version
    except RuntimeError:
        version = None
    try:
        storage_offset = value.storage_offset()
        stride = list(value.stride())
    except RuntimeError:
        storage_offset = None
        stride = None
    return {
        "tensor_id": identities.tensor(value),
        "storage_id": identities.storage(value),
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "shape": list(value.shape),
        "stride": stride,
        "storage_offset": storage_offset,
        "requires_grad": bool(value.requires_grad),
        "is_leaf": bool(value.is_leaf),
        "version": version,
    }


class ValueEncoder:
    def __init__(
        self,
        identities: IdentityRegistry,
        on_unsupported: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.identities = identities
        self.on_unsupported = on_unsupported

    def encode(self, value: Any, *, _seen: set[int] | None = None) -> Any:
        if _seen is None:
            _seen = set()
        if isinstance(value, torch.Tensor):
            return {"type": "tensor_ref", **tensor_metadata(value, self.identities)}
        if value is None or isinstance(value, (bool, int, str)):
            return {"type": "literal", "value": value}
        if isinstance(value, float):
            if math.isnan(value):
                return {"type": "float", "value": "nan"}
            if value == float("inf"):
                return {"type": "float", "value": "inf"}
            if value == float("-inf"):
                return {"type": "float", "value": "-inf"}
            return {"type": "literal", "value": value}
        if isinstance(value, (torch.dtype, torch.device, torch.layout, enum.Enum)):
            return {
                "type": "symbol",
                "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": str(value),
            }
        if isinstance(value, torch.Generator):
            return {
                "type": "generator",
                "object_id": self.identities.object(value),
                "device": str(value.device),
                "initial_seed": value.initial_seed(),
            }
        if isinstance(value, torch.ScriptObject):
            try:
                qualified_name = value._type().qualified_name()
            except Exception as exc:  # noqa: BLE001 - torchbind introspection
                self.on_unsupported(
                    "opaque_torchbind",
                    {"python_type": type(value).__qualname__, "error": repr(exc)},
                )
            else:
                if qualified_name.startswith("__torch__.torch.classes.c10d."):
                    return {
                        "type": "c10d_torchbind_ref",
                        "object_id": self.identities.object(value),
                        "qualified_name": qualified_name,
                    }
                self.on_unsupported(
                    "opaque_torchbind",
                    {"qualified_name": qualified_name},
                )
        object_id = id(value)
        if object_id in _seen:
            return {
                "type": "object_ref",
                "object_id": self.identities.object(value),
                "cycle": True,
            }
        _seen.add(object_id)
        try:
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return {
                    "type": "dataclass",
                    "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "fields": {
                        field.name: self.encode(getattr(value, field.name), _seen=_seen)
                        for field in dataclasses.fields(value)
                    },
                }
            if isinstance(value, Mapping):
                return {
                    "type": "mapping",
                    "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "items": [
                        [self.encode(key, _seen=_seen), self.encode(item, _seen=_seen)]
                        for key, item in value.items()
                    ],
                }
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                return {
                    "type": "namedtuple",
                    "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "items": [self.encode(item, _seen=_seen) for item in value],
                }
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                return {
                    "type": "sequence",
                    "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "items": [self.encode(item, _seen=_seen) for item in value],
                }
            if isinstance(value, torch.nn.Module):
                return {
                    "type": "module_ref",
                    "module_id": self.identities.module(value),
                    "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                }
            if callable(value):
                return {
                    "type": "callable_ref",
                    "callable_id": self.identities.callable(value),
                    "qualified_name": qualified_name(value),
                }
            process_group_type = getattr(torch.distributed, "ProcessGroup", ())
            if process_group_type and isinstance(value, process_group_type):
                return {
                    "type": "process_group_ref",
                    "object_id": self.identities.object(value),
                    "rank": value.rank(),
                    "size": value.size(),
                    "backend": str(value._get_backend_name()),
                }
        except Exception as exc:  # noqa: BLE001 - arbitrary values may raise
            self.on_unsupported(
                "value_encoding_error",
                {"python_type": type(value).__qualname__, "error": repr(exc)},
            )
        finally:
            _seen.discard(object_id)

        self.on_unsupported(
            "opaque_python_value",
            {"python_type": f"{type(value).__module__}.{type(value).__qualname__}"},
        )
        return {
            "type": "opaque_object",
            "object_id": self.identities.object(value),
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": safe_repr(value),
        }


def safe_repr(value: Any, limit: int = 512) -> str:
    try:
        result = repr(value)
    except Exception as exc:  # noqa: BLE001 - repr is user-extensible
        result = f"<repr failed: {type(exc).__name__}>"
    return result if len(result) <= limit else result[: limit - 3] + "..."


def qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
    return f"{module}.{name}"


def callable_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None
