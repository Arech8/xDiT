"""Deterministic append-only artifact writers with atomic completion."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json(value))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


class JsonlWriter:
    """A bounded JSONL stream that is renamed only after successful validation."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.partial_path = path.with_name(path.name + ".partial")
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self._stream = self.partial_path.open("w", encoding="utf-8")
        self._closed = False
        self._lock = threading.RLock()

    def write(self, value: Any) -> None:
        encoded = (canonical_json(value) + "\n").encode("utf-8")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot write to a closed JSONL writer")
            if self.bytes_written + len(encoded) > self.max_bytes:
                raise OSError(
                    f"JSONL quota exhausted at {self.bytes_written} of "
                    f"{self.max_bytes} bytes"
                )
            self._stream.write(encoded.decode("utf-8"))
            self.bytes_written += len(encoded)

    def finalize(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True
            os.replace(self.partial_path, self.path)

    def abort(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.flush()
                self._stream.close()
                self._closed = True
            self.partial_path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()
