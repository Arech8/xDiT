from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_capture_events(root: Path, rank: int = 0) -> list[dict[str, Any]]:
    path = root / f"execution_capture_rank_{rank}" / "events.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_capture_manifest(root: Path, rank: int = 0) -> dict[str, Any]:
    path = root / f"execution_capture_rank_{rank}" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))
