from __future__ import annotations

import ast
from pathlib import Path

import pytest

from xfuser.config import FlexibleArgumentParser, xFuserArgs


def test_capture_is_disabled_by_default():
    args = xFuserArgs(model="toy")
    assert args.capture_execution is False
    assert args.capture_execution_policy is None


def test_runner_cli_accepts_hyphenated_capture_policy():
    parser = FlexibleArgumentParser()
    namespace = xFuserArgs.add_runner_args(parser).parse_args(
        [
            "--model",
            "toy",
            "--capture-execution",
            "--capture-execution-policy",
            '{"max_snapshot_bytes": 4096}',
        ]
    )
    args = xFuserArgs.from_cli_args(namespace)
    assert args.capture_execution is True
    assert args.capture_execution_policy == '{"max_snapshot_bytes": 4096}'


def test_capture_policy_and_incompatible_modes_are_rejected():
    with pytest.raises(ValueError, match="unknown capture"):
        xFuserArgs(
            model="toy",
            capture_execution=True,
            capture_execution_policy='{"unknown_capture_field": true}',
        )
    with pytest.raises(ValueError, match="--profile"):
        xFuserArgs(model="toy", capture_execution=True, profile=True)
    with pytest.raises(ValueError, match="requires eager"):
        xFuserArgs(
            model="toy",
            capture_execution=True,
            use_torch_compile=True,
        )
    args = xFuserArgs(
        model="toy",
        capture_execution=True,
        use_torch_compile=True,
        capture_execution_policy='{"strict": false}',
    )
    assert args.capture_execution


def test_new_sources_parse_as_python_310_and_avoid_newer_typing_self():
    root = Path(__file__).parents[3] / "xfuser" / "model_executor" / "execution_capture"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path), feature_version=(3, 10))
        assert "typing import Self" not in source
        assert "typing import Any, Self" not in source
