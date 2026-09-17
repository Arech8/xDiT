from __future__ import annotations

import json

import pytest
import torch

from xfuser.model_executor.execution_capture import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    CaptureConflictError,
    CapturePolicy,
    CapturePolicyError,
    ExecutionCaptureSession,
)
from xfuser.model_executor.execution_capture.writer import canonical_json

from .conftest import load_capture_manifest


def test_policy_rejects_unknown_and_mistyped_fields():
    with pytest.raises(CapturePolicyError, match="unknown"):
        CapturePolicy.from_json('{"invented": true}')
    with pytest.raises(CapturePolicyError, match="JSON boolean"):
        CapturePolicy.from_json('{"strict": 1}')
    with pytest.raises(CapturePolicyError, match="positive integer"):
        CapturePolicy.from_json('{"max_snapshot_bytes": 0}')


def test_schema_atomic_completion_and_deterministic_policy(tmp_path):
    policy = {
        "strict": True,
        "capture_first_invocation_per_signature": False,
    }
    with ExecutionCaptureSession(tmp_path, policy=policy) as session:
        result = session.capture_call(torch.add, torch.tensor([1]), 2)
    assert result.item() == 3

    capture = tmp_path / "execution_capture_rank_0"
    manifest = load_capture_manifest(tmp_path)
    assert manifest["schema"] == SCHEMA_NAME
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["policy"] == CapturePolicy.from_json(policy).as_dict()
    assert (capture / "COMPLETE").is_file()
    assert not list(capture.rglob("*.partial"))
    lines = (capture / "events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert lines == [canonical_json(event) for event in parsed]
    assert [event["sequence"] for event in parsed] == list(range(len(parsed)))
    assert parsed[-1]["kind"] == "session_end"


def test_model_exception_is_rethrown_unchanged_and_capture_is_incomplete(tmp_path):
    sentinel = RuntimeError("model sentinel")

    def fail():
        raise sentinel

    with (
        pytest.raises(RuntimeError) as caught,
        ExecutionCaptureSession(tmp_path) as session,
    ):
        session.capture_call(fail)
    assert caught.value is sentinel
    manifest = load_capture_manifest(tmp_path)
    assert manifest["status"] == "incomplete"
    assert not (tmp_path / "execution_capture_rank_0" / "COMPLETE").exists()


def test_profiler_conflict_fails_without_replacing_callback(tmp_path):
    import sys

    def callback(frame, event, arg):
        return None

    old = sys.getprofile()
    sys.setprofile(callback)
    try:
        with (
            pytest.raises(CaptureConflictError, match="existing callback"),
            ExecutionCaptureSession(tmp_path),
        ):
            pass
        assert sys.getprofile() is callback
    finally:
        sys.setprofile(old)
