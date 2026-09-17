from __future__ import annotations

import sys
import threading

import pytest
import torch

from xfuser.model_executor.execution_capture import (
    CaptureFinalizationError,
    ExecutionCaptureSession,
)
from xfuser.model_executor.execution_capture.snapshots import SnapshotStore

from .conftest import load_capture_events, load_capture_manifest

_library = torch.library.Library("xdit_capture_test", "DEF")
_library.define("opaque(Tensor value) -> Tensor")
_library.impl("opaque", lambda value: value + 7, "CPU")


class TensorSubclass(torch.Tensor):
    pass


def test_tensor_storage_views_mutations_subclass_and_custom_op(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={
            "strict": False,
            "capture_root_boundary": False,
            "capture_first_invocation_per_signature": False,
            "capture_parameters": False,
        },
    ):
        base = torch.arange(4.0).as_subclass(TensorSubclass)
        view = base.view(2, 2)
        view.add_(1)
        view.resize_(4)
        result = torch.ops.xdit_capture_test.opaque(view)
    assert result.shape == (4,)

    operations = [
        event
        for event in load_capture_events(tmp_path)
        if event["kind"] == "operation"
    ]
    assert any(event["observed_aliases"] for event in operations)
    assert any(event["observed_mutation"] for event in operations)
    assert any(
        metadata["python_type"].endswith("TensorSubclass")
        for event in operations
        for metadata in event.get("post_tensor_metadata", [])
    )
    custom = next(
        event for event in operations if event["op"].startswith("xdit_capture_test.")
    )
    assert custom["opaque_custom_op"] is True
    unsupported = [
        event
        for event in load_capture_events(tmp_path)
        if event["kind"] == "unsupported"
    ]
    assert any(
        event["unsupported_kind"] == "opaque_custom_operator"
        for event in unsupported
    )


def test_snapshot_deduplication_and_quota_loss_are_explicit(tmp_path):
    store = SnapshotStore(
        tmp_path / "direct",
        max_bytes=1024,
        max_tensors=10,
        max_objects=10,
    )
    tensor = torch.tensor([1.0])
    first = store.save(tensor, "first")
    second = store.save(tensor, "second")
    assert first == second
    assert len(store.entries) == 1
    assert next(iter(store.entries.values()))["purposes"] == ["first", "second"]

    capture_root = tmp_path / "capture"
    with ExecutionCaptureSession(
        capture_root,
        policy={
            "strict": False,
            "max_snapshot_bytes": 1,
            "capture_first_invocation_per_signature": False,
        },
    ) as session:
        session.capture_call(torch.clone, torch.ones(2))
    manifest = load_capture_manifest(capture_root)
    assert manifest["status"] == "incomplete"
    assert any(
        loss["kind"] == "snapshot_quota_exhausted"
        for loss in manifest["losses"]
    )


class Unserializable:
    def __reduce__(self):
        raise TypeError("cannot serialize sentinel")


def test_snapshot_failure_preserves_model_execution_and_restores_callbacks(tmp_path):
    called = False
    old_profile = sys.getprofile()
    old_trace = sys.gettrace()

    def model(value):
        nonlocal called
        called = True
        return torch.ones(1)

    with pytest.raises(CaptureFinalizationError), ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        session.capture_call(model, Unserializable())
    assert called
    assert sys.getprofile() is old_profile
    assert sys.gettrace() is old_trace


def test_future_thread_gets_profile_and_dispatch_capture(tmp_path):
    outputs = []
    with ExecutionCaptureSession(
        tmp_path,
        policy={
            "capture_root_boundary": False,
            "capture_first_invocation_per_signature": False,
            "capture_parameters": False,
        },
    ):
        torch.zeros(1)
        thread = threading.Thread(target=lambda: outputs.append(torch.ones(1) + 2))
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert outputs[0].item() == 3
    manifest = load_capture_manifest(tmp_path)
    assert manifest["thread_count"] >= 2
    operations = [
        event
        for event in load_capture_events(tmp_path)
        if event["kind"] == "operation"
    ]
    assert len({event["thread_id"] for event in operations}) >= 2
