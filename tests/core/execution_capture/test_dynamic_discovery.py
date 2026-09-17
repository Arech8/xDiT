from __future__ import annotations

import pytest
import torch

from xfuser.model_executor.execution_capture import ExecutionCaptureSession

from .conftest import load_capture_events, load_capture_manifest


class Reentrant(torch.nn.Module):
    def forward(self, *, value: torch.Tensor, depth: int) -> torch.Tensor:
        value = value + 1
        return self(value=value, depth=depth - 1) if depth else value


class DynamicPipeline:
    def __init__(self) -> None:
        self.shared = Reentrant()

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        dynamic = torch.nn.Identity()
        value = dynamic(value)
        value = self.shared(value=value, depth=1)
        return self.shared(value=value, depth=0)


def test_dynamic_reused_reentrant_kwargs_invocations_are_all_retained(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={
            "capture_first_invocation_per_signature": False,
            "capture_parameters": False,
        },
    ) as session:
        result = session.capture_call(DynamicPipeline(), torch.tensor([0.0]))
    assert result.item() == 3

    events = load_capture_events(tmp_path)
    starts = [
        event for event in events if event["kind"] == "module_invocation_start"
    ]
    ends = [event for event in events if event["kind"] == "module_invocation_end"]
    assert len(starts) == 4
    assert len(ends) == 4
    assert len({event["invocation_id"] for event in starts}) == 4
    assert {event["invocation_id"] for event in starts} == {
        event["invocation_id"] for event in ends
    }
    starts_by_id = {event["invocation_id"]: event for event in starts}
    ends_by_id = {event["invocation_id"]: event for event in ends}
    assert all(
        starts_by_id[invocation_id]["sequence"]
        < ends_by_id[invocation_id]["sequence"]
        for invocation_id in starts_by_id
    )
    for start in starts:
        parent_id = start["parent_invocation_id"]
        if parent_id is not None:
            assert starts_by_id[parent_id]["sequence"] < start["sequence"]
            assert (
                ends_by_id[start["invocation_id"]]["sequence"]
                < ends_by_id[parent_id]["sequence"]
            )
    module_ids = [event["module_id"] for event in starts]
    assert max(module_ids.count(module_id) for module_id in set(module_ids)) == 3
    assert any(event["parent_invocation_id"] is not None for event in starts)
    assert all(event["arguments"]["type"] == "sequence" for event in starts)
    assert load_capture_manifest(tmp_path)["invocation_count"] == 4


class RaisedForward(torch.nn.Module):
    def __init__(self, exception: BaseException) -> None:
        super().__init__()
        self.exception = exception

    def forward(self, *, value: torch.Tensor) -> torch.Tensor:
        raise self.exception


def test_raised_forward_is_recorded_and_same_exception_is_rethrown(tmp_path):
    sentinel = LookupError("forward sentinel")
    with (
        pytest.raises(LookupError) as caught,
        ExecutionCaptureSession(tmp_path) as session,
    ):
        session.capture_call(RaisedForward(sentinel), value=torch.ones(1))
    assert caught.value is sentinel

    events = load_capture_events(tmp_path)
    end = next(
        event for event in events if event["kind"] == "module_invocation_end"
    )
    assert end["status"] == "exception"
    assert end["exception"]["message"] == "forward sentinel"
