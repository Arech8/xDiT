from __future__ import annotations

import threading

import torch
from PIL import Image

from xfuser.model_executor.execution_capture import (
    ExecutionCaptureSession,
)

from .conftest import load_capture_events, load_capture_manifest


class TextEncoder(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class TransformerBlock(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2


class StatefulScheduler:
    def __init__(self) -> None:
        self._step_index = 0
        self.timesteps = torch.tensor([2, 1])
        self.config = {"algorithm": "toy"}

    def step(self, noise: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        self._step_index += 1
        return sample - noise


class ToyVAE:
    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value.sigmoid().mul(255).to(torch.uint8)


class ImageProcessor:
    def postprocess(self, value: torch.Tensor) -> Image.Image:
        array = value.detach().cpu().numpy().reshape(1, 1)
        return Image.fromarray(array, mode="L")


class FluxLikePipeline:
    def __init__(self) -> None:
        self.encoder = TextEncoder()
        self.block = TransformerBlock()
        self.scheduler = StatefulScheduler()
        self.vae = ToyVAE()
        self.processor = ImageProcessor()

    def __call__(self, value: torch.Tensor) -> Image.Image:
        value = self.encoder(value)
        for _ in range(2):
            conditional = self.block(value)
            unconditional = self.block(value)
            value = self.scheduler.step(
                (conditional + unconditional) / 8,
                value,
            )
        return self.processor.postprocess(self.vae.decode(value))


def _tensor_ids(value):
    if isinstance(value, dict):
        result = (
            {value["tensor_id"]}
            if value.get("type") == "tensor_ref"
            else set()
        )
        for child in value.values():
            result.update(_tensor_ids(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_tensor_ids(child))
        return result
    return set()


def test_flux_like_scopes_flow_graph_dedup_and_orchestration_replay(tmp_path):
    pipeline = FluxLikePipeline()
    policy = {
        "strict": False,
        "snapshot_operations": True,
        "capture_first_invocation_per_signature": False,
        "capture_parameters": False,
        "max_snapshot_tensors": 512,
        "max_snapshot_objects": 4096,
    }
    def run_pipeline():
        outputs = []
        thread = threading.Thread(
            target=lambda: outputs.append(pipeline(torch.tensor([0.25])))
        )
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        return outputs[0]

    with ExecutionCaptureSession(tmp_path, policy=policy) as session:
        image = session.capture_call(run_pipeline)
    assert isinstance(image, Image.Image)

    events = load_capture_events(tmp_path)
    invocations = [
        event for event in events if event["kind"] == "module_invocation_start"
    ]
    block_module = next(
        module_id
        for module_id in {event["module_id"] for event in invocations}
        if sum(event["module_id"] == module_id for event in invocations) == 4
    )
    repeated = [event for event in invocations if event["module_id"] == block_module]
    assert len(repeated) == 4

    operation_outputs = set()
    observed_reuse = False
    for event in events:
        if event["kind"] != "operation":
            continue
        inputs = _tensor_ids(event["arguments"])
        if operation_outputs & inputs:
            observed_reuse = True
        operation_outputs.update(_tensor_ids(event["result"]))
    assert observed_reuse

    graph_definitions = [
        event for event in events if event["kind"] == "graph_definition"
    ]
    graph_references = [
        event for event in events if event["kind"] == "graph_reference"
    ]
    assert graph_definitions
    assert graph_references
    assert load_capture_manifest(tmp_path)["status"] == "incomplete"


_wan_library = torch.library.Library("xdit_capture_wan_test", "DEF")
_wan_library.define("packed(Tensor value) -> Tensor")
_wan_library.impl("packed", lambda value: value.cos(), "CPU")


class Expert(torch.nn.Module):
    def forward(self, value: torch.Tensor, timestep: int) -> torch.Tensor:
        return value * (timestep + 1)


class PackedState(torch.Tensor):
    pass


class WanLikePipeline:
    def __init__(self) -> None:
        self.experts = (Expert(), Expert())
        self.scheduler = StatefulScheduler()
        self.vae = ToyVAE()
        self.processor = ImageProcessor()

    def __call__(self, value: torch.Tensor) -> Image.Image:
        value = value.as_subclass(PackedState)
        for timestep in (1, 0):
            expert = self.experts[timestep]
            conditional = expert(value, timestep)
            unconditional = expert(value, timestep)
            packed = torch.ops.xdit_capture_wan_test.packed(
                conditional + unconditional
            )
            value = self.scheduler.step(packed / 4, value)
        return self.processor.postprocess(self.vae.decode(value))


def test_wan_like_expert_context_and_opaque_custom_op_are_explicit(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={
            "strict": False,
            "capture_first_invocation_per_signature": False,
            "capture_parameters": False,
        },
    ) as session:
        output = session.capture_call(WanLikePipeline(), torch.tensor([0.5]))
    assert isinstance(output, Image.Image)

    events = load_capture_events(tmp_path)
    module_starts = [
        event for event in events if event["kind"] == "module_invocation_start"
    ]
    expert_ids = {
        event["module_id"]
        for event in module_starts
        if sum(
            other["module_id"] == event["module_id"] for other in module_starts
        )
        == 2
    }
    assert len(expert_ids) == 2
    custom = [
        event
        for event in events
        if event["kind"] == "operation"
        and event["op"].startswith("xdit_capture_wan_test.")
    ]
    assert len(custom) == 2
    assert all(event["opaque_custom_op"] for event in custom)
    assert any(
        event["kind"] == "unsupported"
        and event["unsupported_kind"] == "opaque_custom_operator"
        for event in events
    )
