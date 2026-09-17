from __future__ import annotations

import subprocess
import sys

from .conftest import load_capture_events


def test_plain_python_scheduler_vae_processor_and_pipeline_scopes(tmp_path):
    program = r"""
import sys
import numpy as np
import torch
from PIL import Image
from xfuser.model_executor.execution_capture import ExecutionCaptureSession

class Scheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1])
        self._step_index = 0
        self.config = {"kind": "toy"}
    def step(self, value):
        self._step_index += 1
        return value - 1

class ToyVAE:
    def decode(self, value):
        return value.sigmoid().mul(255).to(torch.uint8)

class Processor:
    def postprocess(self, value):
        return Image.fromarray(value.detach().cpu().numpy().reshape(1, 1), mode="L")

class Pipeline:
    def __init__(self):
        self.scheduler = Scheduler()
        self.vae = ToyVAE()
        self.processor = Processor()
    def __call__(self, value):
        return self.processor.postprocess(
            self.vae.decode(self.scheduler.step(value))
        )

with ExecutionCaptureSession(
    sys.argv[1],
    policy={
        "strict": False,
        "capture_first_invocation_per_signature": False,
        "capture_parameters": False,
    },
) as session:
    session.capture_call(Pipeline(), torch.tensor([0.5]))
"""
    subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        check=True,
        timeout=30,
        text=True,
        capture_output=True,
    )
    adapters = {
        event["adapter"]["category"]
        for event in load_capture_events(tmp_path)
        if event["kind"] == "scope_start" and event.get("adapter")
    }
    assert {"scheduler", "vae", "processor", "pipeline_control"} <= adapters
