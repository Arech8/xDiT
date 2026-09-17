from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed._functional_collectives as funcol

from xfuser.model_executor.execution_capture import (
    ExecutionCaptureSession,
    RankMergeError,
    merge_rank_captures,
    replay_orchestration,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gloo_worker(rank: int, world_size: int, port: int, root: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group_one = dist.new_group(ranks=[0, 1], backend="gloo")
        group_two = dist.new_group(ranks=[0, 1], backend="gloo")

        class CollectiveChain(torch.nn.Module):
            def forward(self, value):
                gathered = funcol.all_gather_tensor(value, 0, group_one)
                reduced = funcol.all_reduce(gathered, "sum", group_two)
                return funcol.wait_tensor(reduced).clone()

        with ExecutionCaptureSession(
            root,
            rank=rank,
            world_size=world_size,
            capture_uuid="two-rank-gloo-test",
            policy={
                "capture_root_boundary": False,
                "capture_first_invocation_per_signature": False,
                "capture_parameters": False,
            },
        ) as session:
            value = torch.tensor([float(rank + 1)])
            output = session.capture_call(CollectiveChain(), value)
        assert torch.equal(output, torch.tensor([2.0, 4.0]))
    finally:
        dist.destroy_process_group()


def _gloo_replay_worker(rank: int, world_size: int, port: int, root: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        output = replay_orchestration(
            Path(root) / f"execution_capture_rank_{rank}"
        )
        assert torch.equal(output, torch.tensor([2.0, 4.0]))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_rank_gloo_capture_merge_and_mismatch_detection(tmp_path):
    mp.spawn(
        _gloo_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    roots = [
        tmp_path / "execution_capture_rank_0",
        tmp_path / "execution_capture_rank_1",
    ]
    merged = merge_rank_captures(
        roots, tmp_path / "merged.json", allow_partial=False
    )
    assert merged["status"] == "complete"
    assert merged["dag"]["collective_nodes"]
    assert all(
        node["node_id"].startswith(f"r{node['rank']}:")
        for node in merged["dag"]["operation_nodes"]
    )
    manifest = json.loads((roots[0] / "manifest.json").read_text())
    assert len(manifest["process_groups"]) == 2
    assert len({group["group_id"] for group in manifest["process_groups"]}) == 2
    mp.spawn(
        _gloo_replay_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    rank_one_events = roots[1] / "events.jsonl"
    events = [
        json.loads(line)
        for line in rank_one_events.read_text(encoding="utf-8").splitlines()
    ]
    collective = next(
        event
        for event in events
        if event.get("kind") == "operation" and event.get("collective")
    )
    collective["op"] = "c10d.broadcast_.default"
    rank_one_events.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    mismatched = merge_rank_captures(
        roots, tmp_path / "mismatched.json", allow_partial=True
    )
    assert mismatched["status"] == "partial"
    assert any(
        item["kind"] == "collective_sequence_mismatch"
        for item in mismatched["mismatches"]
    )
    with pytest.raises(RankMergeError, match="incomplete|validation failed"):
        merge_rank_captures(
            roots, tmp_path / "strict-mismatch.json", allow_partial=False
        )


def test_escaping_async_work_is_explicitly_nonreplayable(tmp_path):
    init_file = tmp_path / "single-rank-init"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        with ExecutionCaptureSession(
            tmp_path / "async",
            capture_uuid="async-work-test",
            policy={
                "strict": False,
                "capture_first_invocation_per_signature": False,
            },
        ) as session:
            work = session.capture_call(
                dist.all_reduce,
                torch.ones(1),
                async_op=True,
            )
        work.wait()
    finally:
        dist.destroy_process_group()
    manifest = json.loads(
        (
            tmp_path
            / "async"
            / "execution_capture_rank_0"
            / "manifest.json"
        ).read_text()
    )
    assert manifest["status"] == "incomplete"
    assert any(
        loss["kind"] == "escaping_collective_work"
        for loss in manifest["losses"]
    )
