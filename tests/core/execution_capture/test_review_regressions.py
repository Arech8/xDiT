from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from xfuser.model_executor.execution_capture import (
    ExecutionCaptureSession,
    ReplayPreflightError,
    materialize_invocation,
    materialize_operation_range,
    replay_graph,
    replay_orchestration,
    replay_preflight,
)
from xfuser.model_executor.execution_capture.snapshots import (
    SnapshotQuotaExceeded,
    SnapshotStore,
)
from xfuser.model_executor.execution_capture.writer import canonical_json
from xfuser.model_executor.models.runner_models.base_model import xFuserModel

from .conftest import load_capture_events, load_capture_manifest


class CountingMode(TorchDispatchMode):
    def __init__(self) -> None:
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def test_nested_dispatch_mode_semantics_and_recorder_detach_suppression(tmp_path):
    class Parameterized(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * self.weight

    counting = CountingMode()
    with counting, ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        output = session.capture_call(Parameterized(), torch.tensor([3.0]))
    assert output.item() == 6
    assert "aten.mul.Tensor" in counting.operations
    captured = [
        event["op"]
        for event in load_capture_events(tmp_path)
        if event["kind"] == "operation"
    ]
    assert "aten.detach.default" not in captured


def test_higher_order_cond_executes_and_is_explicitly_incomplete(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={
            "strict": False,
            "capture_first_invocation_per_signature": False,
        },
    ) as session:
        result = session.capture_call(
            lambda value: torch.cond(
                value.sum() > 0,
                lambda item: item + 1,
                lambda item: item - 1,
                (value,),
            ),
            torch.tensor([2.0]),
        )
    assert result.item() == 3
    manifest = load_capture_manifest(tmp_path)
    assert manifest["status"] == "incomplete"
    assert not (tmp_path / "execution_capture_rank_0" / "COMPLETE").exists()
    assert any(
        loss["kind"] == "unsupported_higher_order_operator"
        for loss in manifest["losses"]
    )


def test_replay_returns_executed_result_not_root_snapshot(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        captured = session.capture_call(lambda value: value * 3, torch.tensor([2.0]))
    root = tmp_path / "execution_capture_rank_0"
    events = load_capture_events(tmp_path)
    root_snapshot = next(
        event
        for event in events
        if event["kind"] == "snapshot" and event["purpose"] == "root_outputs"
    )
    replacement = root / "snapshots" / "deliberately-wrong.pt"
    torch.save(torch.tensor([-999.0]), replacement)
    root_snapshot["file"] = "snapshots/deliberately-wrong.pt"
    events_path = root / "events.jsonl"
    events_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    manifest = load_capture_manifest(tmp_path)
    manifest["events"]["sha256"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    replayed = replay_orchestration(root)
    assert torch.equal(replayed, captured)
    assert replayed.item() != -999


def test_storage_alias_chain_replays_mutations(tmp_path):
    def alias_chain():
        base = torch.zeros(4)
        view = base.view(2, 2)
        base.add_(1)
        view.add_(1)
        return base.clone()

    with ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        captured = session.capture_call(alias_chain)
    replayed = replay_orchestration(tmp_path / "execution_capture_rank_0")
    assert torch.equal(captured, torch.full((4,), 2.0))
    assert torch.equal(replayed, captured)
    view_event = next(
        event
        for event in load_capture_events(tmp_path)
        if event["kind"] == "operation" and event["op"] == "aten.view.default"
    )
    assert view_event["observed_aliases"]


def test_analysis_policy_cannot_publish_complete(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={
            "strict": False,
            "snapshot_operations": False,
            "capture_first_invocation_per_signature": False,
        },
    ) as session:
        session.capture_call(torch.neg, torch.ones(1))
    manifest = load_capture_manifest(tmp_path)
    assert manifest["status"] == "incomplete"
    assert manifest["replayable"] is False
    with pytest.raises(ReplayPreflightError):
        replay_preflight(tmp_path / "execution_capture_rank_0")


def test_worker_outliving_context_and_overridden_run_are_incomplete(tmp_path):
    release = threading.Event()

    class Worker(threading.Thread):
        def run(self) -> None:
            release.wait(timeout=10)
            torch.ones(1).add_(1)

    worker = Worker()
    with ExecutionCaptureSession(
        tmp_path,
        policy={
            "strict": False,
            "capture_root_boundary": False,
            "capture_first_invocation_per_signature": False,
        },
    ):
        worker.start()
    manifest = load_capture_manifest(tmp_path)
    assert manifest["status"] == "incomplete"
    assert any(
        loss["kind"] == "active_workers_at_finalization"
        for loss in manifest["losses"]
    )
    release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert not list(tmp_path.rglob("*.partial"))


def test_preexisting_thread_coverage_is_explicitly_unsupported(tmp_path):
    release = threading.Event()
    worker = threading.Thread(target=lambda: release.wait(timeout=10))
    worker.start()
    try:
        with ExecutionCaptureSession(
            tmp_path,
            policy={
                "strict": False,
                "capture_root_boundary": False,
                "capture_first_invocation_per_signature": False,
            },
        ):
            torch.ones(1)
    finally:
        release.set()
        worker.join(timeout=10)
    manifest = load_capture_manifest(tmp_path)
    assert manifest["status"] == "incomplete"
    assert any(
        loss["kind"] == "preexisting_threads_uninstrumented"
        for loss in manifest["losses"]
    )


def test_actual_snapshot_quota_dedup_and_concurrent_object_limit(tmp_path):
    probe = SnapshotStore(
        tmp_path / "probe",
        max_bytes=10_000_000,
        max_tensors=10,
        max_objects=10,
    )
    value = "x" * (1 << 20)
    probe.save(value, "probe")
    actual = next(iter(probe.entries.values()))["serialized_bytes"]

    exact = SnapshotStore(
        tmp_path / "exact",
        max_bytes=actual,
        max_tensors=10,
        max_objects=1,
    )
    assert exact.save(value, "first") == exact.save(value, "second")
    assert exact.bytes_reserved == actual

    tiny = SnapshotStore(
        tmp_path / "tiny",
        max_bytes=1,
        max_tensors=10,
        max_objects=10,
    )
    with pytest.raises(SnapshotQuotaExceeded):
        tiny.save(value, "too-large")
    assert not list((tmp_path / "tiny").glob("*.partial"))

    raced = SnapshotStore(
        tmp_path / "raced",
        max_bytes=10_000_000,
        max_tensors=10,
        max_objects=1,
    )
    outcomes = []

    def save(item: str) -> None:
        try:
            raced.save(item, item)
            outcomes.append("ok")
        except SnapshotQuotaExceeded:
            outcomes.append("quota")

    threads = [threading.Thread(target=save, args=(item,)) for item in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert sorted(outcomes) == ["ok", "quota"]
    assert raced.object_count == 1


def test_inference_mutation_and_invocation_graph_bindings(tmp_path):
    class Child(torch.nn.Module):
        def forward(self, value: torch.Tensor, *, scale: int) -> torch.Tensor:
            return value.mul_(scale)

    class Parent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = Child()

        def forward(self, value: torch.Tensor, *, scale: int) -> torch.Tensor:
            return self.child(value, scale=scale)

    parent = Parent()
    with torch.inference_mode(), ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        first = session.capture_call(parent, torch.ones(1), scale=2)
    assert first.item() == 2
    events = load_capture_events(tmp_path)
    mutation = next(
        event
        for event in events
        if event["kind"] == "operation" and event["op"] == "aten.mul_.Tensor"
    )
    assert mutation["declared_alias_mutation"]["arguments"][0]["is_write"]
    assert mutation["observed_mutation"]
    starts = [
        event for event in events if event["kind"] == "module_invocation_start"
    ]
    for start in starts:
        path = materialize_invocation(
            tmp_path / "execution_capture_rank_0",
            start["invocation_id"],
        )
        assert path.is_file()
    parent_end = next(
        event
        for event in events
        if event["kind"] == "module_invocation_end"
        and event["invocation_id"] == starts[0]["invocation_id"]
    )
    assert not parent_end["direct_operation_ids"]
    assert parent_end["subtree_operation_ids"]
    operation = mutation
    scale_arg = next(
        item for item in operation["canonical_arguments"] if item["name"] == "other"
    )
    assert scale_arg["value"]["value"] == 2


def test_graph_hash_distinguishes_scalars_and_random_bindings(tmp_path):
    class Scale(torch.nn.Module):
        def forward(self, value: torch.Tensor, scale: int) -> torch.Tensor:
            return value * scale

    class Random(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.rand_like(value)

    scale = Scale()
    random = Random()

    def execute(value):
        scale(value, 2)
        scale(value, 3)
        scale(value, 2)
        random(value)
        return random(value)

    with ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        session.capture_call(execute, torch.ones(2))
    events = load_capture_events(tmp_path)
    starts = [
        event for event in events if event["kind"] == "module_invocation_start"
    ]
    ends = {
        event["invocation_id"]: event
        for event in events
        if event["kind"] == "module_invocation_end"
    }
    scale_graphs = [
        ends[event["invocation_id"]]["graph_id"]
        for event in starts
        if event["module_id"] == starts[0]["module_id"]
    ]
    assert scale_graphs[0] != scale_graphs[1]
    assert scale_graphs[0] == scale_graphs[2]
    random_graphs = [
        ends[event["invocation_id"]]["graph_id"]
        for event in starts
        if event["module_id"] != starts[0]["module_id"]
    ]
    assert len(random_graphs) == 2
    assert random_graphs[0] != random_graphs[1]


def test_generator_resumption_and_handled_exception_scope_semantics(tmp_path):
    program = r"""
import sys
import torch
from xfuser.model_executor.execution_capture import ExecutionCaptureSession
def generator(value):
    yield value + 1
    yield value + 2
def handled(value):
    value = value + 1
    try:
        raise KeyError("handled")
    except KeyError:
        value = value + 1
    return value
with ExecutionCaptureSession(
    sys.argv[1],
    policy={"capture_root_boundary": False, "capture_first_invocation_per_signature": False},
):
    list(generator(torch.tensor([1.0])))
    assert handled(torch.tensor([1.0])).item() == 3
"""
    subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )
    events = load_capture_events(tmp_path)
    definitions = {
        event["callable_id"]: event
        for event in events
        if event["kind"] == "callable_definition"
    }
    generator_starts = [
        event
        for event in events
        if event["kind"] == "scope_start"
        and definitions[event["callable_id"]]["qualified_name"].endswith("generator")
    ]
    assert len(generator_starts) == 1
    handled_end = next(
        event
        for event in events
        if event["kind"] == "scope_end"
        and definitions[
            next(
                start["callable_id"]
                for start in events
                if start["kind"] == "scope_start"
                and start["scope_id"] == event["scope_id"]
            )
        ]["qualified_name"].endswith("handled")
    )
    assert handled_end["status"] == "ok"
    assert handled_end["exception"] is None


def test_kwargs_canonicalization_and_replay(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        captured = session.capture_call(
            torch.add,
            input=torch.tensor([2.0]),
            other=torch.tensor([3.0]),
            alpha=4,
        )
    operation = next(
        event
        for event in load_capture_events(tmp_path)
        if event["kind"] == "operation" and event["op"] == "aten.add.Tensor"
    )
    alpha = next(
        item for item in operation["canonical_arguments"] if item["name"] == "alpha"
    )
    assert alpha["source"] == "keyword"
    assert alpha["value"]["value"] == 4
    replayed = replay_orchestration(tmp_path / "execution_capture_rank_0")
    assert torch.equal(replayed, captured)


def test_preflight_rejects_tampered_artifacts_and_rank(tmp_path):
    with ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        session.capture_call(torch.neg, torch.ones(1))
    root = tmp_path / "execution_capture_rank_0"
    with pytest.raises(ReplayPreflightError, match="rank"):
        replay_preflight(root, expected_rank=1)

    manifest = load_capture_manifest(tmp_path)
    manifest["replayable"] = False
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReplayPreflightError, match="incomplete"):
        replay_preflight(root)


@pytest.mark.parametrize(
    "corruption",
    [
        "no_complete",
        "missing_events",
        "missing_snapshot",
        "events_checksum",
        "schema",
        "device",
        "custom_op",
        "group_alias",
        "missing_graph",
        "graph_schema",
        "graph_checksum",
    ],
)
def test_preflight_corruption_matrix_rejects_before_execution(
    tmp_path, corruption
):
    with ExecutionCaptureSession(
        tmp_path,
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        session.capture_call(torch.neg, torch.ones(1))
    root = tmp_path / "execution_capture_rank_0"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events_path = root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]

    if corruption == "no_complete":
        (root / "COMPLETE").unlink()
    elif corruption == "missing_events":
        events_path.unlink()
    elif corruption == "missing_snapshot":
        (root / "snapshots" / manifest["snapshots"][0]["file"]).unlink()
    elif corruption == "events_checksum":
        events_path.write_text(events_path.read_text() + " ", encoding="utf-8")
    elif corruption == "schema":
        manifest["schema_version"] = "999"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption in {"device", "custom_op", "group_alias"}:
        operation = next(event for event in events if event["kind"] == "operation")
        if corruption == "device":
            operation["device"]["devices"] = ["cuda:99"]
        elif corruption == "custom_op":
            operation["opaque_custom_op"] = True
        else:
            operation["collective"] = {
                "resolved": False,
                "group_id": "missing",
            }
        events_path.write_text(
            "".join(canonical_json(event) + "\n" for event in events),
            encoding="utf-8",
        )
        manifest["events"]["sha256"] = hashlib.sha256(
            events_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    else:
        operation_ids = [
            event["operation_id"]
            for event in events
            if event["kind"] == "operation"
        ]
        graph = materialize_operation_range(root, operation_ids)
        if corruption == "missing_graph":
            graph.unlink()
        else:
            graph_data = json.loads(graph.read_text())
            if corruption == "graph_schema":
                graph_data["schema_version"] = "999"
            else:
                graph_data["graph_id"] = "sha256:wrong"
            graph.write_text(canonical_json(graph_data) + "\n", encoding="utf-8")
        with pytest.raises(ReplayPreflightError):
            replay_graph(root, graph)
        return

    with pytest.raises(ReplayPreflightError):
        replay_preflight(root)


def test_unwritable_setup_rolls_back_global_ownership(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    with pytest.raises(OSError), ExecutionCaptureSession(blocker):
        pass
    with ExecutionCaptureSession(
        tmp_path / "valid",
        policy={"capture_first_invocation_per_signature": False},
    ) as session:
        session.capture_call(torch.neg, torch.ones(1))


@pytest.mark.parametrize("batched", [False, True])
def test_model_run_capture_owns_actual_selected_path(monkeypatch, tmp_path, batched):
    class Fake:
        run = xFuserModel.run

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    model = Fake()
    model.config = SimpleNamespace(
        warmup_calls=0,
        determinism_check=0,
        num_iterations=1,
        batch_size=1 if batched else None,
        capture_execution=True,
    )
    model._validate_args = lambda args: None
    model._split_prompts_for_dp = lambda args: args
    model._gather_dp_outputs = lambda output: output
    calls = []

    def timed(args):
        calls.append("timed")
        return torch.tensor([1]), 0.1

    def batched_call(args):
        calls.append("batched")
        return torch.tensor([1]), [0.1]

    model._run_timed_pipe = timed
    model._run_pipe_batched = batched_call
    selected = []

    def capture(run_callable, args):
        selected.append(run_callable)
        return run_callable(args)

    model._dump_model = capture
    model.run({"prompt": ["x"] if batched else "x"})
    assert calls == ["batched" if batched else "timed"]
    assert selected == [batched_call if batched else timed]
