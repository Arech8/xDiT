"""Dynamic module, Python-scope, and dispatcher execution recorder."""

from __future__ import annotations

import contextlib
import dis
import hashlib
import inspect
import os
import sys
import threading
import time
import traceback
import types
import uuid
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .identity import IdentityRegistry
from .policy import CapturePolicy
from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    CaptureConflictError,
    CaptureFinalizationError,
    Completeness,
    event,
    manifest_status,
)
from .snapshots import SnapshotQuotaExceeded, SnapshotStore
from .values import (
    ValueEncoder,
    qualified_name,
    tensor_metadata,
)
from .writer import JsonlWriter, atomic_write_json, canonical_json

_OWNER_LOCK = threading.RLock()
_ACTIVE_OWNER: weakref.ReferenceType[ExecutionCaptureSession] | None = None


@dataclass
class _Scope:
    frame: types.FrameType
    code_key: tuple[str, int, str]
    active: bool = False
    scope_id: str | None = None
    parent_scope_id: str | None = None
    exception: dict[str, str] | None = None
    adapter_before: dict[str, Any] | None = None


@dataclass
class _Invocation:
    invocation_id: str
    module_id: str
    signature_id: str
    direct_operation_ids: list[str] = field(default_factory=list)
    subtree_operation_ids: list[str] = field(default_factory=list)
    graph_records: list[dict[str, Any]] = field(default_factory=list)


class _ThreadState(threading.local):
    def __init__(self) -> None:
        self.frame_stack: list[types.FrameType] = []
        self.invocation_stack: list[_Invocation] = []
        self.sequence = 0
        self.suppression = 0
        self.dispatch_modes: list[ExecutionDispatchMode] = []


class ExecutionDispatchMode(TorchDispatchMode):
    """Record every dispatcher call; ``types`` is intentionally not filtered."""

    supports_higher_order_operators = True

    def __init__(self, recorder: ExecutionCaptureSession) -> None:
        super().__init__()
        self.recorder = recorder

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        return self.recorder._dispatch(func, args, kwargs or {})


class ExecutionCaptureSession:
    """Capture one rank-local eager execution into versioned atomic artifacts.

    Instrumentation is process-global while active because module construction and
    calls can happen anywhere in the pipeline. A single owner is therefore
    enforced, and all modified callbacks are restored transactionally.
    """

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        *,
        policy: CapturePolicy | dict[str, Any] | str | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        metadata: dict[str, Any] | None = None,
        capture_uuid: str | None = None,
    ) -> None:
        self.policy = (
            policy if isinstance(policy, CapturePolicy) else CapturePolicy.from_json(policy)
        )
        self.rank = self._distributed_rank() if rank is None else rank
        self.world_size = self._distributed_world_size() if world_size is None else world_size
        self.root = Path(output_root) / f"execution_capture_rank_{self.rank}"
        self.metadata = metadata or {}
        self.capture_uuid = capture_uuid or str(uuid.uuid4())
        self.configuration_id = hashlib.sha256(
            canonical_json(
                {
                    "policy": self.policy.as_dict(),
                    "metadata": self.metadata,
                    "world_size": self.world_size,
                }
            ).encode()
        ).hexdigest()
        self.completeness = Completeness()
        self.identities = IdentityRegistry(self.rank)
        self.values = ValueEncoder(self.identities, self._value_unsupported)
        self._thread = _ThreadState()
        self._sequence_lock = threading.Lock()
        self._emit_lock = threading.RLock()
        self._global_sequence = 0
        self._invocation_ordinal = 0
        self._scope_ordinal = 0
        self._operation_ordinal = 0
        self._thread_ordinal = 0
        self._thread_ids: dict[str, str] = {}
        self._owned_workers: set[threading.Thread] = set()
        self._worker_lock = threading.RLock()
        self._active_scopes: dict[int, _Scope] = {}
        self._scope_lock = threading.RLock()
        self._callable_defs: set[tuple[str, int, str]] = set()
        self._module_defs: set[str] = set()
        self._module_state_saved: set[str] = set()
        self._captured_signatures: set[str] = set()
        self._graphs: dict[str, str] = {}
        self._collective_ordinals: dict[tuple[str, str], int] = {}
        self._process_group_ids: dict[str, str] = {}
        self._process_group_records: dict[str, dict[str, Any]] = {}
        self._process_group_ordinal = 0
        self._functional_collective_by_tensor: dict[str, dict[str, Any]] = {}
        self._writer: JsonlWriter | None = None
        self._snapshots: SnapshotStore | None = None
        self._state = "new"
        self._entered = False
        self._finalized = False
        self._closing = False
        self._writer_failed = False
        self._old_profile: Any = None
        self._old_trace: Any = None
        self._old_thread_profile: Any = None
        self._old_thread_trace: Any = None
        self._old_call_impl: Any = None
        self._old_thread_bootstrap: Any = None
        self._captured_call_impl: Any = None
        self._captured_thread_bootstrap: Any = None
        self._profile_callback = self._profile
        self._trace_callback = self._trace
        self._dispatch_mode: ExecutionDispatchMode | None = None
        self._dispatch_entered = False
        self._start_ns = 0
        self._model_exception: BaseException | None = None

    @staticmethod
    def _distributed_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return int(os.environ.get("RANK", "0"))

    @staticmethod
    def _distributed_world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return int(os.environ.get("WORLD_SIZE", "1"))

    def __enter__(self):
        global _ACTIVE_OWNER
        if self._state != "new":
            raise RuntimeError(
                f"ExecutionCaptureSession cannot be reused (state={self._state})"
            )
        self._state = "entering"
        with _OWNER_LOCK:
            owner = _ACTIVE_OWNER() if _ACTIVE_OWNER is not None else None
            if owner is not None:
                self._state = "failed"
                raise CaptureConflictError("another ExecutionCaptureSession is active")
            _ACTIVE_OWNER = weakref.ref(self)

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "COMPLETE").unlink(missing_ok=True)
            self._writer = JsonlWriter(
                self.root / "events.jsonl", self.policy.max_jsonl_bytes
            )
            self._snapshots = SnapshotStore(
                self.root / "snapshots",
                max_bytes=self.policy.max_snapshot_bytes,
                max_tensors=self.policy.max_snapshot_tensors,
                max_objects=self.policy.max_snapshot_objects,
                suppress=self.suppress,
            )
            self._start_ns = time.time_ns()
            self._install_instrumentation()
            self._entered = True
            self._state = "active"
            if not self.policy.snapshot_operations:
                self.loss(
                    "analysis_only_without_operation_snapshots",
                    "operation snapshots are disabled; replay is unavailable",
                    severity="warning",
                    replayable=False,
                )
            self.emit(
                "session_start",
                capture_uuid=self.capture_uuid,
                configuration_id=self.configuration_id,
                policy=self.policy.as_dict(),
                process_id=os.getpid(),
                world_size=self.world_size,
                metadata=self.metadata,
            )
            return self
        except BaseException:
            self._restore_instrumentation()
            self._release_owner()
            if self._writer is not None:
                self._writer.abort()
            self._state = "failed"
            raise

    def __exit__(self, exc_type, exc, traceback_object) -> bool:
        if self._state != "active":
            raise RuntimeError(f"invalid capture exit state: {self._state}")
        self._state = "closing"
        self._model_exception = exc
        try:
            if exc is not None:
                self.loss(
                    "model_execution_exception",
                    f"captured execution raised {type(exc).__name__}: {exc}",
                    severity="error",
                    replayable=False,
                    context={"traceback": "".join(traceback.format_exception(exc))},
                )
            with self._worker_lock:
                active_workers = [
                    worker
                    for worker in self._owned_workers
                    if worker.is_alive() and worker is not threading.current_thread()
                ]
            if active_workers:
                self.loss(
                    "active_workers_at_finalization",
                    "instrumented worker threads outlived the capture context",
                    severity="fatal",
                    replayable=False,
                    context={
                        "threads": [
                            {"name": worker.name, "thread_id": self._thread_id(worker)}
                            for worker in active_workers
                        ]
                    },
                )
            with self._emit_lock:
                self._closing = True
                self._close_remaining_scopes()
                self.emit(
                    "session_end",
                    force=True,
                    status="exception" if exc is not None else "ok",
                    duration_ns=time.time_ns() - self._start_ns,
                )
        finally:
            self._restore_instrumentation()
            self._release_owner()

        finalization_error: BaseException | None = None
        try:
            self._finalize(success=exc is None)
        except BaseException as caught:  # noqa: BLE001 - preserve a model exception
            finalization_error = caught
        if exc is not None:
            return False
        if finalization_error is not None:
            raise finalization_error
        self._state = "closed"
        return False

    def capture_call(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Capture a root boundary and invoke ``function`` exactly once."""

        if not self._entered:
            raise RuntimeError("capture_call requires an entered session")
        if self.policy.capture_root_boundary:
            self._snapshot_value((args, kwargs), "root_inputs", required=True)
        result = function(*args, **kwargs)
        if self.policy.capture_root_boundary:
            self._snapshot_value(result, "root_outputs", required=True)
        try:
            leaves, _ = torch.utils._pytree.tree_flatten(result)
        except Exception:  # noqa: BLE001 - custom root output
            leaves = [result]
        if any(
            not isinstance(value, (torch.Tensor, str, int, float, bool, type(None)))
            for value in leaves
        ):
            self.loss(
                "opaque_root_output_transition",
                "root output contains native/Python values without a replay adapter",
                severity="fatal",
                replayable=False,
            )
        return result

    @contextlib.contextmanager
    def suppress(self) -> Iterator[None]:
        self._thread.suppression += 1
        try:
            yield
        finally:
            self._thread.suppression -= 1

    def _install_instrumentation(self) -> None:
        self._old_profile = sys.getprofile()
        self._old_trace = sys.gettrace()
        self._old_thread_profile = threading.getprofile()
        self._old_thread_trace = threading.gettrace()
        conflicts = [
            name
            for name, callback in (
                ("sys profile", self._old_profile),
                ("sys trace", self._old_trace),
                ("thread profile", self._old_thread_profile),
                ("thread trace", self._old_thread_trace),
            )
            if callback is not None
        ]
        if conflicts and self.policy.fail_on_profiler_conflict:
            raise CaptureConflictError(
                "execution capture cannot replace existing callback(s): "
                + ", ".join(conflicts)
            )
        if conflicts:
            self.loss(
                "profiler_callback_replaced",
                "existing callbacks were temporarily replaced",
                severity="warning",
                replayable=True,
                context={"callbacks": conflicts},
            )

        self._old_call_impl = torch.nn.Module._call_impl
        self._old_thread_bootstrap = threading.Thread._bootstrap_inner
        session_ref = weakref.ref(self)
        original_call_impl = self._old_call_impl
        original_thread_bootstrap = self._old_thread_bootstrap

        def captured_call_impl(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
            session = session_ref()
            if session is None or session._thread.suppression:
                return original_call_impl(module, *args, **kwargs)
            invocation = session._module_enter(module, args, kwargs)
            try:
                output = original_call_impl(module, *args, **kwargs)
            except BaseException as caught:
                session._module_exit(invocation, status="exception", exception=caught)
                raise
            session._module_exit(invocation, status="ok", output=output)
            return output

        def captured_thread_bootstrap(thread: threading.Thread) -> Any:
            session = session_ref()
            if session is None or session._closing:
                return original_thread_bootstrap(thread)
            with session._worker_lock:
                session._owned_workers.add(thread)
            mode = ExecutionDispatchMode(session)
            session._thread.dispatch_modes.append(mode)
            try:
                with mode:
                    return original_thread_bootstrap(thread)
            finally:
                session._thread.dispatch_modes.pop()
                with session._worker_lock:
                    session._owned_workers.discard(thread)

        self._captured_call_impl = captured_call_impl
        self._captured_thread_bootstrap = captured_thread_bootstrap
        torch.nn.Module._call_impl = captured_call_impl
        threading.Thread._bootstrap_inner = captured_thread_bootstrap
        sys.setprofile(self._profile_callback)
        threading.setprofile(self._profile_callback)
        sys.settrace(self._trace_callback)
        threading.settrace(self._trace_callback)
        if hasattr(threading, "setprofile_all_threads"):
            threading.setprofile_all_threads(self._profile_callback)
        preexisting_threads = [
            thread
            for thread in threading.enumerate()
            if (
                thread is not threading.current_thread()
                and thread.is_alive()
                and not isinstance(thread, threading._DummyThread)
            )
        ]
        if preexisting_threads:
            self.loss(
                "preexisting_threads_uninstrumented",
                "pre-existing threads cannot receive this session's dispatch mode",
                severity="warning",
                replayable=False,
                context={
                    "threads": [thread.name for thread in preexisting_threads],
                    "python_has_setprofile_all_threads": hasattr(
                        threading, "setprofile_all_threads"
                    ),
                },
            )
        self._dispatch_mode = ExecutionDispatchMode(self)
        self._thread.dispatch_modes.append(self._dispatch_mode)
        self._dispatch_mode.__enter__()
        self._dispatch_entered = True

    def _restore_instrumentation(self) -> None:
        if self._dispatch_mode is not None:
            try:
                if self._dispatch_entered:
                    self._dispatch_mode.__exit__(None, None, None)
            finally:
                self._dispatch_entered = False
                self._dispatch_mode = None
                if self._thread.dispatch_modes:
                    self._thread.dispatch_modes.pop()
        if (
            self._old_call_impl is not None
            and torch.nn.Module._call_impl is self._captured_call_impl
        ):
            torch.nn.Module._call_impl = self._old_call_impl
        self._old_call_impl = None
        if (
            self._old_thread_bootstrap is not None
            and threading.Thread._bootstrap_inner
            is self._captured_thread_bootstrap
        ):
            threading.Thread._bootstrap_inner = self._old_thread_bootstrap
        self._old_thread_bootstrap = None
        if (
            hasattr(threading, "setprofile_all_threads")
            and sys.getprofile() == self._profile_callback
        ):
            threading.setprofile_all_threads(self._old_thread_profile)
        if sys.getprofile() == self._profile_callback:
            sys.setprofile(self._old_profile)
        if sys.gettrace() == self._trace_callback:
            sys.settrace(self._old_trace)
        if threading.getprofile() == self._profile_callback:
            threading.setprofile(self._old_thread_profile)
        if threading.gettrace() == self._trace_callback:
            threading.settrace(self._old_thread_trace)

    def _release_owner(self) -> None:
        global _ACTIVE_OWNER
        with _OWNER_LOCK:
            _ACTIVE_OWNER = None

    def _profile(self, frame: types.FrameType, event_name: str, arg: Any) -> None:
        if event_name == "call":
            if (
                "/execution_capture/" not in frame.f_code.co_filename
                and not any(item is frame for item in self._thread.frame_stack)
            ):
                self._thread.frame_stack.append(frame)
                adapter = self._scope_adapter(frame)
                if adapter is not None:
                    parent = next(
                        (
                            self._active_scopes[id(item)].scope_id
                            for item in reversed(self._thread.frame_stack[:-1])
                            if id(item) in self._active_scopes
                            and self._active_scopes[id(item)].frame is item
                        ),
                        None,
                    )
                    self._activate_frame(frame, parent, adapter=adapter)
        elif event_name == "return":
            if self._is_suspension(frame):
                return
            for index in range(len(self._thread.frame_stack) - 1, -1, -1):
                if self._thread.frame_stack[index] is frame:
                    self._thread.frame_stack.pop(index)
                    break
            self._scope_return(frame, arg)
        elif (
            not self._thread.suppression
            and event_name in {"c_call", "c_exception"}
        ):
            self._profile_native_event(event_name, arg)

    def _trace(
        self, frame: types.FrameType, event_name: str, arg: Any
    ) -> Callable[..., Any] | None:
        if "/execution_capture/" in frame.f_code.co_filename:
            return None
        if (
            event_name == "call"
            and not any(item is frame for item in self._thread.frame_stack)
        ):
            self._thread.frame_stack.append(frame)
            adapter = self._scope_adapter(frame)
            if adapter is not None:
                parent = next(
                    (
                        self._active_scopes[id(item)].scope_id
                        for item in reversed(self._thread.frame_stack[:-1])
                        if id(item) in self._active_scopes
                        and self._active_scopes[id(item)].frame is item
                    ),
                    None,
                )
                self._activate_frame(frame, parent, adapter=adapter)
        if event_name == "exception":
            with self._scope_lock:
                scope = self._active_scopes.get(id(frame))
            if scope is not None and scope.frame is frame:
                exception_type, exception, _ = arg
                scope.exception = {
                    "type": qualified_name(exception_type),
                    "message": str(exception),
                }
        elif event_name in {"line", "opcode"}:
            with self._scope_lock:
                scope = self._active_scopes.get(id(frame))
            if scope is not None and scope.frame is frame:
                scope.exception = None
        elif event_name == "return":
            if self._is_suspension(frame):
                return self._trace
            for index in range(len(self._thread.frame_stack) - 1, -1, -1):
                if self._thread.frame_stack[index] is frame:
                    self._thread.frame_stack.pop(index)
                    break
            self._scope_return(frame, arg)
        # Full call tracing is intentional during the bounded acquisition run:
        # Python 3.10 has no API to attach a profiler to arbitrary live frames,
        # and returning None here loses nested scheduler/processor scopes.
        return self._trace

    @staticmethod
    def _is_suspension(frame: types.FrameType) -> bool:
        offset = frame.f_lasti
        if offset < 0 or offset >= len(frame.f_code.co_code):
            return False
        return dis.opname[frame.f_code.co_code[offset]] in {
            "YIELD_VALUE",
            "YIELD_FROM",
        }

    def _activate_scopes(self) -> str | None:
        frame = sys._getframe(1)
        frames: list[types.FrameType] = list(self._thread.frame_stack)
        known_frames = {id(item) for item in frames}
        while frame is not None:
            if (
                "/execution_capture/" not in frame.f_code.co_filename
                and id(frame) not in known_frames
            ):
                frames.append(frame)
                known_frames.add(id(frame))
            frame = frame.f_back
        # Profiled frames are already outer-to-inner. Frames discovered through
        # ``f_back`` are inner-to-outer and are needed only for callers that
        # predate profiler installation.
        profiled_count = len(self._thread.frame_stack)
        frames[profiled_count:] = reversed(frames[profiled_count:])
        parent: str | None = None
        for frame in frames:
            with self._scope_lock:
                scope = self._active_scopes.get(id(frame))
            if scope is not None and scope.frame is frame:
                parent = scope.scope_id
                continue
            scope = self._activate_frame(frame, parent)
            parent = scope.scope_id
        return parent

    def _activate_frame(
        self,
        frame: types.FrameType,
        parent_scope_id: str | None,
        *,
        adapter: dict[str, Any] | None = None,
    ) -> _Scope:
        with self._scope_lock:
            existing = self._active_scopes.get(id(frame))
        if existing is not None and existing.frame is frame:
            return existing
        code_key = (
            frame.f_code.co_filename,
            frame.f_code.co_firstlineno,
            getattr(frame.f_code, "co_qualname", frame.f_code.co_name),
        )
        scope = _Scope(frame=frame, code_key=code_key, active=True)
        scope.scope_id = f"r{self.rank}:scope:{self._scope_ordinal}"
        self._scope_ordinal += 1
        scope.parent_scope_id = parent_scope_id
        with self._scope_lock:
            self._active_scopes[id(frame)] = scope
        code = frame.f_code
        callable_id = (
            f"r{self.rank}:callable:"
            + hashlib.sha256(canonical_json(scope.code_key).encode()).hexdigest()[:20]
        )
        if scope.code_key not in self._callable_defs:
            self._callable_defs.add(scope.code_key)
            self.emit(
                "callable_definition",
                callable_id=callable_id,
                qualified_name=getattr(code, "co_qualname", code.co_name),
                filename=code.co_filename,
                first_line=code.co_firstlineno,
                signature=self._frame_signature(frame),
            )
        adapter_before = adapter if adapter is not None else self._scope_adapter(frame)
        scope.adapter_before = adapter_before
        self.emit(
            "scope_start",
            scope_id=scope.scope_id,
            parent_scope_id=scope.parent_scope_id,
            callable_id=callable_id,
            callable_instance_id=self._callable_instance_id(frame, callable_id),
            thread_id=self._thread_id(),
            adapter=adapter_before,
        )
        self._capture_frame_boundary(scope, callable_id)
        frame.f_trace = self._trace
        return scope

    def _callable_instance_id(
        self, frame: types.FrameType, callable_definition_id: str
    ) -> str:
        owner = frame.f_locals.get("self")
        if owner is not None:
            instance_key = self.identities.object(owner)
        else:
            closure_values = [
                frame.f_locals[name]
                for name in frame.f_code.co_freevars
                if name in frame.f_locals
            ]
            instance_key = "|".join(
                self.identities.object(value) for value in closure_values
            )
        digest = hashlib.sha256(
            f"{callable_definition_id}|{instance_key}".encode()
        ).hexdigest()[:20]
        return f"r{self.rank}:callable-instance:{digest}"

    def _scope_return(self, frame: types.FrameType, result: Any) -> None:
        with self._scope_lock:
            scope = self._active_scopes.get(id(frame))
            if scope is not None and scope.frame is frame:
                del self._active_scopes[id(frame)]
        if scope is None or scope.frame is not frame:
            return
        if scope.active:
            adapter_after = self._scope_adapter(frame)
            changed_keys = []
            if scope.adapter_before and adapter_after:
                before_state = scope.adapter_before.get("state", {})
                after_state = adapter_after.get("state", {})
                changed_keys = sorted(
                    key
                    for key in set(before_state) | set(after_state)
                    if before_state.get(key) != after_state.get(key)
                )
            self.emit(
                "scope_end",
                scope_id=scope.scope_id,
                status="exception" if scope.exception else "ok",
                exception=scope.exception,
                result=self.values.encode(result),
                adapter_after=adapter_after,
                adapter_changed_keys=changed_keys,
                thread_id=self._thread_id(),
            )

    def _frame_signature(self, frame: types.FrameType) -> str:
        code = frame.f_code
        count = code.co_argcount + code.co_kwonlyargcount
        names = list(code.co_varnames[:count])
        has_varargs = bool(code.co_flags & inspect.CO_VARARGS)
        if has_varargs and count < len(code.co_varnames):
            names.append("*" + code.co_varnames[count])
        keyword_index = count + int(has_varargs)
        if (
            code.co_flags & inspect.CO_VARKEYWORDS
            and keyword_index < len(code.co_varnames)
        ):
            names.append("**" + code.co_varnames[keyword_index])
        return "(" + ", ".join(names) + ")"

    def _capture_frame_boundary(self, scope: _Scope, callable_id: str) -> None:
        if not self.policy.capture_first_invocation_per_signature:
            return
        signature = self._value_signature(tuple(scope.frame.f_locals.values()))
        signature_id = self._signature_id(callable_id, signature)
        if signature_id in self._captured_signatures:
            return
        self._captured_signatures.add(signature_id)
        self._snapshot_value(
            dict(scope.frame.f_locals),
            f"scope_inputs:{scope.scope_id}",
            required=False,
        )

    def _profile_native_event(self, event_name: str, function: Any) -> None:
        module = getattr(function, "__module__", "") or ""
        name = (
            getattr(function, "__qualname__", getattr(function, "__name__", "")) or ""
        )
        if module.startswith(("torch.cuda", "torch.distributed")) or name in {
            "numpy",
            "record",
            "wait",
            "synchronize",
        }:
            self.emit(
                "native_api",
                status="exception" if event_name == "c_exception" else "call",
                callable=f"{module}.{name}",
                scope_id=self._current_scope_id(),
                severity="warning",
                replayable=False,
            )

    def _scope_adapter(self, frame: types.FrameType) -> dict[str, Any] | None:
        name = frame.f_code.co_name.lower()
        owner = frame.f_locals.get("self")
        owner_name = type(owner).__qualname__.lower() if owner is not None else ""
        category: str | None = None
        if "scheduler" in owner_name or name in {"step", "scale_model_input"}:
            category = "scheduler"
        elif name in {"encode", "decode"} and any(
            token in owner_name for token in ("vae", "autoencoder", "codec")
        ):
            category = "vae"
        elif name in {"postprocess", "preprocess"} or "processor" in owner_name:
            category = "processor"
        elif "pipeline" in owner_name:
            category = "pipeline_control"
        if category is None:
            return None
        state: dict[str, Any] = {}
        for attribute in ("timesteps", "sigmas", "_step_index", "config"):
            if owner is not None and hasattr(owner, attribute):
                value = getattr(owner, attribute)
                state[attribute] = self.values.encode(value)
        return {"category": category, "state": state}

    def _module_enter(
        self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> _Invocation:
        if hasattr(module, "_orig_mod") or type(module).__module__.startswith(
            "torch._dynamo"
        ):
            self.loss(
                "opaque_compiled_region",
                f"compiled module {qualified_name(type(module))} is not semantic capture",
                severity="error",
                replayable=False,
            )
            if self.policy.strict:
                raise CaptureConflictError(
                    "strict semantic capture encountered a compiled module; "
                    "unwrap or disable torch.compile"
                )
        scope_id = self._activate_scopes()
        module_id = self.identities.module(module)
        if module_id not in self._module_defs:
            self._module_defs.add(module_id)
            self.emit(
                "module_definition",
                module_id=module_id,
                python_type=qualified_name(type(module)),
                training=bool(module.training),
            )
            if self.policy.capture_parameters:
                self._snapshot_module_state(module, module_id)
        signature = self._value_signature((args, kwargs))
        signature_id = self._signature_id(module_id, signature)
        invocation_id = f"r{self.rank}:invocation:{self._invocation_ordinal}"
        self._invocation_ordinal += 1
        parent = (
            self._thread.invocation_stack[-1].invocation_id
            if self._thread.invocation_stack
            else None
        )
        invocation = _Invocation(invocation_id, module_id, signature_id)
        self._thread.invocation_stack.append(invocation)
        snapshot = None
        if (
            self.policy.capture_first_invocation_per_signature
            and signature_id not in self._captured_signatures
        ):
            self._captured_signatures.add(signature_id)
            snapshot = self._snapshot_value(
                (args, kwargs), f"invocation_inputs:{invocation_id}", required=False
            )
        self.emit(
            "module_invocation_start",
            invocation_id=invocation_id,
            module_id=module_id,
            signature_id=signature_id,
            parent_invocation_id=parent,
            scope_id=scope_id,
            thread_id=self._thread_id(),
            arguments=self.values.encode((args, kwargs)),
            snapshot=snapshot,
        )
        return invocation

    def _module_exit(
        self,
        invocation: _Invocation,
        *,
        status: str,
        output: Any = None,
        exception: BaseException | None = None,
    ) -> None:
        if self._thread.invocation_stack:
            if self._thread.invocation_stack[-1] is invocation:
                self._thread.invocation_stack.pop()
            else:
                self.loss(
                    "invocation_stack_corruption",
                    "module invocation exited out of stack order",
                    severity="error",
                    replayable=False,
                    context={"invocation_id": invocation.invocation_id},
                )
                self._thread.invocation_stack = [
                    item for item in self._thread.invocation_stack if item is not invocation
                ]
        graph_id = self._record_graph_candidate(invocation)
        self.emit(
            "module_invocation_end",
            invocation_id=invocation.invocation_id,
            status=status,
            result=self.values.encode(output) if status == "ok" else None,
            exception=(
                {
                    "type": qualified_name(type(exception)),
                    "message": str(exception),
                }
                if exception is not None
                else None
            ),
            graph_id=graph_id,
            direct_operation_ids=invocation.direct_operation_ids,
            subtree_operation_ids=invocation.subtree_operation_ids,
            thread_id=self._thread_id(),
        )
        self.emit(
            "invocation_binding",
            invocation_id=invocation.invocation_id,
            graph_id=graph_id,
            direct_operation_ids=invocation.direct_operation_ids,
            subtree_operation_ids=invocation.subtree_operation_ids,
        )

    def _snapshot_module_state(self, module: torch.nn.Module, module_id: str) -> None:
        if module_id in self._module_state_saved:
            return
        self._module_state_saved.add(module_id)
        with self.suppress():
            state = {
                "parameters": {
                    name: parameter.detach()
                    for name, parameter in module.named_parameters(recurse=False)
                },
                "buffers": {
                    name: buffer.detach()
                    for name, buffer in module.named_buffers(recurse=False)
                    if buffer is not None
                },
            }
        if state["parameters"] or state["buffers"]:
            reference = self._snapshot_value(
                state, f"module_state:{module_id}", required=True
            )
            self.emit("module_state", module_id=module_id, snapshot=reference)

    def _dispatch(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if self._thread.suppression or self._closing:
            return func(*args, **kwargs)
        scope_id = self._activate_scopes()
        operation_id = f"r{self.rank}:operation:{self._operation_ordinal}"
        self._operation_ordinal += 1
        overload = self._operation_name(func)
        collective = self._collective_metadata(overload, args, kwargs)
        metadata_args, metadata_kwargs = (
            self._sanitize_collective_value((args, kwargs))
            if collective is not None
            else (args, kwargs)
        )
        input_tensors = self._tensors((args, kwargs))
        schema = getattr(func, "_schema", None)
        declared = self._schema_aliases(schema)
        with self.suppress():
            encoded_arguments = self.values.encode((metadata_args, metadata_kwargs))
            canonical_arguments = self._canonical_arguments(
                schema, metadata_args, metadata_kwargs
            )
            pre = {
                self.identities.tensor(tensor): tensor_metadata(
                    tensor, self.identities
                )
                for tensor in input_tensors
            }
        invocation_id = (
            self._thread.invocation_stack[-1].invocation_id
            if self._thread.invocation_stack
            else None
        )
        start = time.time_ns()
        argument_snapshot = None
        if self.policy.snapshot_operations:
            snapshot_value = (
                self._sanitize_collective_value((args, kwargs))
                if collective is not None
                else (args, kwargs)
            )
            argument_snapshot = self._snapshot_value(
                snapshot_value,
                f"operation_inputs:{operation_id}",
                required=True,
            )
        base_record = {
            "operation_id": operation_id,
            "op": overload,
            "schema": str(schema) if schema is not None else None,
            "arguments": encoded_arguments,
            "canonical_arguments": canonical_arguments,
            "argument_snapshot": argument_snapshot,
            "invocation_id": invocation_id,
            "scope_id": scope_id,
            "thread_id": self._thread_id(),
            "thread_sequence": self._next_thread_sequence(),
            "declared_alias_mutation": declared,
            "pre_tensor_metadata": list(pre.values()),
            "ambient": {
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode_enabled": torch.is_inference_mode_enabled(),
                "autocast_enabled": torch.is_autocast_enabled(),
            },
        }
        try:
            # PyTorch has already popped this mode before invoking
            # ``__torch_dispatch__``. Calling directly preserves every lower mode.
            result = func(*args, **kwargs)
        except BaseException as caught:
            failed_record = {
                **base_record,
                "status": "exception",
                "result": None,
                "result_snapshot": None,
                "exception": {
                    "type": qualified_name(type(caught)),
                    "message": str(caught),
                },
                "timing": {
                    "start_ns": start,
                    "duration_ns": time.time_ns() - start,
                },
                "device": self._device_metadata(input_tensors),
                "post_tensor_metadata": [
                    tensor_metadata(tensor, self.identities)
                    for tensor in input_tensors
                ],
                "observed_mutation": self._observed_mutation(
                    pre, input_tensors, declared
                ),
                "observed_aliases": [],
                "collective": collective,
                "opaque_custom_op": schema is None,
            }
            self.loss(
                "failed_dispatch_operation",
                f"operation {overload} raised {type(caught).__name__}",
                severity="error",
                replayable=False,
                context={"operation_id": operation_id},
            )
            self.emit("operation", **failed_record)
            self._bind_operation_to_invocations(failed_record)
            raise
        output_tensors = self._tensors(result)
        result_snapshot = None
        if self.policy.snapshot_operations and collective is None:
            result_snapshot = self._snapshot_value(
                result, f"operation_outputs:{operation_id}", required=True
            )
        is_higher_order = schema is None
        opaque = not overload.startswith(
            ("aten.", "prims.", "_c10d_functional.", "c10d.")
        )
        if opaque:
            self.loss(
                (
                    "unsupported_higher_order_operator"
                    if is_higher_order
                    else "opaque_custom_operator"
                ),
                f"operator {overload} is not semantically replayable",
                severity="warning",
                replayable=False,
                context={"operation_id": operation_id},
            )
        operation_record = {
            **base_record,
            "status": "ok",
            "result": (
                self.values.encode(result)
                if output_tensors or collective is None
                else {
                    "type": "collective_internal_work",
                    "async_requested": collective.get("async_requested"),
                }
            ),
            "result_snapshot": result_snapshot,
            "exception": None,
            "timing": {"start_ns": start, "duration_ns": time.time_ns() - start},
            "device": self._device_metadata(input_tensors + output_tensors),
            "pre_tensor_metadata": list(pre.values()),
            "post_tensor_metadata": [
                tensor_metadata(tensor, self.identities)
                for tensor in input_tensors + output_tensors
            ],
            "observed_mutation": self._observed_mutation(
                pre, input_tensors, declared
            ),
            "observed_aliases": self._observed_aliases(
                input_tensors, output_tensors
            ),
            "collective": collective,
            "opaque_custom_op": opaque,
        }
        if collective is not None and output_tensors and not collective.get("is_wait"):
            producer = {
                **collective,
                "collective_operation_id": operation_id,
            }
            for tensor in output_tensors:
                self._functional_collective_by_tensor[
                    self.identities.tensor(tensor)
                ] = producer
        elif collective is None and output_tensors:
            for output_tensor in output_tensors:
                output_storage = self.identities.storage(output_tensor)
                for input_tensor in input_tensors:
                    pending = self._functional_collective_by_tensor.get(
                        self.identities.tensor(input_tensor)
                    )
                    if (
                        pending is not None
                        and (
                            "_wrap_tensor" in overload
                            or (
                                output_storage is not None
                                and output_storage
                                == self.identities.storage(input_tensor)
                            )
                        )
                    ):
                        self._functional_collective_by_tensor[
                            self.identities.tensor(output_tensor)
                        ] = pending
                        break
        self.emit("operation", **operation_record)
        self._bind_operation_to_invocations(operation_record)
        return result

    def _sanitize_collective_value(self, value: Any) -> Any:
        process_group_type = getattr(torch.distributed, "ProcessGroup", ())
        if process_group_type and isinstance(value, process_group_type):
            object_identity = self.identities.object(value)
            return {
                "__xdit_process_group__": self._process_group_ids.get(
                    object_identity
                )
            }
        if isinstance(value, torch.ScriptObject):
            try:
                qualified = value._type().qualified_name()
            except Exception:  # noqa: BLE001 - torchbind inspection
                qualified = type(value).__qualname__
            return {"__xdit_torchbind__": qualified}
        if isinstance(value, tuple):
            return tuple(self._sanitize_collective_value(item) for item in value)
        if isinstance(value, list):
            return [self._sanitize_collective_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._sanitize_collective_value(item)
                for key, item in value.items()
            }
        return value

    def _canonical_arguments(
        self, schema: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if schema is None:
            return [{"name": "*args", "value": self.values.encode((args, kwargs))}]
        canonical: list[dict[str, Any]] = []
        positional_index = 0
        for argument in schema.arguments:
            name = argument.name
            if not argument.kwarg_only and positional_index < len(args):
                value = args[positional_index]
                positional_index += 1
                source = "positional"
            elif name in kwargs:
                value = kwargs[name]
                source = "keyword"
            elif argument.has_default_value():
                value = argument.default_value
                source = "default"
            else:
                canonical.append(
                    {"name": name, "source": "missing", "value": None}
                )
                continue
            canonical.append(
                {
                    "name": name,
                    "source": source,
                    "value": self.values.encode(value),
                }
            )
        return canonical

    def _bind_operation_to_invocations(
        self, operation_record: dict[str, Any]
    ) -> None:
        stack = self._thread.invocation_stack
        if not stack:
            return
        operation_id = operation_record["operation_id"]
        semantic = self._operation_semantic(operation_record)
        for invocation in stack:
            if (
                len(invocation.subtree_operation_ids)
                >= self.policy.max_index_entries
            ):
                self.loss(
                    "invocation_index_overflow",
                    "invocation operation index exceeded policy limit",
                    severity="fatal",
                    replayable=False,
                    context={"invocation_id": invocation.invocation_id},
                )
                continue
            invocation.subtree_operation_ids.append(operation_id)
            invocation.graph_records.append(semantic)
        stack[-1].direct_operation_ids.append(operation_id)

    @staticmethod
    def _operation_semantic(operation: dict[str, Any]) -> dict[str, Any]:
        tensor_slots: dict[str, int] = {}
        storage_slots: dict[str, int] = {}

        def normalize(value: Any) -> Any:
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if not isinstance(value, dict):
                return value
            if value.get("type") == "tensor_ref":
                tensor_id = value["tensor_id"]
                storage_id = value.get("storage_id")
                tensor_slot = tensor_slots.setdefault(tensor_id, len(tensor_slots))
                storage_slot = (
                    storage_slots.setdefault(storage_id, len(storage_slots))
                    if storage_id is not None
                    else None
                )
                return {
                    "type": "tensor",
                    "tensor_slot": tensor_slot,
                    "storage_slot": storage_slot,
                    "python_type": value.get("python_type"),
                    "dtype": value.get("dtype"),
                    "device": value.get("device"),
                    "shape": value.get("shape"),
                    "stride": value.get("stride"),
                    "storage_offset": value.get("storage_offset"),
                }
            return {
                key: normalize(item)
                for key, item in value.items()
                if key not in {"tensor_id", "storage_id", "object_id"}
            }

        op_name = operation["op"]
        semantic = {
            "op": op_name,
            "schema": operation.get("schema"),
            "canonical_arguments": normalize(operation.get("canonical_arguments")),
            "declared_alias_mutation": operation.get("declared_alias_mutation"),
            "collective": normalize(operation.get("collective")),
            "ambient": operation.get("ambient"),
        }
        if any(token in op_name for token in ("rand", "normal", "bernoulli")):
            result_snapshot = operation.get("result_snapshot") or {}
            semantic["random_result_sha256"] = result_snapshot.get("sha256")
        return semantic

    @staticmethod
    def _operation_name(func: Any) -> str:
        # ``str(OpOverload)`` is the resolvable ``namespace.packet.overload``
        # spelling; ``name()`` uses ``namespace::packet.overload``.
        return str(func)

    def _tensors(self, value: Any) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        seen: set[int] = set()
        try:
            leaves, _ = torch.utils._pytree.tree_flatten(value)
        except Exception as exc:  # noqa: BLE001 - custom pytrees are user code
            self.loss(
                "pytree_traversal_failed",
                f"unable to flatten operation values: {exc}",
                severity="error",
                replayable=False,
            )
            leaves = [value]
        for item in leaves:
            if isinstance(item, torch.Tensor) and id(item) not in seen:
                seen.add(id(item))
                tensors.append(item)
        return tensors

    @staticmethod
    def _schema_aliases(schema: Any) -> dict[str, Any]:
        if schema is None:
            return {"arguments": [], "returns": []}

        def alias(item: Any) -> dict[str, Any]:
            info = getattr(item, "alias_info", None)
            return {
                "name": getattr(item, "name", ""),
                "type": str(getattr(item, "type", "")),
                "is_write": bool(getattr(info, "is_write", False)),
                "before_set": sorted(getattr(info, "before_set", set())),
                "after_set": sorted(getattr(info, "after_set", set())),
            }

        return {
            "arguments": [alias(item) for item in schema.arguments],
            "returns": [alias(item) for item in schema.returns],
        }

    def _observed_mutation(
        self,
        pre: dict[str, dict[str, Any]],
        inputs: list[torch.Tensor],
        declared: dict[str, Any],
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for tensor in inputs:
            tensor_id = self.identities.tensor(tensor)
            before = pre.get(tensor_id)
            after = tensor_metadata(tensor, self.identities)
            if before != after:
                fields = sorted(
                    key
                    for key in set(before or {}) | set(after)
                    if (before or {}).get(key) != after.get(key)
                )
                changes.append({"tensor_id": tensor_id, "changed_fields": fields})
        if any(item.get("is_write") for item in declared.get("arguments", [])):
            changed_ids = {item["tensor_id"] for item in changes}
            for tensor in inputs:
                tensor_id = self.identities.tensor(tensor)
                if tensor_id not in changed_ids:
                    changes.append(
                        {
                            "tensor_id": tensor_id,
                            "changed_fields": [],
                            "content_mutation": "unknown",
                            "reason": "schema declares mutation but metadata/version did not prove it",
                        }
                    )
        return changes

    def _observed_aliases(
        self, inputs: list[torch.Tensor], outputs: list[torch.Tensor]
    ) -> list[dict[str, str]]:
        aliases: list[dict[str, str]] = []
        for output in outputs:
            output_storage = self.identities.storage(output)
            for input_tensor in inputs:
                if (
                    output_storage is not None
                    and output_storage == self.identities.storage(input_tensor)
                ):
                    aliases.append(
                        {
                            "input_tensor_id": self.identities.tensor(input_tensor),
                            "output_tensor_id": self.identities.tensor(output),
                            "storage_id": output_storage,
                        }
                    )
        return aliases

    def _device_metadata(self, tensors: list[torch.Tensor]) -> dict[str, Any]:
        devices = sorted({str(tensor.device) for tensor in tensors})
        streams: dict[str, Any] = {}
        with self.suppress():
            for tensor in tensors:
                if tensor.device.type == "cuda":
                    try:
                        stream = torch.cuda.current_stream(tensor.device)
                        streams[str(tensor.device)] = {
                            "stream_id": int(stream.stream_id),
                            "device_index": stream.device_index,
                        }
                    except Exception as exc:  # noqa: BLE001 - optional device metadata
                        streams[str(tensor.device)] = {"error": repr(exc)}
        return {"devices": devices, "current_streams": streams}

    def _collective_metadata(
        self, overload: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        if any(
            marker in overload
            for marker in ("_wrap_tensor", "_unwrap_tensor")
        ):
            return None
        if "c10d" not in overload and not any(
            token in overload
            for token in ("all_reduce", "all_gather", "all_to_all", "broadcast")
        ):
            return None
        if "wait_tensor" in overload:
            tensor = next(
                (value for value in self._tensors((args, kwargs))),
                None,
            )
            pending = (
                self._functional_collective_by_tensor.get(
                    self.identities.tensor(tensor)
                )
                if tensor is not None
                else None
            )
            if pending is None:
                self.loss(
                    "unresolved_collective_wait",
                    "functional collective wait has no captured producer",
                    severity="fatal",
                    replayable=False,
                )
                return {"resolved": False, "op": overload, "wait_for": None}
            return {
                **pending,
                "op": overload,
                "wait_for": pending["collective_operation_id"],
                "is_wait": True,
            }
        process_group_type = getattr(torch.distributed, "ProcessGroup", ())
        group = None
        registered_name = None
        try:
            leaves, _ = torch.utils._pytree.tree_flatten((args, kwargs))
        except Exception:  # noqa: BLE001 - metadata only
            leaves = list(args) + list(kwargs.values())
        for value in leaves:
            if process_group_type and isinstance(value, process_group_type):
                group = value
                break
            if isinstance(value, torch.ScriptObject):
                try:
                    if (
                        value._type().qualified_name()
                        == "__torch__.torch.classes.c10d.ProcessGroup"
                    ):
                        group = torch.distributed.ProcessGroup.unbox(value)
                        break
                except Exception:  # noqa: BLE001,S110 - optional torchbind probe
                    pass
            if isinstance(value, str):
                try:
                    from torch.distributed.distributed_c10d import _world

                    candidate = next(
                        (
                            pg
                            for pg, name in _world.pg_names.items()
                            if name == value
                        ),
                        None,
                    )
                except Exception:  # noqa: BLE001 - private API varies
                    candidate = None
                if candidate is not None:
                    group = candidate
                    registered_name = value
                    break
        if group is None:
            self.loss(
                "unresolved_collective_group",
                f"collective {overload} has no resolvable process group",
                severity="fatal",
                replayable=False,
            )
            return {
                "group_id": None,
                "local_ordinal": None,
                "op": overload,
                "resolved": False,
            }
        object_identity = self.identities.object(group)
        group_id = self._process_group_ids.get(object_identity)
        if group_id is None:
            group_id = (
                f"{self.capture_uuid}:process-group:{self._process_group_ordinal}"
            )
            self._process_group_ordinal += 1
            self._process_group_ids[object_identity] = group_id
        key = (group_id, "*")
        ordinal = self._collective_ordinals.get(key, 0)
        self._collective_ordinals[key] = ordinal + 1
        ranks = None
        if group is not None and hasattr(torch.distributed, "get_process_group_ranks"):
            try:
                ranks = list(torch.distributed.get_process_group_ranks(group))
            except Exception:  # noqa: BLE001 - optional distributed metadata
                ranks = None
        if registered_name is None:
            try:
                from torch.distributed.distributed_c10d import _get_process_group_name

                registered_name = _get_process_group_name(group)
            except Exception:  # noqa: BLE001 - private API varies
                registered_name = None
        aliases = [registered_name] if registered_name not in {None, "None"} else []
        try:
            from torch.distributed.distributed_c10d import _world

            aliases.extend(
                name for pg, name in _world.pg_names.items() if pg is group
            )
        except Exception:  # noqa: BLE001,S110 - optional private aliases
            pass
        aliases = sorted(set(aliases))
        group_record = {
            "group_id": group_id,
            "registered_names": aliases,
            "ranks": ranks,
            "backend": str(group._get_backend_name()),
            "world_rank": torch.distributed.get_rank(),
            "local_rank": group.rank(),
            "size": group.size(),
            "creation_ordinal": int(group_id.rsplit(":", 1)[-1]),
            "source_object_id": object_identity,
            "is_world": group is torch.distributed.group.WORLD,
        }
        self._process_group_records[group_id] = group_record
        async_requested = any(
            frame.f_locals.get("async_op") is True
            for frame in self._thread.frame_stack
        )
        if async_requested:
            self.loss(
                "escaping_collective_work",
                "async collective Work lifecycle is not replayable",
                severity="fatal",
                replayable=False,
                context={"group_id": group_id, "op": overload},
            )
        tensor_contract = [
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "device": str(tensor.device),
            }
            for tensor in self._tensors((args, kwargs))
        ]
        return {
            "group_id": group_id,
            "local_ordinal": ordinal,
            "op": overload,
            "group": group_record,
            "ranks": ranks,
            "resolved": True,
            "async_requested": async_requested,
            "tensor_contract": tensor_contract,
            "split_sizes": [
                value
                for value in leaves
                if isinstance(value, list)
                and all(isinstance(item, int) for item in value)
            ],
            "is_wait": False,
        }

    def _record_graph_candidate(self, invocation: _Invocation) -> str | None:
        if not self.policy.graph_candidates or not invocation.graph_records:
            return None
        key = canonical_json(
            {
                "signature_id": invocation.signature_id,
                "operations": invocation.graph_records,
            }
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        graph_id = f"sha256:{digest}"
        if graph_id not in self._graphs:
            self._graphs[graph_id] = invocation.invocation_id
            self.emit(
                "graph_definition",
                graph_id=graph_id,
                signature_id=invocation.signature_id,
                source_invocation_id=invocation.invocation_id,
                operations=invocation.graph_records,
            )
        else:
            self.emit(
                "graph_reference",
                graph_id=graph_id,
                invocation_id=invocation.invocation_id,
            )
        return graph_id

    def _snapshot_value(
        self, value: Any, purpose: str, *, required: bool
    ) -> dict[str, Any] | None:
        assert self._snapshots is not None
        try:
            reference = self._snapshots.save(value, purpose)
        except SnapshotQuotaExceeded as exc:
            self.loss(
                "snapshot_quota_exhausted",
                str(exc),
                severity="error" if required else "warning",
                replayable=False,
                context={"purpose": purpose, "required": required},
            )
            return None
        except BaseException as exc:  # noqa: BLE001 - serialization is untrusted
            self.loss(
                "snapshot_serialization_failed",
                f"{type(exc).__name__}: {exc}",
                severity="error" if required else "warning",
                replayable=False,
                context={"purpose": purpose, "required": required},
            )
            return None
        self.emit("snapshot", purpose=purpose, **reference)
        return reference

    def _value_unsupported(self, kind: str, context: dict[str, Any]) -> None:
        self.loss(
            kind,
            "value could not be represented exactly in JSON metadata",
            severity="warning",
            replayable=False,
            context=context,
        )

    def loss(
        self,
        kind: str,
        message: str,
        *,
        severity: str,
        replayable: bool,
        context: dict[str, Any] | None = None,
    ) -> None:
        loss = self.completeness.add(
            kind=kind,
            message=message,
            severity=severity,
            replayable=replayable,
            context=context,
        )
        event_payload = dict(loss)
        event_payload["unsupported_kind"] = event_payload.pop("kind")
        self.emit("unsupported", **event_payload)

    def emit(self, kind: str, *, force: bool = False, **payload: Any) -> None:
        if (
            self._writer is None
            or self._writer_failed
            or (self._closing and not force)
        ):
            return
        with self._emit_lock:
            sequence = self._next_sequence()
            try:
                with self.suppress():
                    self._writer.write(event(kind, sequence, self.rank, **payload))
            except BaseException as exc:  # noqa: BLE001 - writer failure contained
                self._writer_failed = True
                self.completeness.add(
                    kind="event_write_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    severity="fatal",
                    replayable=False,
                )

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            value = self._global_sequence
            self._global_sequence += 1
            return value

    def _next_thread_sequence(self) -> int:
        value = self._thread.sequence
        self._thread.sequence += 1
        return value

    def _thread_id(self, thread: threading.Thread | None = None) -> str:
        thread = thread or threading.current_thread()
        identity = self.identities.identify(thread, "thread")
        with self._sequence_lock:
            if identity not in self._thread_ids:
                self._thread_ids[identity] = (
                    f"r{self.rank}:thread:{self._thread_ordinal}"
                )
                self._thread_ordinal += 1
            return self._thread_ids[identity]

    def _current_scope_id(self) -> str | None:
        frame = sys._getframe(1)
        while frame is not None:
            with self._scope_lock:
                scope = self._active_scopes.get(id(frame))
            if scope is not None and scope.frame is frame:
                return scope.scope_id
            frame = frame.f_back
        return None

    @staticmethod
    def _value_signature(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {
                "kind": "tensor",
                "type": qualified_name(type(value)),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "shape": list(value.shape),
                "stride": list(value.stride()),
            }
        if isinstance(value, dict):
            return {
                "kind": "mapping",
                "items": [
                    [
                        ExecutionCaptureSession._value_signature(key),
                        ExecutionCaptureSession._value_signature(item),
                    ]
                    for key, item in value.items()
                ],
            }
        if isinstance(value, (tuple, list)):
            return {
                "kind": type(value).__name__,
                "items": [
                    ExecutionCaptureSession._value_signature(item) for item in value
                ],
            }
        return {"kind": qualified_name(type(value))}

    @staticmethod
    def _signature_id(callable_id: str, signature: Any) -> str:
        digest = hashlib.sha256(canonical_json(signature).encode()).hexdigest()
        return f"{callable_id}:signature:{digest[:20]}"

    def _close_remaining_scopes(self) -> None:
        with self._scope_lock:
            remaining_scopes = list(self._active_scopes.values())
            self._active_scopes.clear()
        for scope in remaining_scopes:
            self.emit(
                "scope_end",
                force=True,
                scope_id=scope.scope_id,
                status="session_boundary",
                exception=scope.exception,
                result=None,
                thread_id=None,
            )

    def _finalize(self, *, success: bool) -> None:
        if self._finalized:
            return
        self._finalized = True
        with self._worker_lock:
            active_worker_count = sum(
                worker.is_alive() for worker in self._owned_workers
            )
        status, replayable = manifest_status(
            execution_succeeded=success,
            completeness=self.completeness,
            writer_failed=self._writer_failed,
            active_workers=active_worker_count,
        )
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "world_size": self.world_size,
            "capture_uuid": self.capture_uuid,
            "configuration_id": self.configuration_id,
            "status": status,
            "replayable": replayable,
            "event_count": self._global_sequence,
            "thread_count": len(self._thread_ids),
            "module_count": len(self._module_defs),
            "invocation_count": self._invocation_ordinal,
            "operation_count": self._operation_ordinal,
            "scope_count": self._scope_ordinal,
            "graph_count": len(self._graphs),
            "process_groups": sorted(
                self._process_group_records.values(),
                key=lambda item: item["creation_ordinal"],
            ),
            "policy": self.policy.as_dict(),
            "metadata": self.metadata,
            "losses": self.completeness.losses,
            "snapshots": (
                sorted(
                    self._snapshots.entries.values(),
                    key=lambda item: item["snapshot_id"],
                )
                if self._snapshots is not None
                else []
            ),
        }
        if self._writer is not None:
            if self._writer_failed:
                self._writer.abort()
            else:
                self._writer.finalize()
        events_path = self.root / "events.jsonl"
        if events_path.exists():
            events_digest = hashlib.sha256(events_path.read_bytes()).hexdigest()
            manifest["events"] = {
                "file": "events.jsonl",
                "sha256": events_digest,
                "bytes": events_path.stat().st_size,
            }
        atomic_write_json(self.root / "manifest.json", manifest)
        if manifest["status"] == "complete":
            complete_partial = self.root / "COMPLETE.partial"
            complete_partial.write_text(
                f"{SCHEMA_NAME} {SCHEMA_VERSION}\n", encoding="utf-8"
            )
            os.replace(complete_partial, self.root / "COMPLETE")
        self._state = "closed"
        if (
            self.policy.strict
            and manifest["status"] != "complete"
            and self._model_exception is None
        ):
            raise CaptureFinalizationError(
                f"execution capture is incomplete; see {self.root / 'manifest.json'}"
            )
