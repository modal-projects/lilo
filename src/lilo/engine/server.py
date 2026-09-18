from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from tinker.types.forward_backward_input import ForwardBackwardInput

from lilo.encoding import fingerprint
from lilo.errors import EngineSaturated, RecordNotFound, SequenceConflict

from .api import Command, Executor, FutureState, FutureStatus, OperationKind
from .ingress import decode_forward_backward, decode_json_operation
from .operations import (
    LoadCheckpointPayload,
    OperationPayload,
    SkipPayload,
    serialize_operation_payload,
)

PERSISTED_OPERATIONS = {
    OperationKind.SAVE_WEIGHTS,
    OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
}
PERSIST_LANES = {
    OperationKind.SAVE_WEIGHTS: "checkpoint",
    OperationKind.SAVE_WEIGHTS_FOR_SAMPLER: "sampler",
}


class Observer(Protocol):
    def register_model(self, model_id: str, spec: object) -> None: ...
    def forget_model(self, model_id: str) -> None: ...
    def begin(self, operation: Operation) -> None: ...
    def reuse(self, request_id: str) -> None: ...
    def finish(self, operation: Operation, state: FutureState) -> None: ...
    def set_activity(self, lane: str, operation: str) -> None: ...

    def span(
        self,
        model: str | tuple[str, ...] | list[str],
        name: str,
        lane: str,
        t0: float,
        t1: float | None = None,
        **attrs,
    ) -> None: ...

    def state(
        self,
        model: str | tuple[str, ...] | list[str],
        state: str,
        **detail,
    ) -> None: ...


def _failure(exc: BaseException, what: str) -> str:
    logging.getLogger(__name__).exception("engine %s failed", what)
    return f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class Operation:
    request_id: str
    model_id: str
    seq_id: int
    kind: OperationKind
    payload: OperationPayload


@dataclass(frozen=True)
class _PersistJob:
    operation: Operation
    capture: object


@dataclass(frozen=True)
class _AcceptOperation:
    model_id: str
    spec: object
    done: asyncio.Future[bool]


@dataclass(frozen=True)
class _UnloadOperation:
    model_id: str
    done: asyncio.Future[None]


@dataclass
class _ModelState:
    spec: object
    next_seq: int = 1
    buffered: dict[int, Operation] = field(default_factory=dict)
    fingerprints: dict[int, str] = field(default_factory=dict)
    done: deque[int] = field(default_factory=deque)
    retrieved: set[int] = field(default_factory=set)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    registration: asyncio.Future[bool] | None = None
    unload: asyncio.Future[None] | None = None


class Engine:
    """Schedule model operations and track their asynchronous results."""

    def __init__(
        self,
        executor: Executor,
        *,
        max_models: int = 8,
        max_buffered: int = 256,
        max_results: int = 128,
        observer: Observer | None = None,
        sampler_persistence_concurrency: int = 1,
    ) -> None:
        if sampler_persistence_concurrency < 1:
            raise ValueError("sampler_persistence_concurrency must be positive")
        self.executor = executor
        self.observer = observer
        self._persistence_active: dict[str, int] = {}
        self.max_models = max_models
        self.max_buffered = max_buffered
        self.max_results = max_results
        self.sampler_persistence_concurrency = sampler_persistence_concurrency
        self._sampler_inflight: set[str] = set()
        self._serial_persistence_inflight: set[OperationKind] = set()
        self.draining = False
        self._models: dict[str, _ModelState] = {}
        self._futures: dict[str, FutureState] = {}
        self._lock = asyncio.Lock()
        self._work = asyncio.Condition(self._lock)
        self._completed = asyncio.Condition(self._lock)
        self._lifecycle: deque[_AcceptOperation | _UnloadOperation] = deque()
        self._checkpoint_persistence: asyncio.Queue[_PersistJob] = asyncio.Queue()
        self._sampler_persistence: asyncio.Queue[_PersistJob] = asyncio.Queue()
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._closing = False

    async def accept_model(self, model_id: str, spec: object) -> bool:
        async with self._lock:
            model = self._models.get(model_id)
            registering = model is None
            if registering:
                if self.draining or len(self._models) >= self.max_models:
                    return False
                if self.observer is not None:
                    self.observer.register_model(model_id, spec)
                registration = asyncio.get_running_loop().create_future()
                model = self._models[model_id] = _ModelState(
                    spec=spec,
                    registration=registration,
                )
                self._lifecycle.append(_AcceptOperation(model_id, spec, registration))
                self._start_tasks()
                self._work.notify_all()
            elif model.unload is not None:
                return False
            else:
                registration = model.registration
        return await asyncio.shield(registration)

    async def model_ids(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(self._models)

    @property
    def loaded(self) -> tuple[str, ...]:
        return tuple(self._models)

    async def forward_backward(self, body: bytes, content_type: str) -> str:
        model_id, seq_id, kind, payload = decode_forward_backward(body, content_type)
        return await self._submit(kind, model_id, seq_id, payload)

    async def forward(self, request: dict) -> str:
        model_id, seq_id, payload = decode_json_operation(
            OperationKind.FORWARD,
            request,
        )
        return await self._submit(OperationKind.FORWARD, model_id, seq_id, payload)

    async def optim_step(self, request: dict) -> str:
        model_id, seq_id, payload = decode_json_operation(
            OperationKind.OPTIM_STEP,
            request,
        )
        return await self._submit(OperationKind.OPTIM_STEP, model_id, seq_id, payload)

    async def save_weights(self, request: dict) -> str:
        model_id, seq_id, payload = decode_json_operation(
            OperationKind.SAVE_WEIGHTS,
            request,
        )
        return await self._submit(OperationKind.SAVE_WEIGHTS, model_id, seq_id, payload)

    async def load_weights(self, request: dict) -> str:
        model_id, seq_id, payload = decode_json_operation(
            OperationKind.LOAD_WEIGHTS,
            request,
        )
        return await self._submit(OperationKind.LOAD_WEIGHTS, model_id, seq_id, payload)

    async def save_weights_for_sampler(self, request: dict) -> str:
        model_id, seq_id, payload = decode_json_operation(
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
            request,
        )
        return await self._submit(
            OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
            model_id,
            seq_id,
            payload,
        )

    async def skip_sequence(self, model_id: str, seq_id: int, error: str) -> str:
        return await self._submit(
            OperationKind.SKIP,
            model_id,
            seq_id,
            SkipPayload(error=error),
        )

    async def retrieve_future(
        self,
        request_id: str,
        timeout: float = 0.0,
    ) -> FutureState | None:
        deadline = asyncio.get_running_loop().time() + timeout
        async with self._lock:
            while True:
                state = self._futures.get(request_id)
                remaining = deadline - asyncio.get_running_loop().time()
                if (
                    state is None
                    or state.status != FutureStatus.PENDING
                    or remaining <= 0
                ):
                    self._mark_retrieved(request_id, state)
                    return state
                try:
                    await asyncio.wait_for(self._completed.wait(), remaining)
                except TimeoutError:
                    state = self._futures.get(request_id)
                    self._mark_retrieved(request_id, state)
                    return state

    def _mark_retrieved(self, request_id: str, state: FutureState | None) -> None:
        if state is None or state.status == FutureStatus.PENDING:
            return
        model_id, _, seq = request_id.rpartition(":")
        model = self._models.get(model_id)
        if model is None:
            return
        try:
            model.retrieved.add(int(seq))
        except ValueError:
            return

    async def shutdown_if_idle(self) -> bool:
        async with self._lock:
            if self._models:
                return False
            self.draining = True
            return True

    async def unload_model(self, model_id: str) -> None:
        while True:
            async with self._lock:
                if self._closing:
                    return
                model = self._models.get(model_id)
                if model is None:
                    return
                if not model.ready.is_set():
                    ready = model.ready
                    done = None
                else:
                    ready = None
                    done = model.unload
                    if done is None:
                        done = asyncio.get_running_loop().create_future()
                        model.unload = done
                        if self.observer is not None:
                            for pending in model.buffered.values():
                                self.observer.finish(
                                    pending, FutureState(FutureStatus.FAILED)
                                )
                        model.buffered.clear()
                        for seq_id in model.fingerprints:
                            self._futures.pop(f"{model_id}:{seq_id}", None)
                        model.retrieved.clear()
                        self._lifecycle.append(_UnloadOperation(model_id, done))
                        self._start_tasks()
                        self._work.notify_all()
                        self._completed.notify_all()
            if ready is None:
                await asyncio.shield(done)
                return
            await ready.wait()

    async def close(self) -> None:
        async with self._lock:
            if self._closing:
                return
            self._closing = True
            self.draining = True
            tasks, self._tasks = self._tasks, ()
            self._work.notify_all()
        if tasks:
            command_task, *persistence_tasks = tasks
            await asyncio.gather(command_task, return_exceptions=True)
            await self._join_persistence()
            for task in persistence_tasks:
                task.cancel()
            await asyncio.gather(
                *persistence_tasks,
                return_exceptions=True,
            )
        for model in self._models.values():
            if model.registration is not None and not model.registration.done():
                model.registration.cancel()
            if model.unload is not None and not model.unload.done():
                model.unload.cancel()

    async def _submit(
        self,
        kind: OperationKind,
        model_id: str,
        seq_id: int,
        payload: OperationPayload,
    ) -> str:
        operation = Operation(
            request_id=f"{model_id}:{seq_id}",
            model_id=model_id,
            seq_id=seq_id,
            kind=kind,
            payload=payload,
        )
        async with self._lock:
            model = self._models.get(operation.model_id)
            if model is None or model.unload is not None:
                raise RecordNotFound("model", operation.model_id)
            mark = fingerprint(
                operation.kind.value,
                serialize_operation_payload(operation.payload),
            )
            seen = model.fingerprints.get(operation.seq_id)
            if seen is not None:
                if seen != mark:
                    raise SequenceConflict(operation.model_id, operation.seq_id)
                if self.observer is not None:
                    self.observer.reuse(operation.request_id)
                return operation.request_id
            if operation.seq_id < model.next_seq:
                raise SequenceConflict(operation.model_id, operation.seq_id)
            if self.draining:
                raise EngineSaturated("draining")
            if len(model.buffered) >= self.max_buffered:
                raise EngineSaturated("operation buffer full")
            if self.observer is not None:
                self.observer.begin(operation)
            model.fingerprints[operation.seq_id] = mark
            model.buffered[operation.seq_id] = operation
            self._futures[operation.request_id] = FutureState(FutureStatus.PENDING)
            self._start_tasks()
            self._work.notify_all()
        return operation.request_id

    def _start_tasks(self) -> None:
        if self._tasks or self._closing:
            return
        self._tasks = (
            asyncio.create_task(self._run_loop()),
            asyncio.create_task(self._persistence_loop(self._checkpoint_persistence)),
            *(
                asyncio.create_task(self._persistence_loop(self._sampler_persistence))
                for _ in range(self.sampler_persistence_concurrency)
            ),
        )

    def _observe_state(self, models: tuple[str, ...] | list[str], state: str) -> None:
        if self.observer is not None:
            self.observer.state(tuple(models), state)

    def _observe_span(
        self,
        models: tuple[str, ...] | list[str],
        name: str,
        lane: str,
        started: float,
        **attrs,
    ) -> None:
        if self.observer is not None and models:
            self.observer.span(tuple(models), name, lane, started, time.time(), **attrs)

    async def _run_loop(self) -> None:
        busy = False
        while True:
            async with self._lock:
                if busy and (self._closing or self._ready_batch() is None):
                    self._observe_state(tuple(self._models), "idle")
                    busy = False
                while not self._closing and (operations := self._ready_batch()) is None:
                    await self._work.wait()
                if self._closing:
                    return
                self._consume_ready(operations)
            busy = True
            operation = operations[0]
            if isinstance(operation, _AcceptOperation):
                await self._run_accept(operation)
                continue
            if isinstance(operation, _UnloadOperation):
                await self._run_unload(operation)
                continue
            if operation.kind == OperationKind.SKIP:
                assert isinstance(operation.payload, SkipPayload)
                async with self._lock:
                    self._finish(
                        operation,
                        FutureState(FutureStatus.FAILED, error=operation.payload.error),
                    )
                continue
            if operation.kind in PERSISTED_OPERATIONS:
                await self._capture_for_persistence(operation)
                continue
            models = tuple(dict.fromkeys(item.model_id for item in operations))
            self._observe_state(models, f"executing:{operation.kind.value}")
            if operation.kind == OperationKind.LOAD_WEIGHTS:
                await self._join_persistence()
            started = time.time()
            try:
                if operation.kind == OperationKind.FORWARD_BACKWARD:
                    results = await self.executor.execute_forward_backward_batch(
                        tuple(
                            Command(item.model_id, item.kind, item.payload)
                            for item in operations
                        )
                    )
                    if len(results) != len(operations):
                        raise RuntimeError("executor returned the wrong result count")
                else:
                    results = (
                        await self.executor.execute(
                            operation.model_id,
                            operation.kind,
                            operation.payload,
                        ),
                    )
                states = tuple(
                    FutureState(FutureStatus.COMPLETE, result=result)
                    for result in results
                )
            except Exception as exc:  # noqa: BLE001
                error = _failure(exc, f"{operation.kind.value} x{len(operations)}")
                states = tuple(
                    FutureState(FutureStatus.FAILED, error=error) for _ in operations
                )
            self._observe_span(
                models,
                operation.kind.value,
                "gpu",
                started,
                seq_ids=[item.seq_id for item in operations],
                n=len(operations),
                ok=all(state.status == FutureStatus.COMPLETE for state in states),
            )
            async with self._lock:
                for item, state in zip(operations, states, strict=True):
                    self._finish(item, state)

    async def _run_accept(self, operation: _AcceptOperation) -> None:
        error = None
        started = time.time()
        self._observe_state((operation.model_id,), "executing:accept")
        try:
            await self._join_persistence()
            await self.executor.accept_model(operation.model_id, operation.spec)
            if isinstance(operation.spec, Mapping):
                checkpoint = operation.spec.get("checkpoint")
                if isinstance(checkpoint, Mapping):
                    await self.executor.execute(
                        operation.model_id,
                        OperationKind.LOAD_WEIGHTS,
                        LoadCheckpointPayload.model_validate(checkpoint),
                    )
        except Exception as exc:
            error = ValueError(f"accept model: {exc}")
            logging.getLogger(__name__).exception("executor accept_model")
            try:
                await self.executor.unload_model(operation.model_id)
            except Exception:
                logging.getLogger(__name__).exception("executor unload_model")
        self._observe_span(
            (operation.model_id,),
            "accept",
            "gpu",
            started,
            ok=error is None,
        )
        async with self._lock:
            model = self._models.get(operation.model_id)
            if model is not None and model.registration is operation.done:
                if error is not None:
                    self._models.pop(operation.model_id)
                model.ready.set()
        if not operation.done.done():
            if error is None:
                operation.done.set_result(True)
            else:
                operation.done.set_exception(error)

    async def _run_unload(self, operation: _UnloadOperation) -> None:
        error = None
        started = time.time()
        self._observe_state((operation.model_id,), "executing:unload")
        try:
            await self._join_persistence()
            await self.executor.unload_model(operation.model_id)
        except Exception as exc:  # noqa: BLE001
            error = exc
        self._observe_span(
            (operation.model_id,), "unload", "gpu", started, ok=error is None
        )
        if self.observer is not None:
            self.observer.forget_model(operation.model_id)
        async with self._lock:
            model = self._models.get(operation.model_id)
            if model is not None and model.unload is operation.done:
                self._models.pop(operation.model_id)
        if operation.done.done():
            return
        if error is None:
            operation.done.set_result(None)
        else:
            operation.done.set_exception(error)

    async def _capture_for_persistence(self, operation: Operation) -> None:
        queue = (
            self._checkpoint_persistence
            if operation.kind == OperationKind.SAVE_WEIGHTS
            else self._sampler_persistence
        )
        parallel_sampler = (
            queue is self._sampler_persistence
            and self.sampler_persistence_concurrency > 1
        )
        if parallel_sampler:
            # Reservation covers capture and persistence; eligibility is checked
            # before consuming the operation so backpressure cannot stall dispatch.
            self._sampler_inflight.add(operation.model_id)
        else:
            # Reserve before capture, but defer busy lanes in _ready_batch.
            # Waiting here would stop dispatch for every other client.
            self._serial_persistence_inflight.add(operation.kind)
        name = f"capture:{operation.kind.value}"
        self._observe_state((operation.model_id,), f"executing:{name}")
        started = time.time()
        try:
            capture = await self.executor.capture_snapshot(
                operation.model_id,
                operation.kind,
                operation.payload,
            )
        except Exception as exc:  # noqa: BLE001
            self._observe_span(
                (operation.model_id,),
                name,
                "gpu",
                started,
                seq_ids=[operation.seq_id],
                ok=False,
            )
            async with self._lock:
                if parallel_sampler:
                    self._sampler_inflight.remove(operation.model_id)
                else:
                    self._serial_persistence_inflight.remove(operation.kind)
                self._work.notify_all()
                self._finish(
                    operation,
                    FutureState(FutureStatus.FAILED, error=_failure(exc, name)),
                )
            return
        self._observe_span(
            (operation.model_id,),
            name,
            "gpu",
            started,
            seq_ids=[operation.seq_id],
            ok=True,
        )
        await queue.put(_PersistJob(operation, capture))

    async def _persistence_loop(self, queue: asyncio.Queue[_PersistJob]) -> None:
        while True:
            job = await queue.get()
            started = time.time()
            lane = PERSIST_LANES[job.operation.kind]
            self._persistence_active[lane] = self._persistence_active.get(lane, 0) + 1
            if self.observer is not None and self._persistence_active[lane] == 1:
                self.observer.set_activity(lane, job.operation.kind.value)
            try:
                try:
                    result = await self.executor.persist_snapshot(
                        job.operation.model_id,
                        job.operation.kind,
                        job.operation.payload,
                        job.capture,
                    )
                    state = FutureState(FutureStatus.COMPLETE, result=result)
                except Exception as exc:  # noqa: BLE001
                    state = FutureState(
                        FutureStatus.FAILED,
                        error=_failure(exc, f"persist:{job.operation.kind.value}"),
                    )
                self._observe_span(
                    (job.operation.model_id,),
                    f"persist:{job.operation.kind.value}",
                    PERSIST_LANES[job.operation.kind],
                    started,
                    seq_ids=[job.operation.seq_id],
                    ok=state.status == FutureStatus.COMPLETE,
                )
                async with self._lock:
                    self._finish(job.operation, state)
                    if (
                        queue is self._sampler_persistence
                        and self.sampler_persistence_concurrency > 1
                    ):
                        self._sampler_inflight.remove(job.operation.model_id)
                    else:
                        self._serial_persistence_inflight.remove(job.operation.kind)
                    self._work.notify_all()
            finally:
                self._persistence_active[lane] -= 1
                if self.observer is not None and self._persistence_active[lane] == 0:
                    self.observer.set_activity(lane, "idle")
                queue.task_done()

    async def _join_persistence(self) -> None:
        await asyncio.gather(
            self._checkpoint_persistence.join(),
            self._sampler_persistence.join(),
        )

    def _finish(self, operation: Operation, state: FutureState) -> None:
        if self.observer is not None:
            self.observer.finish(operation, state)
        model = self._models.get(operation.model_id)
        if model is not None and model.unload is None:
            self._futures[operation.request_id] = state
            model.done.append(operation.seq_id)
            while len(model.done) > self.max_results:
                oldest = model.done[0]
                if oldest not in model.retrieved:
                    break
                model.done.popleft()
                model.retrieved.discard(oldest)
                self._futures.pop(f"{operation.model_id}:{oldest}", None)
                model.fingerprints.pop(oldest, None)
        self._completed.notify_all()

    def _ready_lifecycle(
        self,
    ) -> Operation | _AcceptOperation | _UnloadOperation | None:
        if self._lifecycle:
            return self._lifecycle[0]

    def _ready_batch(
        self,
    ) -> tuple[Operation | _AcceptOperation | _UnloadOperation, ...] | None:
        lifecycle = self._ready_lifecycle()
        if lifecycle is not None:
            return (lifecycle,)
        ready = []
        for model in self._models.values():
            if not model.ready.is_set() or model.unload is not None:
                continue
            operation = model.buffered.get(model.next_seq)
            if operation is not None:
                if operation.kind in self._serial_persistence_inflight:
                    continue
                if (
                    operation.kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER
                    and self.sampler_persistence_concurrency > 1
                    and (
                        operation.model_id in self._sampler_inflight
                        or len(self._sampler_inflight)
                        >= self.sampler_persistence_concurrency
                    )
                ):
                    # Serialize versions of one adapter while letting other
                    # adapters train or publish using independent snapshots.
                    continue
                ready.append(operation)
        if not ready:
            return None

        selected = [ready[0]]
        if ready[0].kind == OperationKind.FORWARD_BACKWARD:
            key = self._forward_backward_batch_key(ready[0])
            selected.extend(
                operation
                for operation in ready[1:]
                if operation.kind == OperationKind.FORWARD_BACKWARD
                and self._forward_backward_batch_key(operation) == key
            )
            for operation in tuple(selected):
                buffered = self._models[operation.model_id].buffered
                seq_id = operation.seq_id + 1
                while (
                    (queued := buffered.get(seq_id)) is not None
                    and queued.kind == OperationKind.FORWARD_BACKWARD
                    and self._forward_backward_batch_key(queued) == key
                ):
                    selected.append(queued)
                    seq_id += 1
        return tuple(selected)

    def _consume_ready(
        self,
        operations: tuple[Operation | _AcceptOperation | _UnloadOperation, ...],
    ) -> None:
        first = operations[0]
        if isinstance(first, _AcceptOperation | _UnloadOperation):
            queued = self._lifecycle.popleft()
            if queued is not first:
                raise RuntimeError("lifecycle operation changed before execution")
            return
        for operation in operations:
            if not isinstance(operation, Operation):
                raise RuntimeError("operation batch contains lifecycle work")
            model = self._models[operation.model_id]
            queued = model.buffered.pop(model.next_seq)
            if queued is not operation:
                raise RuntimeError("model operation changed before execution")
            model.next_seq += 1

    @staticmethod
    def _forward_backward_batch_key(operation: Operation) -> str:
        payload = operation.payload
        if not isinstance(payload, ForwardBackwardInput):
            raise ValueError("forward_backward requires a forward payload")
        return fingerprint(
            operation.kind.value,
            {
                "loss_fn": payload.loss_fn,
                "loss_fn_config": payload.loss_fn_config,
            },
        )
