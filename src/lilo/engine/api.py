from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .operations import OperationPayload


class OperationKind(StrEnum):
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    SAVE_WEIGHTS = "save_weights"
    LOAD_WEIGHTS = "load_weights"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
    SKIP = "skip"


JSON_OPERATIONS = (
    OperationKind.FORWARD,
    OperationKind.OPTIM_STEP,
    OperationKind.SAVE_WEIGHTS,
    OperationKind.LOAD_WEIGHTS,
    OperationKind.SAVE_WEIGHTS_FOR_SAMPLER,
)


class FutureStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class FutureState:
    status: FutureStatus
    result: object | None = None
    error: str | None = None


@dataclass(frozen=True)
class Execution:
    model_id: str
    kind: OperationKind
    payload: OperationPayload


class EngineApi(Protocol):
    async def accept_model(self, model_id: str, spec: object) -> bool: ...

    async def model_ids(self) -> tuple[str, ...]: ...

    async def forward_backward(self, body: bytes, content_type: str) -> str: ...

    async def forward(self, request: dict) -> str: ...

    async def optim_step(self, request: dict) -> str: ...

    async def save_weights(self, request: dict) -> str: ...

    async def load_weights(self, request: dict) -> str: ...

    async def save_weights_for_sampler(self, request: dict) -> str: ...

    async def skip_sequence(self, model_id: str, seq_id: int, error: str) -> str: ...

    async def retrieve_future(
        self,
        request_id: str,
        timeout: float = 0.0,
    ) -> FutureState | None: ...

    async def unload_model(self, model_id: str) -> None: ...

    async def shutdown_if_idle(self) -> bool: ...


async def submit_json_operation(
    engine: EngineApi,
    kind: OperationKind,
    request: dict,
) -> str:
    match kind:
        case OperationKind.FORWARD:
            return await engine.forward(request)
        case OperationKind.OPTIM_STEP:
            return await engine.optim_step(request)
        case OperationKind.SAVE_WEIGHTS:
            return await engine.save_weights(request)
        case OperationKind.LOAD_WEIGHTS:
            return await engine.load_weights(request)
        case OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
            return await engine.save_weights_for_sampler(request)
        case _:
            raise ValueError(f"{kind.value} is not a JSON operation")


class Executor(Protocol):
    async def accept_model(self, model_id: str, spec: object) -> None: ...

    async def execute(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object: ...

    async def execute_batch(
        self,
        executions: tuple[Execution, ...],
    ) -> tuple[object, ...]: ...

    async def capture_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
    ) -> object: ...

    async def persist_operation(
        self,
        model_id: str,
        kind: OperationKind,
        payload: OperationPayload,
        capture: object,
    ) -> object: ...

    async def unload_model(self, model_id: str) -> None: ...

    async def close(self) -> None: ...
