from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


@dataclass(frozen=True)
class SamplingTask:
    request_id: str
    session_id: str
    sampling_session_id: str
    base_model: str
    engine_definition_id: str | None
    model_path: str | None
    model_id: str | None
    publish_version: int | None
    payload: dict
    latest: bool = False


class SamplingTaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class SamplingTaskState:
    status: SamplingTaskStatus
    result: object | None = None
    error: str | None = None


class SamplingTaskPlatform(Protocol):
    async def submit(self, task: SamplingTask) -> str: ...

    async def retrieve(
        self,
        task_id: str,
        timeout: float = 0.0,
    ) -> SamplingTaskState | None: ...
