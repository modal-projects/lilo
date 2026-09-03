from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from ..contracts import (
    SamplingTask,
    SamplingTaskState,
    SamplingTaskStatus,
)

SamplingTaskRunner = Callable[[SamplingTask], Awaitable[object]]


class LocalSamplingTaskPlatform:
    def __init__(self, runner: SamplingTaskRunner) -> None:
        self.runner = runner
        self._ids: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task[object]] = {}
        self._lock = asyncio.Lock()

    async def submit(self, task: SamplingTask) -> str:
        async with self._lock:
            existing = self._ids.get(task.request_id)
            if existing is not None:
                return existing
            task_id = f"task-{uuid.uuid4().hex}"
            self._ids[task.request_id] = task_id
            self._tasks[task_id] = asyncio.create_task(self.runner(task))
            return task_id

    async def retrieve(
        self,
        task_id: str,
        timeout: float = 0.0,
    ) -> SamplingTaskState | None:
        task = self._tasks.get(task_id)
        if task is None or task.cancelled():
            return None
        if not task.done() and timeout > 0:
            await asyncio.wait((task,), timeout=timeout)
        if not task.done():
            return SamplingTaskState(SamplingTaskStatus.PENDING)
        try:
            result = task.result()
        except Exception as exc:  # noqa: BLE001
            return SamplingTaskState(SamplingTaskStatus.FAILED, error=str(exc))
        return SamplingTaskState(SamplingTaskStatus.COMPLETE, result=result)

    async def forget(self, task_id: str) -> None:
        task = self._tasks.pop(task_id, None)
        if task is not None and not task.done():
            task.cancel()
