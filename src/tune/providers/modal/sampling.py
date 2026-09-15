from __future__ import annotations

from collections.abc import Awaitable, Callable

import modal

from ..contracts import (
    SamplingTask,
    SamplingTaskState,
    SamplingTaskStatus,
    SessionKeyValueStores,
)

SpawnSamplingTask = Callable[[SamplingTask], Awaitable[str]]


def call_key(request_id: str) -> str:
    return f"sampling_call:{request_id}"


class ModalSamplingTaskPlatform:
    def __init__(
        self,
        stores: SessionKeyValueStores,
        spawn: SpawnSamplingTask,
    ) -> None:
        self.stores = stores
        self.spawn = spawn

    async def submit(self, task: SamplingTask) -> str:
        store = self.stores.for_session(task.session_id)
        key = call_key(task.request_id)
        existing = await store.get(key)
        if existing is not None:
            return str(existing)
        call_id = await self.spawn(task)
        inserted = await store.put_if_absent(key, call_id)
        if inserted.created:
            return call_id
        await modal.FunctionCall.from_id(call_id).cancel.aio()
        return str(inserted.value)

    async def retrieve(
        self,
        task_id: str,
        timeout: float = 0.0,
    ) -> SamplingTaskState | None:
        try:
            result = await modal.FunctionCall.from_id(task_id).get.aio(timeout=timeout)
        except TimeoutError:
            return SamplingTaskState(SamplingTaskStatus.PENDING)
        except (modal.exception.NotFoundError, modal.exception.OutputExpiredError):
            return None
        except (
            modal.exception.ConnectionError,
            modal.exception.InternalError,
            modal.exception.ResourceExhaustedError,
            modal.exception.ServiceError,
            OSError,
        ):
            return SamplingTaskState(SamplingTaskStatus.PENDING)
        except Exception as exc:  # noqa: BLE001
            return SamplingTaskState(SamplingTaskStatus.FAILED, error=str(exc))
        return SamplingTaskState(SamplingTaskStatus.COMPLETE, result=result)
