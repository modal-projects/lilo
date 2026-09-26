"""Concurrent immutable snapshot reads, exclusive mount refreshes."""

import asyncio
from contextlib import asynccontextmanager


class SnapshotAccess:
    def __init__(self, bulletin):
        self.bulletin = bulletin
        self._condition = asyncio.Condition()
        self._readers = 0
        self._refresh_task = None

    @asynccontextmanager
    async def read(self):
        async with self._condition:
            # A queued refresh blocks new readers so sustained traffic cannot
            # indefinitely postpone discovering a newly published version.
            await self._condition.wait_for(lambda: self._refresh_task is None)
            self._readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._readers -= 1
                self._condition.notify_all()

    async def refresh(self):
        # Share a refresh across models and version constraints, not just across
        # callers asking for the same adapter. No TTL: later calls still refresh.
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._refresh())
            self._refresh_task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        await asyncio.shield(self._refresh_task)

    async def _refresh(self):
        try:
            async with self._condition:
                await self._condition.wait_for(lambda: self._readers == 0)
            await self.bulletin.refresh()
        finally:
            async with self._condition:
                self._refresh_task = None
                self._condition.notify_all()

    async def close(self):
        if self._refresh_task is not None:
            # Don't cancel a to_thread volume reload: cancellation doesn't stop
            # the underlying reload, which must finish before shutdown.
            await asyncio.gather(self._refresh_task, return_exceptions=True)
