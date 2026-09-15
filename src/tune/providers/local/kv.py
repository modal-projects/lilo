from __future__ import annotations

import asyncio
import copy

from ..contracts import InsertResult


class InMemoryKeyValueStore:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> object | None:
        async with self._lock:
            return copy.deepcopy(self._values.get(key))

    async def put(self, key: str, value: object) -> None:
        async with self._lock:
            self._values[key] = copy.deepcopy(value)

    async def put_if_absent(self, key: str, value: object) -> InsertResult:
        async with self._lock:
            if key in self._values:
                return InsertResult(False, copy.deepcopy(self._values[key]))
            stored = copy.deepcopy(value)
            self._values[key] = stored
            return InsertResult(True, copy.deepcopy(stored))

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._values.pop(key, None)

    async def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(key for key, _ in await self.list_items(prefix))

    async def list_items(
        self,
        *prefixes: str,
    ) -> tuple[tuple[str, object], ...]:
        async with self._lock:
            return tuple(
                (key, copy.deepcopy(value))
                for key, value in sorted(self._values.items())
                if any(key.startswith(prefix) for prefix in prefixes)
            )


class InMemorySessionKeyValueStores:
    def __init__(self) -> None:
        self.stores: dict[str, InMemoryKeyValueStore] = {}

    def for_session(self, session_id: str) -> InMemoryKeyValueStore:
        return self.stores.setdefault(session_id, InMemoryKeyValueStore())

    async def delete_session(self, session_id: str) -> None:
        self.stores.pop(session_id, None)
