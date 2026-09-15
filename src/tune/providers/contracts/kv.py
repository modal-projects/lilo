from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InsertResult:
    created: bool
    value: object


class KeyValueStore(Protocol):
    async def get(self, key: str) -> object | None: ...

    async def put(self, key: str, value: object) -> None: ...

    async def put_if_absent(self, key: str, value: object) -> InsertResult: ...

    async def delete(self, key: str) -> None: ...

    async def list_keys(self, prefix: str) -> tuple[str, ...]: ...

    async def list_items(
        self,
        *prefixes: str,
    ) -> tuple[tuple[str, object], ...]: ...


class SessionKeyValueStores(Protocol):
    def for_session(self, session_id: str) -> KeyValueStore: ...

    async def delete_session(self, session_id: str) -> None: ...
