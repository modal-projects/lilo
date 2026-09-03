from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable

import modal
from grpclib.exceptions import StreamTerminatedError

from ..contracts import InsertResult

SESSION_KV_NAME = "lilo-sessions"
MODEL_KV_NAME = "lilo-models"
ENGINE_KV_NAME = "lilo-engines"
SAMPLING_SESSION_KV_NAME = "lilo-sampling-sessions"
EXPORT_KV_NAME = "lilo-exports"
ARTIFACT_KV_NAME = "lilo-artifacts"
FFT_POOL_KV_NAME = "lilo-fft-pools"
LIST_ITEMS_ATTEMPTS = 3
LIST_ITEMS_TIMEOUT_SECONDS = 30.0
DeleteDict = Callable[..., Awaitable[None]]

STORE_NAMES = {
    "sessions": SESSION_KV_NAME,
    "models": MODEL_KV_NAME,
    "engines": ENGINE_KV_NAME,
    "sampling_sessions": SAMPLING_SESSION_KV_NAME,
    "exports": EXPORT_KV_NAME,
    "artifacts": ARTIFACT_KV_NAME,
}

KEY_STORES = {
    "session": "sessions",
    "session_last_seen": "sessions",
    "session_closed": "sessions",
    "model_creation": "models",
    "model": "models",
    "placement": "models",
    "placement_claim": "models",
    "trainer_demand": "models",
    "engine_instance": "engines",
    "engine_call": "engines",
    "engine_current": "engines",
    "trainer_plan": "engines",
    "trainer_reconcile": "engines",
    "trainer_reconcile_request": "engines",
    "trainer_reconcile_complete": "engines",
    "sampling_session_creation": "sampling_sessions",
    "sampling_session": "sampling_sessions",
    "sampler_export_submission": "exports",
    "sampler_export_result": "exports",
    "sampler_artifact": "artifacts",
}


def current_app_id() -> str:
    app = modal.App._get_container_app()
    if app is None or app.app_id is None:
        raise RuntimeError("Modal App ID is unavailable")
    return app.app_id


def app_store_name(name: str, app_id: str) -> str:
    return f"{app_id}-{name.removeprefix('lilo-')}"


class ModalKeyValueStore:
    def __init__(self, dictionary: modal.Dict) -> None:
        self.dictionary = dictionary

    async def get(self, key: str) -> object | None:
        return await self.dictionary.get.aio(key)

    async def put(self, key: str, value: object) -> None:
        await self.dictionary.put.aio(key, value)

    async def put_if_absent(self, key: str, value: object) -> InsertResult:
        created = await self.dictionary.put.aio(key, value, skip_if_exists=True)
        if created:
            return InsertResult(True, value)
        return InsertResult(False, await self.dictionary.get.aio(key))

    async def delete(self, key: str) -> None:
        await self.dictionary.pop.aio(key, None)

    async def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(key for key, _ in await self.list_items(prefix))

    async def list_items(
        self,
        *prefixes: str,
    ) -> tuple[tuple[str, object], ...]:
        for attempt in range(LIST_ITEMS_ATTEMPTS):
            try:
                async with asyncio.timeout(LIST_ITEMS_TIMEOUT_SECONDS):
                    items = [
                        (key, value)
                        async for key, value in self.dictionary.items.aio()
                        if any(key.startswith(prefix) for prefix in prefixes)
                    ]
                return tuple(sorted(items))
            except (StreamTerminatedError, TimeoutError):
                if attempt == LIST_ITEMS_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(0.2 * 2**attempt)
        raise AssertionError("unreachable")


def session_task_store_name(session_id: str, app_id: str) -> str:
    if not session_id:
        raise ValueError("session_id must be non-empty")
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:24]
    return f"{app_id}-tasks-{digest}"


class ModalSessionKeyValueStores:
    def __init__(
        self,
        app_id: str | None = None,
        delete: DeleteDict | None = None,
    ) -> None:
        self.app_id = app_id or current_app_id()
        self.stores: dict[str, ModalKeyValueStore] = {}
        self.delete = delete or modal.Dict.objects.delete.aio

    def for_session(self, session_id: str) -> ModalKeyValueStore:
        store = self.stores.get(session_id)
        if store is None:
            store = ModalKeyValueStore(
                modal.Dict.from_name(
                    session_task_store_name(session_id, self.app_id),
                    create_if_missing=True,
                )
            )
            self.stores[session_id] = store
        return store

    async def delete_session(self, session_id: str) -> None:
        self.stores.pop(session_id, None)
        await self.delete(
            session_task_store_name(session_id, self.app_id),
            allow_missing=True,
        )


class RoutedKeyValueStore:
    def __init__(self, stores: dict[str, ModalKeyValueStore]) -> None:
        self.stores = stores

    def _store(self, key: str) -> ModalKeyValueStore:
        family = key.partition(":")[0]
        try:
            return self.stores[KEY_STORES[family]]
        except KeyError:
            raise ValueError(f"unknown key family: {family}") from None

    def _matching_stores(
        self, prefixes: tuple[str, ...]
    ) -> tuple[ModalKeyValueStore, ...]:
        domains = {
            domain
            for family, domain in KEY_STORES.items()
            if any(
                f"{family}:".startswith(prefix) or prefix.startswith(f"{family}:")
                for prefix in prefixes
            )
        }
        if not domains:
            raise ValueError(f"unknown key prefixes: {prefixes}")
        return tuple(self.stores[domain] for domain in sorted(domains))

    async def get(self, key: str) -> object | None:
        return await self._store(key).get(key)

    async def put(self, key: str, value: object) -> None:
        await self._store(key).put(key, value)

    async def put_if_absent(self, key: str, value: object) -> InsertResult:
        return await self._store(key).put_if_absent(key, value)

    async def delete(self, key: str) -> None:
        await self._store(key).delete(key)

    async def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(key for key, _ in await self.list_items(prefix))

    async def list_items(
        self,
        *prefixes: str,
    ) -> tuple[tuple[str, object], ...]:
        items = []
        for store in self._matching_stores(prefixes):
            items.extend(await store.list_items(*prefixes))
        return tuple(sorted(items))


def shared_kv(app_id: str | None = None) -> RoutedKeyValueStore:
    app_id = app_id or current_app_id()
    return RoutedKeyValueStore(
        {
            domain: ModalKeyValueStore(
                modal.Dict.from_name(
                    app_store_name(name, app_id),
                    create_if_missing=True,
                )
            )
            for domain, name in STORE_NAMES.items()
        }
    )


def fft_pool_kv(app_id: str | None = None) -> ModalKeyValueStore:
    app_id = app_id or current_app_id()
    return ModalKeyValueStore(
        modal.Dict.from_name(
            app_store_name(FFT_POOL_KV_NAME, app_id),
            create_if_missing=True,
        )
    )
