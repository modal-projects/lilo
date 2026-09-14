import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from lilo.providers.modal import kv


class RetryableStreamError(Exception):
    pass


class StreamingItems:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts = 0

    def aio(self):
        async def stream():
            self.attempts += 1
            if self.attempts <= self.failures:
                raise RetryableStreamError
            yield "placement:model-a", {"instance": "engine-a"}
            yield "model:model-a", {"state": "ready"}

        return stream()


class PartiallyTerminatedItems:
    def __init__(self) -> None:
        self.attempts = 0

    def aio(self):
        async def stream():
            self.attempts += 1
            yield "placement:model-a", {"instance": "engine-a"}
            if self.attempts == 1:
                raise RetryableStreamError
            yield "placement:model-b", {"instance": "engine-b"}

        return stream()


class HangingItems:
    def __init__(self) -> None:
        self.attempts = 0

    def aio(self):
        async def stream():
            self.attempts += 1
            await asyncio.Event().wait()
            yield "unreachable", {}

        return stream()


def test_list_items_retries_terminated_stream(monkeypatch) -> None:
    async def run() -> None:
        items = StreamingItems(failures=2)
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items))
        sleep = AsyncMock()
        monkeypatch.setattr(kv, "StreamTerminatedError", RetryableStreamError)
        monkeypatch.setattr(kv.asyncio, "sleep", sleep)

        assert await store.list_items("placement:") == (("placement:model-a", {"instance": "engine-a"}),)
        assert items.attempts == 3
        assert [call.args[0] for call in sleep.await_args_list] == [0.2, 0.4]

    asyncio.run(run())


def test_list_items_discards_partial_retry(monkeypatch) -> None:
    async def run() -> None:
        items = PartiallyTerminatedItems()
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items))
        monkeypatch.setattr(kv, "StreamTerminatedError", RetryableStreamError)
        monkeypatch.setattr(kv.asyncio, "sleep", AsyncMock())

        assert await store.list_items("placement:") == (
            ("placement:model-a", {"instance": "engine-a"}),
            ("placement:model-b", {"instance": "engine-b"}),
        )
        assert items.attempts == 2

    asyncio.run(run())


def test_list_items_retries_stalled_stream(monkeypatch) -> None:
    async def run() -> None:
        items = HangingItems()
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items))
        monkeypatch.setattr(kv, "LIST_ITEMS_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(kv.asyncio, "sleep", AsyncMock())

        with pytest.raises(TimeoutError):
            await store.list_items("engine_instance:")
        assert items.attempts == 3

    asyncio.run(run())


def test_list_items_does_not_retry_other_errors(monkeypatch) -> None:
    async def run() -> None:
        items = StreamingItems(failures=1)
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items))
        monkeypatch.setattr(kv, "StreamTerminatedError", ValueError)
        sleep = AsyncMock()
        monkeypatch.setattr(kv.asyncio, "sleep", sleep)

        with pytest.raises(RetryableStreamError):
            await store.list_items("placement:")
        assert items.attempts == 1
        sleep.assert_not_awaited()

    asyncio.run(run())


def test_list_items_reraises_after_retry_limit(monkeypatch) -> None:
    async def run() -> None:
        items = StreamingItems(failures=3)
        store = kv.ModalKeyValueStore(SimpleNamespace(items=items))
        monkeypatch.setattr(kv, "StreamTerminatedError", RetryableStreamError)
        monkeypatch.setattr(kv.asyncio, "sleep", AsyncMock())

        with pytest.raises(RetryableStreamError):
            await store.list_items("engine_instance:")
        assert items.attempts == 3

    asyncio.run(run())


def test_lora_pool_registry_supports_publication_and_cleanup():
    async def run():
        from lilo.providers.local.kv import InMemoryKeyValueStore

        stores = {name: InMemoryKeyValueStore() for name in kv.STORE_NAMES}
        routed = kv.RoutedKeyValueStore(stores)
        key = "lora_pool:lilo-test"
        value = {"definition_id": "test", "touched_at": 1.0}
        await routed.put(key, value)
        assert await routed.get(key) == value
        assert await routed.list_items("lora_pool:") == ((key, value),)
        await routed.delete(key)
        assert await routed.list_items("lora_pool:") == ()

    asyncio.run(run())
