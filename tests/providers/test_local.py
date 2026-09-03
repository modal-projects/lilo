import asyncio

import pytest

from tests.support import EchoExecutor
from lilo.errors import RecordNotFound
from lilo.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
)
from lilo.providers.modal.kv import RoutedKeyValueStore


def test_values_cross_a_serialization_boundary() -> None:
    async def run() -> None:
        store = InMemoryKeyValueStore()
        original = {"items": [1]}
        await store.put("key", original)
        original["items"].append(2)
        stored = await store.get("key")
        assert stored == {"items": [1]}
        stored["items"].append(3)
        assert await store.get("key") == {"items": [1]}

    asyncio.run(run())


def test_list_items_returns_matching_values() -> None:
    async def run() -> None:
        store = InMemoryKeyValueStore()
        await store.put("session:a", {"n": 1})
        await store.put("session_last_seen:a", {"n": 2})
        await store.put("model:a", {"n": 3})
        assert await store.list_items("session:", "session_last_seen:") == (
            ("session:a", {"n": 1}),
            ("session_last_seen:a", {"n": 2}),
        )

    asyncio.run(run())


def test_routed_store_lists_only_matching_families() -> None:
    async def run() -> None:
        sessions = InMemoryKeyValueStore()
        models = InMemoryKeyValueStore()
        store = RoutedKeyValueStore(
            {
                "sessions": sessions,
                "models": models,
                "engines": InMemoryKeyValueStore(),
                "sampling_sessions": InMemoryKeyValueStore(),
                "exports": InMemoryKeyValueStore(),
                "artifacts": InMemoryKeyValueStore(),
            }
        )
        await store.put("session:a", 1)
        await store.put("model:a", 2)
        await store.put("placement:a", 3)
        assert await sessions.list_keys("") == ("session:a",)
        assert await models.list_keys("") == ("model:a", "placement:a")
        assert await store.list_items("session:", "placement:") == (
            ("placement:a", 3),
            ("session:a", 1),
        )

    asyncio.run(run())


def test_put_if_absent_has_one_winner() -> None:
    async def run() -> None:
        store = InMemoryKeyValueStore()
        results = await asyncio.gather(
            *(
                store.put_if_absent("placement:model", {"writer": index})
                for index in range(32)
            )
        )
        assert sum(result.created for result in results) == 1
        winner = next(result.value for result in results if result.created)
        assert all(result.value == winner for result in results)
        assert await store.get("placement:model") == winner

    asyncio.run(run())


def test_platform_reuses_and_replaces_instances() -> None:
    async def run() -> None:
        definition_id = "qwen3_4b_lora32_16k"
        engines = LocalEnginePlatform(definition_id, EchoExecutor)
        first = await engines.ensure_instance(definition_id)
        assert await engines.ensure_instance(definition_id) == first
        assert not first.terminal

        dead = await engines.mark_dead(first.instance_id)
        assert dead.terminal
        replacement = await engines.ensure_instance(definition_id)
        assert replacement.instance_id != first.instance_id
        assert replacement.state == "running"

    asyncio.run(run())


def test_platform_only_embodies_its_own_definition() -> None:
    async def run() -> None:
        engines = LocalEnginePlatform(
            "qwen3_4b_lora32_16k",
            EchoExecutor,
        )
        with pytest.raises(RecordNotFound):
            await engines.ensure_instance("other_model")

    asyncio.run(run())
