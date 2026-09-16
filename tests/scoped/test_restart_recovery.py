import asyncio
from types import SimpleNamespace

import pytest

from lilo.control_plane import ControlPlane
from lilo.control_plane.records import PlacementRecord
from lilo.errors import ModelLost
from lilo.providers.local import InMemoryKeyValueStore
from lilo.providers.modal.scoped_assignment import claim_model
from tests.scoped.test_replacement import Registry


@pytest.mark.parametrize("loss_reported", [False, True])
def test_restart_reuses_trainer_and_retires_old_model(loss_reported):
    async def run():
        kv = InMemoryKeyValueStore()
        await kv.put("trainer_demand:a", {})
        await kv.put("trainer_demand:b", {})
        placement = PlacementRecord(
            model_id="a",
            engine_definition_id="definition",
            engine_instance_id="same-invocation",
            engine_boot_id="old-boot",
            placed_at=0,
        )
        await kv.put("placement:a", placement.model_dump())
        restarted = SimpleNamespace(
            instance_id="same-invocation",
            boot_id="new-boot",
            terminal=False,
        )

        async def active(_):
            return [restarted]

        async def get(instance_id):
            assert instance_id == "same-invocation"
            return restarted

        async def spawn(_):
            raise AssertionError("recovery must reuse the existing trainer")

        engines = SimpleNamespace(
            active_instances=active, get_instance=get, spawn_instance=spawn
        )
        registry = Registry({"slot:0": "a", "routes": [{}, {"url": "latest"}]})
        if loss_reported:
            # Exercise the real loss path that deletes both placement and demand.
            plane = ControlPlane(kv, engines)
            with pytest.raises(ModelLost):
                await plane._live_instance(placement)
            assert await kv.get("placement:a") is None
        with pytest.raises(ValueError, match="trainer was lost"):
            await claim_model(registry, kv, engines, "definition", "a")
        await claim_model(registry, kv, engines, "definition", "b")
        assert registry.values["slot:0"] == "b"
        assert registry.values["retired:a"] is True
        assert await kv.get("trainer_demand:a") is None
        assert await kv.get("trainer_demand:b") == {}
        with pytest.raises(ValueError, match="replaced"):
            await claim_model(registry, kv, engines, "definition", "a")

    asyncio.run(run())


@pytest.mark.parametrize("placed", [False, True])
def test_pending_and_healthy_models_keep_the_only_slot(placed):
    async def run():
        kv = InMemoryKeyValueStore()
        await kv.put("trainer_demand:a", {})
        instance = SimpleNamespace(
            instance_id="trainer", boot_id="boot", terminal=False
        )
        if placed:
            await kv.put(
                "placement:a",
                PlacementRecord(
                    model_id="a",
                    engine_definition_id="definition",
                    placed_at=0,
                    engine_instance_id="trainer",
                    engine_boot_id="boot",
                ).model_dump(),
            )

        async def active(_):
            return [instance]

        async def get(_):
            return instance

        engines = SimpleNamespace(active_instances=active, get_instance=get)
        registry = Registry({"slot:0": "a", "routes": [{}, {"url": "latest"}]})
        await claim_model(registry, kv, engines, "definition", "a")
        with pytest.raises(ValueError, match="already active"):
            await claim_model(registry, kv, engines, "definition", "b")
        assert registry.values["slot:0"] == "a"
        assert await kv.get("trainer_demand:a") == {}

    asyncio.run(run())
