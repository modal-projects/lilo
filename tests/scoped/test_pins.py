import asyncio
from types import SimpleNamespace

from lilo.providers.modal.scoped_pins import (
    touch_pin,
    pin_demand,
    publish_pin,
    forget_pin_route,
)


class Registry:
    def __init__(self, values=None):
        self.values = values or {}
        self.get = SimpleNamespace(aio=self.read)
        self.put = SimpleNamespace(aio=self.write)
        self.pop = SimpleNamespace(aio=self.remove)

    async def read(self, key):
        return self.values.get(key)

    async def write(self, key, value):
        self.values[key] = value

    async def remove(self, key, default=None):
        return self.values.pop(key, default)


def test_demand_precedes_app_and_expires_without_successful_provisioning():
    async def check():
        registry = Registry()
        await touch_pin(registry, "pinned:m:1", now=1000)
        assert "pinned:m:1" in await pin_demand(registry, now=1100)
        assert await publish_pin(registry, "pinned:m:1", {"url": "old"}, now=1100)
        assert not await pin_demand(registry, now=1700)
        assert "pinned:m:1" not in registry.values
        assert not await publish_pin(registry, "pinned:m:1", {"url": "late"}, now=1700)
        # Failed creations leave no route and are subject to the same expiry.
        await touch_pin(registry, "pinned:m:2", now=2000)
        assert not await pin_demand(registry, now=2700)

    asyncio.run(check())


def test_active_lease_protects_demand_and_abandoned_lease_expires():
    async def check():
        registry = Registry()
        await touch_pin(registry, "pin", now=0, lease="request")
        assert "pin" in await pin_demand(registry, now=3800)
        await touch_pin(registry, "pin", now=3800, lease="request")
        assert "pin" in await pin_demand(registry, now=4500)
        assert not await pin_demand(registry, now=7800)
        await touch_pin(registry, "pin", now=7800, lease="request", release=True)
        assert not registry.values["pin_records"]

    asyncio.run(check())


def test_cleanup_cannot_erase_replacement_route_or_refreshed_demand():
    async def check():
        registry = Registry()
        await touch_pin(registry, "pin", now=1000)
        old, new = {"url": "old"}, {"url": "new"}
        await publish_pin(registry, "pin", old, now=1000)
        await pin_demand(registry, now=1700)
        # A waiting sampler recreates demand while the old app is closing.
        await touch_pin(registry, "pin", now=1701, lease="new-request")
        await publish_pin(registry, "pin", new, now=1701)
        await forget_pin_route(registry, "pin", old)
        assert registry.values["pin"] == new
        assert "pin" in await pin_demand(registry, now=1800)

    asyncio.run(check())


def test_owner_repairs_missing_route_after_cleanup_response_was_lost():
    async def check():
        registry = Registry()
        await touch_pin(registry, "pin", now=1000)
        old = {"url": "still-running"}
        await publish_pin(registry, "pin", old, now=1000)
        # Pruning committed, but its reply never reached the owner.
        await pin_demand(registry, now=1700)
        await touch_pin(registry, "pin", now=1701)
        await pin_demand(registry, now=1702, routes={"pin": old})
        assert registry.values["pin"] == old
    asyncio.run(check())
