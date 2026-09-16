import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import modal
import pytest

from lilo.providers.modal.scoped_pin_owner import reconcile_pins


@pytest.mark.parametrize("model_id", ["model-a", "session-a:train:1"])
def test_pinned_app_preserves_complete_model_identity(monkeypatch, model_id):
    from lilo.providers.modal import scoped
    from lilo.providers.modal.scoped_pin_owner import open_pinned_app

    lifecycle, registrations = [], []

    @asynccontextmanager
    async def run():
        lifecycle.append("opened")
        try:
            yield
        finally:
            lifecycle.append("closed")

    child = SimpleNamespace(app_id="ap-child", run=SimpleNamespace(aio=run))
    image = SimpleNamespace(add_local_python_source=lambda _: "image-with-source")
    server = SimpleNamespace(
        object_id="fu-sampler",
        get_url=SimpleNamespace(aio=AsyncMock(return_value="https://sampler")),
    )

    def register(app, **kwargs):
        assert app is child
        registrations.append(kwargs)
        return server

    monkeypatch.setattr(modal, "App", lambda _: child)
    monkeypatch.setattr(modal.Image, "from_id", lambda _: image)
    monkeypatch.setattr(modal.Volume, "from_name", lambda *a, **k: object())
    monkeypatch.setattr(scoped, "register_sampler", register)

    async def check():
        async with open_pinned_app(
            f"pinned:{model_id}:7",
            engine=object(), name="test", image_id="im-test",
            registry_name="registry", pool=object(), proxy_secret=None,
        ) as route:
            assert route == {
                "url": "https://sampler", "function_id": "fu-sampler", "app_id": "ap-child"
            }

    asyncio.run(check())
    assert registrations[0]["model_id"] == model_id
    assert registrations[0]["version"] == 7
    assert lifecycle == ["opened", "closed"]


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


def test_owner_reconciles_demand_recreates_and_closes_contexts():
    async def check():
        demand = {"pinned:m:1": {}}
        routes, opened, closed = {}, [], []
        stop = threading.Event()

        async def manage(action, model_id=None, route=None):
            if action == "pin_demand":
                return dict(demand)
            if action == "pin_route":
                if model_id not in demand:
                    return False
                routes[model_id] = route
                return True
            if action == "forget_pin_route" and routes.get(model_id) == route:
                routes.pop(model_id, None)

        @asynccontextmanager
        async def open_pin(key):
            identity = len(opened)
            opened.append(key)
            try:
                yield {"url": f"url-{identity}"}
            finally:
                closed.append(key)

        task = asyncio.create_task(
            reconcile_pins(manage, open_pin, stop, interval=0.005)
        )
        await until(lambda: len(routes) == 1)
        await asyncio.sleep(0.02)
        assert opened == ["pinned:m:1"]
        demand.clear()
        await until(lambda: len(closed) == 1 and not routes)
        demand["pinned:m:1"] = {}
        await until(lambda: len(opened) == 2 and bool(routes))
        stop.set()
        await task
        assert len(closed) == 2 and not routes

    asyncio.run(check())


def test_slow_creation_does_not_block_polling_and_startup_is_limited():
    async def check():
        stop = threading.Event()
        started, closed = [], []
        polls = 0

        async def manage(action, **kwargs):
            nonlocal polls
            if action == "pin_demand":
                polls += 1
                return {f"pinned:m:{i}": {} for i in range(5)}

        @asynccontextmanager
        async def open_pin(key):
            started.append(key)
            try:
                await asyncio.Future()
                yield {}
            finally:
                closed.append(key)

        task = asyncio.create_task(
            reconcile_pins(manage, open_pin, stop, interval=0.005)
        )
        await until(lambda: polls > 4)
        assert len(started) == 2
        stop.set()
        await task
        assert len(closed) == 2

    asyncio.run(check())


def test_failed_creation_retries_while_demand_remains():
    async def check():
        stop = threading.Event()
        attempts = 0
        ready = asyncio.Event()
        closed = []

        async def manage(action, **kwargs):
            if action == "pin_demand":
                return {"pinned:m:1": {}}
            if action == "pin_route":
                ready.set()
                return True

        @asynccontextmanager
        async def open_pin(key):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary provisioning failure")
            try:
                yield {"url": "recovered"}
            finally:
                closed.append(key)

        task = asyncio.create_task(
            reconcile_pins(manage, open_pin, stop, interval=0.005)
        )
        try:
            async with asyncio.timeout(8):
                await ready.wait()
            assert attempts == 2
        finally:
            stop.set()
            await task
        assert closed == ["pinned:m:1"]

    asyncio.run(check())
