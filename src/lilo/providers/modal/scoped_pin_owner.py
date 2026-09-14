"""Keep ephemeral pinned-app contexts alive in the process owning lilo.run."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import AsyncExitStack, asynccontextmanager

log = logging.getLogger(__name__)


async def reconcile_pins(manage, open_pin, stop, *, interval=1.0):
    # App startup yields a gateway before GPU readiness. Limit concurrent app
    # provisioning, but don't let one slow startup block polling or idle cleanup.
    starting = asyncio.Semaphore(2)
    tasks = {}
    routes = {}

    async def own(key):
        route = None
        try:
            async with AsyncExitStack() as stack:
                async with starting:
                    route = await stack.enter_async_context(open_pin(key))
                    accepted = await manage("pin_route", model_id=key, route=route)
                if accepted:
                    routes[key] = route
                    await asyncio.Future()  # Cancellation closes the owned app.
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Pinned sampler failed; demand will be retried: %s", key)
            await asyncio.sleep(5)
        finally:
            routes.pop(key, None)
            if route is not None:
                try:
                    await manage("forget_pin_route", model_id=key, route=route)
                except Exception:
                    log.exception("Could not remove pinned route: %s", key)

    try:
        while not stop.is_set():
            try:
                demand = await manage("pin_demand", route=dict(routes))
                for key, task in list(tasks.items()):
                    if task.done():
                        if not task.cancelled():
                            await task
                        del tasks[key]
                    elif key not in demand and not task.cancelling():
                        routes.pop(key, None)
                        task.cancel()
                for key in demand:
                    if key not in tasks:
                        tasks[key] = asyncio.create_task(own(key))
            except Exception:
                log.exception("Pinned demand reconciliation failed; will retry")
            await asyncio.sleep(interval)
    finally:
        for task in tasks.values():
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


class PinOwner:
    def __init__(self, manage, open_pin):
        self.stop = threading.Event()
        self.failure = None

        def run():
            try:
                asyncio.run(reconcile_pins(manage, open_pin, self.stop))
            except BaseException as exc:
                self.failure = exc
                log.exception("Pinned sampler owner failed")

        self.thread = threading.Thread(target=run, name="lilo-pinned-apps", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=120)
        if self.thread.is_alive():
            raise TimeoutError("Pinned sampler contexts did not close within 120s")
        if self.failure:
            raise RuntimeError("Pinned sampler owner failed") from self.failure


@asynccontextmanager
async def open_pinned_app(
    key, *, engine, name, image_id, registry_name, pool, proxy_secret
):
    import hashlib

    import modal

    from .scoped import register_sampler

    _, model_id, version = key.split(":")
    child = modal.App(name + "-pin-" + hashlib.sha256(key.encode()).hexdigest()[:12])
    server = register_sampler(
        child,
        engine=engine,
        image=modal.Image.from_id(image_id).add_local_python_source("lilo"),
        assets=modal.Volume.from_name("lilo-model-assets"),
        bulletin=modal.Volume.from_name("lilo-snapshot-bulletin", version=2),
        registry_name=registry_name,
        slot=None,
        model_id=model_id,
        version=int(version),
        pool=pool,
        proxy_secret=proxy_secret,
        name="sampler",
    )
    async with child.run.aio():
        yield {
            "url": await server.get_url.aio(),
            "function_id": server.object_id,
            "app_id": child.app_id,
        }
