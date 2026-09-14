"""Scoped Modal provisioning. Returned credentials work with unmodified Tinker."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import secrets
import time
import uuid

from lilo.engines import Engine


@dataclass(frozen=True)
class Pool:
    min_containers: int = 0
    max_containers: int = 8
    scaledown_window: int = 300

    def __post_init__(self):
        if not 0 <= self.min_containers <= self.max_containers or self.max_containers < 1:
            raise ValueError("require 0 <= min_containers <= max_containers and max >= 1")
        if self.scaledown_window < 1:
            raise ValueError("scaledown_window must be positive")


def stop_app(name_or_id: str) -> None:
    """Stop a specific owned app. Modal retains stopped app history."""
    import asyncio
    from modal.client import _Client
    from modal_proto import api_pb2
    import modal
    async def stop():
        app_id = name_or_id
        if not app_id.startswith("ap-"):
            try:
                app = await modal.App.lookup.aio(name_or_id)
            except modal.exception.NotFoundError:
                return
            app_id = app.app_id
        client = await _Client.from_env()
        await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app_id))
    asyncio.run(stop())


def stop_children(children, stop=stop_app) -> None:
    """Try every child before allowing the parent's context to exit."""
    failures = []
    for child in children:
        for attempt in range(3):
            try:
                stop(child)
                break
            except Exception as exc:
                if attempt == 2:
                    failures.append(exc)
                else:
                    time.sleep(0.5 * 2**attempt)
    if failures:
        raise ExceptionGroup("Some pinned sampler apps could not be stopped", failures)


@contextmanager
def run(*, engine: Engine, warm: bool = True, max_trainers: int = 1,
        latest: Pool | None = None, pinned: Pool | None = None,
        name: str = "lilo-run", api_key: str | None = None,
        checkpoint_volume: str = "lilo-checkpoints",
        proxy_secret=None):
    """Yield (url, api_key); stop pinned apps before the ephemeral parent.

    warm explicitly starts one trainer invocation; trainer min_containers is zero.
    Trainers remain until exit (no idle cleaner). max_trainers bounds separate
    Tinker training models, each with its own pre-registered latest pool.
    Pinned pools always have min_containers=0. A killed owner cannot run normal
    cleanup: pinned deployments then remain registered and scale to zero when idle.
    """
    import modal
    from lilo.providers.modal.scoped import build_app
    engine.validate()
    if isinstance(max_trainers, bool) or not isinstance(max_trainers, int) or max_trainers < 1:
        raise ValueError("max_trainers must be a positive integer")
    latest = latest or Pool()
    pinned = pinned or Pool()
    if pinned.min_containers:
        raise ValueError("pinned samplers must have min_containers=0")
    api_key = api_key or "tml-lilo-" + secrets.token_urlsafe(32)
    run_name = name + "-" + uuid.uuid4().hex[:12]
    registry_name = run_name + "-ownership"
    registry = modal.Dict.from_name(registry_name, create_if_missing=True)
    registry.put("closing", False)
    registry.put("children", [])
    resources = None
    try:
        resources = build_app(engine, run_name, registry_name, api_key,
                              max_trainers, latest, pinned, checkpoint_volume,
                              proxy_secret or modal.Secret.from_name("lilo-proxy"))
        app, api, manage, servers, prepare_assets, sampler_image = resources
        with app.run():
            registry.put("routes", [
                {"url": server.get_url(), "function_id": server.object_id}
                for server in servers
            ])
            registry.put("sampler_image_id", sampler_image.object_id)
            try:
                prepare_assets.remote()
                if warm:
                    manage.remote("warm")
                url = api.get_web_url()
                if not url:
                    raise RuntimeError("Modal did not return the API URL")
                yield url, api_key
            finally:
                registry.put("closing", True)
                # Serialized with pool creation; no late deploy can escape the list.
                failures = []
                try:
                    manage.remote("close")
                except Exception as exc:
                    failures.append(exc)
                try:
                    stop_children(registry.get("children") or [])
                except Exception as exc:
                    failures.append(exc)
                if failures:
                    raise ExceptionGroup("Scoped shutdown failed", failures)
    finally:
        # Also covers app startup/prepare failure before yield. Keep ownership
        # metadata if cleanup fails so operators can retry the exact owned apps.
        children = registry.get("children") or []
        stop_children(children)
        modal.Dict.objects.delete(registry_name, allow_missing=True)
