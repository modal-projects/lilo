"""Scoped Modal provisioning. Returned credentials work with unmodified Tinker."""

from __future__ import annotations

import secrets
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial

from lilo.engines import Engine


@dataclass(frozen=True)
class Pool:
    min_containers: int = 0
    max_containers: int = 8
    scaledown_window: int = 300

    def __post_init__(self):
        if (
            not 0 <= self.min_containers <= self.max_containers
            or self.max_containers < 1
        ):
            raise ValueError(
                "require 0 <= min_containers <= max_containers and max >= 1"
            )
        if self.scaledown_window < 1:
            raise ValueError("scaledown_window must be positive")


@contextmanager
def run(
    *,
    engine: Engine,
    warm: bool = True,
    latest: Pool | None = None,
    name: str = "lilo",
    api_key: str | None = None,
    checkpoint_volume: str = "lilo-checkpoints",
    proxy_secret=None,
    telemetry_secret=None,
):
    """Yield (url, api_key); stop pinned apps before the ephemeral parent.

    warm explicitly starts one trainer invocation; trainer min_containers is zero.
    One training model is active at a time. A lost trainer may be replaced
    by creating a new full training client in the same deployment.
    Pinned pools have min_containers=0 and are ephemeral apps owned by this
    process. Owner loss stops them after Modal's heartbeat timeout.
    """
    import modal

    from lilo.providers.modal.scoped import build_app
    from lilo.providers.modal.scoped_pin_owner import PinOwner, open_pinned_app

    engine.validate()
    latest = latest or Pool()
    pinned = Pool()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Scoped runs require Python 3.12 to match the bundled runtime images"
        )
    api_key = api_key or "tml-lilo-" + secrets.token_urlsafe(32)
    run_name = name + "-" + uuid.uuid4().hex[:12]
    registry_name = run_name + "-ownership"
    registry = modal.Dict.from_name(registry_name, create_if_missing=True)
    registry.put("closing", False)
    owner = None
    failure = None
    try:
        resources = build_app(
            engine,
            run_name,
            registry_name,
            api_key,
            1,
            latest,
            pinned,
            checkpoint_volume,
            proxy_secret or modal.Secret.from_name("lilo-proxy"),
            telemetry_secret=telemetry_secret,
        )
        app, api, manage, servers, prepare_assets, sampler_image = resources
        with app.run():
            registry.put(
                "routes",
                [
                    {"url": server.get_url(), "function_id": server.object_id}
                    for server in servers
                ],
            )
            sampler_engine = replace(
                engine,
                training=replace(
                    engine.training,
                    hf_checkpoint=engine.training.hf_checkpoint
                    or "/assets/" + engine.model.rsplit("/", 1)[-1],
                ),
            )
            owner = PinOwner(
                manage.remote.aio,
                partial(
                    open_pinned_app,
                    engine=sampler_engine,
                    name=run_name,
                    image_id=sampler_image.object_id,
                    registry_name=registry_name,
                    pool=pinned,
                    proxy_secret=proxy_secret,
                ),
            )
            try:
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
                    # No new demand is accepted; child contexts close before parent.
                    owner.close()
                    owner = None
            except BaseException as exc:
                failure = exc
                raise
        # Modal suppresses KeyboardInterrupt on app.run exit. The provisioning
        # context must not turn an interrupted user's training block into success.
        if failure is not None:
            raise failure
    finally:
        # Keep ownership metadata if child context shutdown fails.
        if owner is not None:
            owner.close()
        modal.Dict.objects.delete(registry_name, allow_missing=True)
