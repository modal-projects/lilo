"""Stitch pool routing by hydrated IDs, including ephemeral parent apps."""
from __future__ import annotations

import os

import httpx
import modal
from stitch.pools.base import Pool

from .fft_pool import proxy_auth_headers
from modal._utils.async_utils import synchronize_api


class ScopedFlashPool(Pool):
    def __init__(self, route: dict) -> None:
        self.route = route

    def gateway_url(self) -> str:
        return self.route["url"].rstrip("/")

    async def discover_replicas_async(self) -> list[str]:
        return await list_replicas.aio(self.route["function_id"])

    def discover_replicas(self) -> list[str]:
        return list_replicas(self.route["function_id"])

    def wake(self, replicas, ref) -> None:
        # Requests enter through the authenticated gateway, selecting a replica.
        import logging
        with httpx.Client(timeout=5, headers=proxy_auth_headers()) as client:
            for upstream in replicas:
                try:
                    client.post(self.gateway_url() + "/wake", headers={
                        "modal-flash-upstream": upstream.removeprefix("https://").removeprefix("http://")
                    }).raise_for_status()
                except httpx.HTTPError:
                    logging.getLogger(__name__).warning("sampler wake failed", exc_info=True)


def publication_pool(definition_id: str, model_id: str) -> Pool:
    registry_name = os.environ.get("LILO_SCOPED_REGISTRY")
    if registry_name:
        route = modal.Dict.from_name(registry_name)["model:" + model_id]
        return ScopedFlashPool(route)
    from .fft_pool import FFTLatestPool
    return FFTLatestPool(definition_id, model_id)


@synchronize_api
async def set_minimum(function_id: str, minimum: int) -> None:
    # Server has no public from_id constructor. Avoid deployed-name lookup,
    # which cannot address an ephemeral server.
    from modal.client import _Client
    from modal_proto import api_pb2
    client = await _Client.from_env()
    await client.stub.FunctionUpdateSchedulingParams(
        api_pb2.FunctionUpdateSchedulingParamsRequest(
            function_id=function_id,
            settings=api_pb2.AutoscalerSettings(min_containers=minimum),
        )
    )


@synchronize_api
async def list_replicas(function_id: str) -> list[str]:
    from modal.client import _Client
    from modal_proto import api_pb2
    client = await _Client.from_env()
    response = await client.stub.FlashContainerList(
        api_pb2.FlashContainerListRequest(function_id=function_id)
    )
    return [f"{c.host}:{c.port}" if c.port else c.host
            for c in response.containers if c.host]
