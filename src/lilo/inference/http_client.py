"""HTTP connection reuse for the persistent event loop in a sampling worker."""

import asyncio
from weakref import WeakKeyDictionary

import httpx

from .sampling import TCP_KEEPALIVE_OPTIONS

_clients = WeakKeyDictionary()


def sampling_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=3000,
            trust_env=False,
            transport=httpx.AsyncHTTPTransport(
                socket_options=list(TCP_KEEPALIVE_OPTIONS),
                limits=httpx.Limits(
                    # A worker can serve 128 logical requests, each with multiple
                    # samples. Don't introduce a hidden 100-connection queue.
                    max_connections=None,
                    max_keepalive_connections=256,
                    keepalive_expiry=60,
                ),
            ),
        )
        _clients[loop] = client
    return client


async def close_sampling_client() -> None:
    """Explicit cleanup for embedded workers/tests; process exit closes sockets."""
    client = _clients.pop(asyncio.get_running_loop(), None)
    if client is not None:
        await client.aclose()
