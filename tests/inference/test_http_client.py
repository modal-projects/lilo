import asyncio

import httpx

from lilo.inference.http_client import close_sampling_client, sampling_client
from lilo.inference.sampling import sample_task


def test_worker_client_is_reused_and_can_be_closed():
    async def scenario():
        first = sampling_client()
        try:
            await asyncio.sleep(0)
            assert sampling_client() is first
        finally:
            await close_sampling_client()
        assert first.is_closed
        second = sampling_client()
        try:
            assert second is not first
        finally:
            await close_sampling_client()

    asyncio.run(scenario())
    asyncio.run(scenario())  # A second worker loop must not borrow old sockets.


def test_shared_client_keeps_auth_and_routing_headers_request_scoped():
    seen = []

    def upstream(request):
        seen.append((request.headers["x-api-key"], request.headers["Modal-Session-ID"]))
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "finish_reason": "stop",
                    "output_token_logprobs": [[-0.1, 5]],
                    "weight_version_start": 7,
                    "weight_version_end": 7,
                }
            },
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:

            async def sample(key):
                return await sample_task(
                    {
                        "request_id": key,
                        "sampling_session_id": key,
                        "model_id": "model-a",
                        "publish_version": 7,
                        "latest": False,
                        "payload": {
                            "prompt": {
                                "chunks": [{"type": "encoded_text", "tokens": [1]}]
                            },
                            "sampling_params": {"max_tokens": 1},
                        },
                    },
                    "http://replica",
                    client=client,
                    api_key=key,
                )

            await asyncio.gather(sample("a"), sample("b"))
            assert not client.is_closed
            assert "x-api-key" not in client.headers
            assert "Modal-Session-ID" not in client.headers

    asyncio.run(scenario())
    assert sorted(seen) == [("a", "a"), ("b", "b")]
