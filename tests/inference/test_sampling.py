import asyncio
import json
import pytest
import socket
from unittest.mock import AsyncMock, patch

import httpx
from lilo.inference import sampling
from lilo.inference.sampling import sample_task


def response(
    version: int,
    token: int = 5,
    *,
    end_version: int | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "meta_info": {
                "finish_reason": "stop",
                "output_token_logprobs": [[-0.1, token]],
                "weight_version_start": version,
                "weight_version_end": (version if end_version is None else end_version),
            }
        },
    )


def task(*, latest: bool = False) -> dict:
    return {
        "request_id": "request-a",
        "sampling_session_id": "sample-a",
        "model_id": "model-a",
        "publish_version": 7,
        "latest": latest,
        "payload": {
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
            "sampling_params": {"max_tokens": 1, "seed": 10},
        },
    }


def test_exact_sampling_pins_version() -> None:
    body = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return response(7)

    result = asyncio.run(
        sample_task(
            task(),
            "http://rollout",
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["weight_run_id"] == "model-a"
    assert body["weight_version"] == {"min_version": None, "exact_version": 7}
    assert result["sequences"][0]["tokens"] == [5]


@pytest.mark.parametrize("first", [None, [None, 1], {"logprob": None, "token_id": 1}])
def test_prompt_logprobs_preserve_undefined_first_token(first) -> None:
    request = task()
    request["payload"]["prompt_logprobs"] = True
    request["payload"]["prompt"]["chunks"][0]["tokens"] = [1, 2]
    body = response(7).json()
    body["meta_info"]["input_token_logprobs"] = [first, [-0.25, 2]]
    result = asyncio.run(
        sample_task(
            request,
            "http://rollout",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
        )
    )
    assert result["prompt_logprobs"] == [None, -0.25]


def test_grouped_sampling_shares_session_and_dp_rank() -> None:
    request = task()
    request["payload"]["num_samples"] = 4
    sessions = []
    bodies = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        sessions.append(http_request.headers["Modal-Session-ID"])
        bodies.append(json.loads(http_request.content))
        return response(7)

    result = asyncio.run(
        sample_task(
            request,
            "http://rollout",
            data_parallel_size=4,
            transport=httpx.MockTransport(handle),
        )
    )

    assert len(result["sequences"]) == 4
    assert sessions == ["request-a"] * 4
    assert len({body["routed_dp_rank"] for body in bodies}) == 1
    assert 0 <= bodies[0]["routed_dp_rank"] < 4
    assert {body["sampling_params"]["sampling_seed"] for body in bodies} == {
        10,
        11,
        12,
        13,
    }


def test_missing_max_tokens_defaults_to_remaining_context() -> None:
    request = task()
    request["payload"]["prompt"]["chunks"][0]["tokens"] = [1, 2, 3]
    request["payload"]["sampling_params"] = {"temperature": 0.7}
    body = {}

    def handle(http_request: httpx.Request) -> httpx.Response:
        body.update(json.loads(http_request.content))
        return response(7)

    asyncio.run(
        sample_task(
            request,
            "http://rollout",
            context_length=10,
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["sampling_params"]["max_new_tokens"] == 6


def test_explicit_max_tokens_wins_over_context_default() -> None:
    body = {}

    def handle(http_request: httpx.Request) -> httpx.Response:
        body.update(json.loads(http_request.content))
        return response(7)

    asyncio.run(
        sample_task(
            task(),
            "http://rollout",
            context_length=10,
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["sampling_params"]["max_new_tokens"] == 1


def test_prompt_filling_context_is_rejected() -> None:
    request = task()
    request["payload"]["prompt"]["chunks"][0]["tokens"] = [1, 2, 3]
    request["payload"]["sampling_params"] = {}

    with pytest.raises(ValueError, match="no room"):
        asyncio.run(
            sample_task(
                request,
                "http://rollout",
                context_length=3,
                transport=httpx.MockTransport(lambda _: response(7)),
            )
        )


def test_latest_sampling_sets_version_floor() -> None:
    body = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return response(8)

    asyncio.run(
        sample_task(
            task(latest=True),
            "http://rollout",
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["weight_version"] == {"min_version": 7, "exact_version": None}


def test_latest_sampling_accepts_in_place_version_advance() -> None:
    result = asyncio.run(
        sample_task(
            task(latest=True),
            "http://rollout",
            transport=httpx.MockTransport(lambda _: response(7, end_version=8)),
        )
    )

    assert result["sequences"][0]["tokens"] == [5]


def test_exact_sampling_retries_in_place_version_advance() -> None:
    responses = iter((response(7, end_version=8), response(7)))
    sleep = AsyncMock()
    with patch("lilo.inference.sampling.asyncio.sleep", sleep):
        result = asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(lambda _: next(responses)),
            )
        )

    sleep.assert_awaited_once()
    assert result["sequences"][0]["tokens"] == [5]


def test_sampling_backs_off_version_conflicts() -> None:
    statuses = iter((409, 200))

    def handle(_: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return response(7) if status == 200 else httpx.Response(status)

    sleep = AsyncMock()
    with (
        patch("lilo.inference.sampling.asyncio.sleep", sleep),
        patch("lilo.inference.sampling.random.uniform", return_value=1),
    ):
        result = asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(handle),
            )
        )

    sleep.assert_awaited_once_with(1.0)
    assert result["sequences"][0]["tokens"] == [5]


def test_sampling_reroutes_immediate_overload_response() -> None:
    sessions = []

    def handle(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["Modal-Session-ID"])
        assert request.url.path == "/generate"
        if len(sessions) == 1:
            return httpx.Response(503, text="queue full")
        return response(7)

    with (
        patch(
            "lilo.inference.sampling.asyncio.sleep",
            AsyncMock(),
        ),
        patch("lilo.inference.sampling.random.uniform", return_value=1),
    ):
        result = asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(handle),
            )
        )

    assert sessions == ["request-a", "request-a:retry-1"]
    assert result["sequences"][0]["tokens"] == [5]


def test_sampling_sends_extra_headers() -> None:
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return response(7)

    asyncio.run(
        sample_task(
            task(),
            "http://rollout",
            transport=httpx.MockTransport(handle),
            api_key="key",
            headers={"Modal-Key": "wk-a", "Modal-Secret": "ws-a"},
        )
    )

    assert seen["modal-key"] == "wk-a"
    assert seen["modal-secret"] == "ws-a"
    assert seen["x-api-key"] == "key"


def test_sampling_reports_saturation_and_touches_while_waiting() -> None:
    on_wait = AsyncMock()
    with (
        patch("lilo.inference.sampling.asyncio.sleep", AsyncMock()),
        pytest.raises(RuntimeError, match=r"rollout saturated after 0s: HTTP 503"),
    ):
        asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                retry_timeout=0,
                transport=httpx.MockTransport(lambda _: httpx.Response(503, text="queue full")),
                on_wait=on_wait,
            )
        )

    on_wait.assert_not_awaited()
    responses = iter((httpx.Response(503, text="queue full"), response(7)))
    with patch("lilo.inference.sampling.asyncio.sleep", AsyncMock()):
        asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(lambda _: next(responses)),
                on_wait=on_wait,
            )
        )

    on_wait.assert_awaited_once()


def test_exact_sampling_rejects_wrong_response_version() -> None:
    hosts = []

    async def gateways() -> tuple[str, ...]:
        return ("http://replica-a", "http://replica-b")

    def handle(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return response(6 if request.url.host == "replica-a" else 7)

    with patch(
        "lilo.inference.sampling.asyncio.sleep",
        AsyncMock(),
    ):
        result = asyncio.run(
            sample_task(
                task(),
                gateways,
                transport=httpx.MockTransport(handle),
            )
        )

    assert hosts == ["replica-a", "replica-b"]
    assert result["sequences"][0]["tokens"] == [5]


def test_default_transport_enables_tcp_keepalive() -> None:
    transport = sampling.keepalive_transport()
    assert transport._pool._socket_options == list(sampling.TCP_KEEPALIVE_OPTIONS)
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in sampling.TCP_KEEPALIVE_OPTIONS

    with patch.object(
        sampling,
        "keepalive_transport",
        return_value=httpx.MockTransport(lambda _: response(7)),
    ) as default_transport:
        asyncio.run(sample_task(task(), "http://rollout"))
    default_transport.assert_called_once_with()
