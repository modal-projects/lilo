import asyncio
import json
import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from lilo.inference import sampling
from lilo.inference.sampling import sample_task


def response(
    version: int,
    token: int = 5,
    *,
    end_version: int | None = None,
    cached_tokens: int = 0,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "meta_info": {
                "finish_reason": "stop",
                "output_token_logprobs": [[-0.1, token]],
                "weight_version_start": version,
                "weight_version_end": (version if end_version is None else end_version),
                "cached_tokens": cached_tokens,
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
    assert all("session_id" not in body for body in bodies)
    assert 0 <= bodies[0]["routed_dp_rank"] < 4
    assert {body["sampling_params"]["sampling_seed"] for body in bodies} == {
        10,
        11,
        12,
        13,
    }


def test_multiturn_affinity_tracks_prefix_cache_hit_rate() -> None:
    cached_sequences: dict[tuple[str, int], list[int]] = {}
    sessions = []
    bodies = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        session_id = http_request.headers["Modal-Session-ID"]
        body = json.loads(http_request.content)
        route = (session_id, body["routed_dp_rank"])
        prompt = body["input_ids"]
        previous = cached_sequences.get(route, [])
        cached_tokens = 0
        for actual, expected in zip(prompt, previous, strict=False):
            if actual != expected:
                break
            cached_tokens += 1
        output_token = 100 + len(bodies)
        cached_sequences[route] = [*prompt, output_token]
        sessions.append(session_id)
        bodies.append(body)
        return response(7, token=output_token, cached_tokens=cached_tokens)

    prompt = [11, 12, 13, 14]
    prompt_lengths = []
    cache_hits = []
    expected_reusable = []
    for turn in range(4):
        request = task()
        request["request_id"] = f"sample-a:{turn}"
        request["payload"]["cache_affinity_key"] = "trajectory-a"
        request["payload"]["prompt"]["chunks"][0]["tokens"] = list(prompt)
        result = asyncio.run(
            sample_task(
                request,
                "http://rollout",
                data_parallel_size=4,
                transport=httpx.MockTransport(handle),
            )
        )
        prompt_lengths.append(len(prompt))
        cache_hits.append(result["prompt_cache_hit_tokens"])
        if turn:
            expected_reusable.append(len(prompt) - 1)
        prompt.extend(result["sequences"][0]["tokens"])
        prompt.append(20 + turn)

    assert len(set(sessions)) == 1
    assert len({body["routed_dp_rank"] for body in bodies}) == 1
    assert {body["session_id"] for body in bodies} == set(sessions)
    assert cache_hits == [0, *expected_reusable]
    eligible_prefix_hit_rate = sum(cache_hits[1:]) / sum(expected_reusable)
    assert eligible_prefix_hit_rate == 1.0
    assert sum(cache_hits) / sum(prompt_lengths) > 0.7


def test_affinity_route_is_namespaced_by_sampling_session_and_key() -> None:
    sessions = []

    def handle(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["Modal-Session-ID"])
        return response(7)

    for sampling_session_id, affinity_key in (
        ("sample-a", "trajectory-a"),
        ("sample-b", "trajectory-a"),
        ("sample-a", "trajectory-b"),
    ):
        request = task()
        request["sampling_session_id"] = sampling_session_id
        request["payload"]["cache_affinity_key"] = affinity_key
        asyncio.run(
            sample_task(
                request,
                "http://rollout",
                transport=httpx.MockTransport(handle),
            )
        )

    assert len(set(sessions)) == 3
    assert all(session.startswith("affinity-") for session in sessions)


@pytest.mark.parametrize("key", ["", "   ", 42, "x" * 257])
def test_invalid_cache_affinity_key_is_rejected(key) -> None:
    request = task()
    request["payload"]["cache_affinity_key"] = key
    with pytest.raises(ValueError, match="cache_affinity_key"):
        asyncio.run(sample_task(request, "http://rollout"))


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
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(503, text="queue full")
                ),
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
