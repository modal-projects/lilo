from __future__ import annotations

import asyncio
import hashlib
import random
import socket
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

import httpx
from stitch.publish import constrain_request

from lilo.telemetry.sample_stats import SampleAttempt, event_sink, timing_metadata
from lilo.telemetry.http_trace import HTTPTrace

RETRY_INITIAL_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 5.0
TCP_KEEPALIVE_OPTIONS = (
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5),
)
GatewayResolver = Callable[[], Awaitable[tuple[str, ...]]]


def keepalive_transport() -> httpx.AsyncHTTPTransport:
    return httpx.AsyncHTTPTransport(socket_options=list(TCP_KEEPALIVE_OPTIONS))


async def sample_task(
    task: dict[str, Any],
    gateway: str | GatewayResolver,
    *,
    timeout: float = 3000.0,
    retry_timeout: float = 3000.0,
    data_parallel_size: int = 1,
    context_length: int | None = None,
    context_reserve_tokens: int = 1,
    transport: httpx.AsyncBaseTransport | None = None,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    on_wait: Callable[[], Awaitable[None]] | None = None,
    stats: dict[str, Any] | None = None,
    session_id: str | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    request = task["payload"]
    prompt = _prompt_tokens(request["prompt"])
    params = _sampling_params(
        request.get("sampling_params") or {},
        remaining_context=(
            context_length - len(prompt) - context_reserve_tokens
            if context_length is not None
            else None
        ),
    )
    model_id = task.get("model_id")
    version = task.get("publish_version")
    latest = bool(task.get("latest"))
    num_samples = int(request.get("num_samples", 1))
    prompt_logprobs = bool(request.get("prompt_logprobs"))
    topk_prompt_logprobs = int(request.get("topk_prompt_logprobs") or 0)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if topk_prompt_logprobs < 0:
        raise ValueError("topk_prompt_logprobs must be non-negative")
    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be positive")
    request_id = str(task["request_id"])
    cache_affinity_id = _cache_affinity_id(task)
    routing_session_id = cache_affinity_id or session_id or request_id
    routed_dp_rank = (
        int.from_bytes(
            hashlib.sha256(routing_session_id.encode()).digest()[:8],
            "big",
        )
        % data_parallel_size
    )
    headers = dict(headers or {})
    if api_key:
        headers["x-api-key"] = api_key

    sequence_stats = [{} for _ in range(num_samples)]
    trace_attrs = {
        "request_id": str(task["request_id"]),
        "sampling_session_id": task.get("sampling_session_id"),
    }
    emit = event_sink(task, on_event)

    if client is not None and transport is not None:
        raise ValueError("pass either client or transport")
    client_context = (
        nullcontext(client)
        if client is not None
        else httpx.AsyncClient(
            timeout=timeout,
            trust_env=False,
            transport=transport or keepalive_transport(),
            headers=headers,
        )
    )
    async with client_context as active_client:
        outputs = await asyncio.gather(
            *(
                _sample_one(
                    active_client,
                    prompt,
                    params,
                    model_id=model_id,
                    version=version,
                    latest=latest,
                    index=index,
                    request_id=request_id,
                    routing_session_id=routing_session_id,
                    cache_affinity_id=cache_affinity_id,
                    routed_dp_rank=routed_dp_rank,
                    prompt_logprobs=prompt_logprobs and index == 0,
                    topk_prompt_logprobs=(topk_prompt_logprobs if index == 0 else 0),
                    retry_timeout=retry_timeout,
                    gateway=gateway,
                    on_wait=on_wait,
                    stats=sequence_stats[index],
                    on_event=emit,
                    trace_attrs=trace_attrs,
                    base_headers=headers,
                    request_timeout=timeout,
                )
                for index in range(num_samples)
            )
        )

    sequences = [_sequence(output) for output in outputs]
    if stats is not None:
        meta = outputs[0].get("meta_info") or {}
        stats.update(sequence_stats[0])
        stats["sequence_stats"] = sequence_stats
        stats["wait_s"] = sum(row.get("wait_s", 0.0) for row in sequence_stats)
        stats["reroutes"] = sum(row.get("reroutes", 0) for row in sequence_stats)
        stats.update(
            prompt_tokens=len(prompt),
            generated_tokens=sum(len(sequence["tokens"]) for sequence in sequences),
            version_served_start=meta.get("weight_version_start"),
            version_served_end=meta.get("weight_version_end"),
        )
    response: dict[str, Any] = {"type": "sample", "sequences": sequences}
    if prompt_logprobs:
        response["prompt_logprobs"] = _logprobs(
            (outputs[0].get("meta_info") or {}).get("input_token_logprobs")
        )
    if topk_prompt_logprobs:
        response["topk_prompt_logprobs"] = _topk_logprobs(
            (outputs[0].get("meta_info") or {}).get("input_top_logprobs")
        )
    response["prompt_cache_hit_tokens"] = int(
        (outputs[0].get("meta_info") or {}).get("cached_tokens") or 0
    )
    return response


async def _sample_one(
    client: httpx.AsyncClient,
    prompt: list[int],
    params: dict[str, Any],
    *,
    model_id: str | None,
    version: int | None,
    latest: bool,
    index: int,
    request_id: str,
    routing_session_id: str,
    cache_affinity_id: str | None,
    routed_dp_rank: int,
    prompt_logprobs: bool,
    topk_prompt_logprobs: int,
    retry_timeout: float,
    gateway: str | GatewayResolver,
    on_wait: Callable[[], Awaitable[None]] | None = None,
    stats: dict[str, Any] | None = None,
    on_event: Callable[[dict], None] = lambda event: None,
    trace_attrs: dict | None = None,
    base_headers: dict | None = None,
    request_timeout: float = 3000.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "input_ids": prompt,
        "sampling_params": dict(params),
        "return_logprob": True,
        "routed_dp_rank": routed_dp_rank,
    }
    if cache_affinity_id is not None:
        body["session_id"] = cache_affinity_id
    if prompt_logprobs or topk_prompt_logprobs:
        body["logprob_start_len"] = 0
    if topk_prompt_logprobs:
        body["top_logprobs_num"] = topk_prompt_logprobs
    if params.get("seed") is not None:
        body["sampling_params"].pop("seed")
        body["sampling_params"]["sampling_seed"] = int(params["seed"]) + index
    if model_id is not None:
        if version is None:
            raise ValueError("trained-model sampling requires a publish version")
        body["weight_run_id"] = model_id
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    deadline = started_at + retry_timeout
    delay = RETRY_INITIAL_DELAY_SECONDS
    reroute_attempt = 0
    transport_retries = 0
    rejected: set[str] = set()
    waiting_logged = False
    physical_attempt = 0
    while True:
        cause: httpx.TransportError | None = None
        resolved = (
            (gateway.rstrip("/"),) if isinstance(gateway, str) else await gateway()
        )
        gateways = tuple(item for item in resolved if item not in rejected)
        if not gateways:
            reason = "no compatible rollout replicas"
            if index == 0 and not waiting_logged:
                print(
                    "execute_sample route_wait "
                    f"request={request_id} model={model_id} "
                    f"version={version}",
                    flush=True,
                )
                waiting_logged = True
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError(
                    f"rollout generate unavailable after {retry_timeout:g}s: {reason}"
                )
            if on_wait is not None:
                await on_wait()
            await _backoff(stats, min(remaining, delay * random.uniform(0.8, 1.2)))
            delay = min(RETRY_MAX_DELAY_SECONDS, delay * 2)
            rejected.clear()
            continue
        waiting_logged = False
        gateway_url = gateways[(index + reroute_attempt) % len(gateways)]
        modal_session_id = routing_session_id
        if reroute_attempt:
            modal_session_id = f"{routing_session_id}:retry-{reroute_attempt}"
        headers = {**(base_headers or {}), "Modal-Session-ID": modal_session_id}
        if model_id is not None:
            constrain_request(
                body,
                headers,
                latest=int(version) if latest else None,
                exact=None if latest else int(version),
            )
        physical_attempt += 1
        observation = SampleAttempt(
            on_event,
            {
                **(trace_attrs or {}),
                "sequence_index": index,
                "attempt_number": physical_attempt,
                "routed_dp_rank": routed_dp_rank,
                "upstream": gateway_url,
                "prompt_tokens": len(prompt),
            },
        )
        body["rid"] = observation.id
        attempt_result = observation.attrs
        try:
            response = await client.post(
                f"{gateway_url}/generate",
                json=body,
                headers=headers,
                timeout=request_timeout,
                extensions={"trace": HTTPTrace(attempt_result).record},
            )
        except httpx.TransportError as exc:
            attempt_result["error_type"] = type(exc).__name__
            cause = exc
            reason = f"{type(exc).__name__}: {exc}"
            if not isinstance(gateway, str):
                rejected.add(gateway_url)
            reroute_attempt += 1
            transport_retries += 1
            print(
                "execute_sample reroute "
                f"request={request_id} sample={index} attempt={reroute_attempt} "
                f"elapsed_seconds={loop.time() - started_at:.3f} "
                f"upstream={gateway_url} reason={reason}",
                flush=True,
            )
        else:
            attempt_result["http_status"] = response.status_code
            retryable = (
                response.status_code == 409
                or response.status_code >= 500
                or (response.status_code == 404 and not isinstance(gateway, str))
            )
            if not retryable:
                if response.is_error:
                    raise RuntimeError(
                        f"rollout generate returned {response.status_code}: {response.text[:500]}"
                    )
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("SGLang returned a non-object response")
                attempt_result.update(
                    generated_tokens=len(_sequence(result)["tokens"]),
                    **timing_metadata(result.get("meta_info") or {}),
                )
                version_error = (
                    _response_version_error(result, int(version), minimum=latest)
                    if model_id is not None and version is not None
                    else None
                )
                if version_error is not None:
                    reason = version_error
                    if not isinstance(gateway, str):
                        rejected.add(gateway_url)
                    reroute_attempt += 1
                else:
                    if reroute_attempt and (index == 0 or transport_retries):
                        print(
                            "execute_sample reroute_complete "
                            f"request={request_id} sample={index} attempts={reroute_attempt} "
                            f"transport_retries={transport_retries} elapsed_seconds={loop.time() - started_at:.3f} "
                            f"upstream={gateway_url}",
                            flush=True,
                        )
                    attempt_result["ok"] = True
                    if stats is not None:
                        stats["reroutes"] = stats.get("reroutes", 0) + reroute_attempt
                        stats["upstream"] = gateway_url
                    return result
            else:
                reason = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code >= 500:
                    if not isinstance(gateway, str):
                        rejected.add(gateway_url)
                    reroute_attempt += 1
                    if index == 0:
                        print(
                            "execute_sample reroute "
                            f"request={request_id} attempt={reroute_attempt} "
                            f"upstream={gateway_url} status={response.status_code} "
                            f"body={response.text[:120]!r}",
                            flush=True,
                        )
        finally:
            observation.finish()
        remaining = deadline - loop.time()
        if remaining <= 0:
            state = "saturated" if reason.startswith("HTTP 503") else "unavailable"
            raise RuntimeError(
                f"rollout {state} after {retry_timeout:g}s: {reason}"
            ) from cause
        if on_wait is not None:
            await on_wait()
        await _backoff(stats, min(remaining, delay * random.uniform(0.8, 1.2)))
        delay = min(RETRY_MAX_DELAY_SECONDS, delay * 2)


async def _backoff(stats: dict[str, Any] | None, seconds: float) -> None:
    if stats is not None:
        stats["wait_s"] = stats.get("wait_s", 0.0) + seconds
    await asyncio.sleep(seconds)


def _cache_affinity_id(task: dict[str, Any]) -> str | None:
    affinity_key = task["payload"].get("cache_affinity_key")
    if affinity_key is None:
        return None
    if (
        not isinstance(affinity_key, str)
        or not affinity_key.strip()
        or len(affinity_key) > 256
    ):
        raise ValueError(
            "cache_affinity_key must be a non-empty string of at most 256 characters"
        )
    sampling_session_id = str(task["sampling_session_id"])
    digest = hashlib.sha256(
        f"{sampling_session_id}\0{affinity_key}".encode()
    ).hexdigest()
    return f"affinity-{digest[:32]}"


def _prompt_tokens(prompt: dict[str, Any]) -> list[int]:
    tokens: list[int] = []
    for chunk in prompt.get("chunks") or ():
        if not isinstance(chunk, dict) or chunk.get("type") != "encoded_text":
            raise ValueError("only encoded_text prompt chunks are supported")
        values = chunk.get("tokens")
        if not isinstance(values, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in values
        ):
            raise ValueError("encoded_text tokens must be integers")
        tokens.extend(values)
    return tokens


def _sampling_params(
    params: dict[str, Any],
    *,
    remaining_context: int | None = None,
) -> dict[str, Any]:
    output = {
        key: params[key]
        for key in ("temperature", "top_k", "top_p", "seed")
        if params.get(key) is not None
    }
    stop = params.get("stop")
    if stop == []:
        output["ignore_eos"] = True
    elif (
        isinstance(stop, list)
        and stop
        and all(
            not isinstance(token, bool) and isinstance(token, int) for token in stop
        )
    ):
        output["stop_token_ids"] = stop
    elif stop is not None:
        output["stop"] = stop
    if params.get("max_tokens") is not None:
        output["max_new_tokens"] = int(params["max_tokens"])
    elif remaining_context is not None:
        if remaining_context <= 0:
            raise ValueError("prompt leaves no room in the context window")
        output["max_new_tokens"] = remaining_context
    return output


def _response_version_error(
    response: dict[str, Any],
    expected: int,
    *,
    minimum: bool = False,
) -> str | None:
    meta = response.get("meta_info")
    source = meta if isinstance(meta, dict) else response
    start = source.get("weight_version_start")
    end = source.get("weight_version_end")
    if isinstance(start, int) and isinstance(end, int):
        if minimum and start >= expected and end >= start:
            return None
        if not minimum and start == end == expected:
            return None
    relation = "at least" if minimum else "exactly"
    return f"rollout generated weight versions {start!r}..{end!r}, expected {relation} {expected}"


def _sequence(output: dict[str, Any]) -> dict[str, Any]:
    meta = output.get("meta_info") or {}
    token_logprobs = meta.get("output_token_logprobs") or ()
    tokens = [_token(item) for item in token_logprobs]
    logprobs = [_logprob(item) for item in token_logprobs]
    finish = meta.get("finish_reason")
    if isinstance(finish, dict):
        finish = finish.get("type")
    if finish == "abort":
        raise RuntimeError("SGLang aborted generation; sample is incomplete")
    sequence: dict[str, Any] = {
        "stop_reason": "length" if finish == "length" else "stop",
        "tokens": tokens,
    }
    if logprobs:
        sequence["logprobs"] = logprobs
    return sequence


def _token(item: Any) -> int:
    if isinstance(item, dict):
        return int(item["token_id"])
    return int(item[1])


def _logprob(item: Any) -> float:
    if isinstance(item, dict):
        return float(item["logprob"])
    return float(item[0])


def _logprobs(items: Any) -> list[float | None]:
    if not items:
        return []
    values = []
    for item in items:
        value = (
            None
            if item is None
            else item["logprob"]
            if isinstance(item, dict)
            else item[0]
        )
        values.append(None if value is None else float(value))
    return values


def _topk_logprobs(items: Any) -> list[list[tuple[int, float]] | None]:
    if not items:
        return []
    return [
        None
        if item is None
        else [(_token(candidate), _logprob(candidate)) for candidate in item]
        for item in items
    ]
