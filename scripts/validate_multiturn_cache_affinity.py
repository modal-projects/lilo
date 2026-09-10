from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from typing import Any

import httpx


async def _post(
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
) -> httpx.Response:
    response = await client.post(path, json=payload)
    if response.status_code >= 400 and response.status_code != 408:
        raise RuntimeError(
            f"{path} returned {response.status_code}: {response.text[:500]}"
        )
    return response


async def _retrieve(
    client: httpx.AsyncClient,
    request_id: str,
    timeout: float,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        response = await _post(
            client,
            "/api/v1/retrieve_future",
            {"request_id": request_id},
        )
        if response.status_code != 408:
            result = response.json()
            if not isinstance(result, dict):
                raise RuntimeError("retrieve_future returned a non-object response")
            return result
        if loop.time() >= deadline:
            raise TimeoutError(f"sample {request_id} did not complete within {timeout}s")
        await asyncio.sleep(1)


async def validate(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"x-api-key": args.api_key}
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=args.timeout,
    ) as client:
        created = await _post(client, "/api/v1/create_session", {})
        session_id = created.json()["session_id"]
        sampler = await _post(
            client,
            "/api/v1/create_sampling_session",
            {
                "session_id": session_id,
                "sampling_session_seq_id": 0,
                "base_model": args.base_model,
            },
        )
        sampling_session_id = sampler.json()["sampling_session_id"]

        prompt = [
            args.initial_token + (index % args.token_span)
            for index in range(args.initial_prompt_tokens)
        ]
        turns = []
        total_hit_tokens = 0
        total_prompt_tokens = 0
        eligible_hit_tokens = 0
        eligible_prefix_tokens = 0

        for turn in range(args.turns):
            expected_reusable = 0 if turn == 0 else len(prompt) - 1
            submitted = await _post(
                client,
                "/api/v1/asample",
                {
                    "type": "sample",
                    "sampling_session_id": sampling_session_id,
                    "seq_id": turn,
                    "cache_affinity_key": args.affinity_key,
                    "num_samples": 1,
                    "prompt": {
                        "chunks": [
                            {
                                "type": "encoded_text",
                                "tokens": prompt,
                            }
                        ]
                    },
                    "sampling_params": {
                        "max_tokens": args.output_tokens,
                        "temperature": 0.0,
                        "seed": args.seed + turn,
                    },
                },
            )
            result = await _retrieve(
                client,
                submitted.json()["request_id"],
                args.timeout,
            )
            sequences = result.get("sequences") or []
            if len(sequences) != 1:
                raise RuntimeError(f"turn {turn} returned {len(sequences)} sequences")
            output_tokens = list(sequences[0].get("tokens") or ())
            if not output_tokens:
                raise RuntimeError(f"turn {turn} returned no tokens")

            cached_tokens = int(result.get("prompt_cache_hit_tokens") or 0)
            prompt_tokens = len(prompt)
            turn_result = {
                "turn": turn,
                "prompt_tokens": prompt_tokens,
                "expected_reusable_tokens": expected_reusable,
                "prompt_cache_hit_tokens": cached_tokens,
                "input_cache_hit_rate": cached_tokens / prompt_tokens,
                "output_tokens": len(output_tokens),
            }
            turns.append(turn_result)
            print(json.dumps(turn_result, sort_keys=True))

            total_hit_tokens += cached_tokens
            total_prompt_tokens += prompt_tokens
            if expected_reusable:
                eligible_hit_tokens += min(cached_tokens, expected_reusable)
                eligible_prefix_tokens += expected_reusable

            prompt.extend(output_tokens)
            prompt.append(args.turn_separator + turn)

    eligible_prefix_hit_rate = (
        eligible_hit_tokens / eligible_prefix_tokens if eligible_prefix_tokens else 0.0
    )
    summary = {
        "sampling_session_id": sampling_session_id,
        "cache_affinity_key": args.affinity_key,
        "turns": turns,
        "input_cache_hit_rate": total_hit_tokens / total_prompt_tokens,
        "eligible_prefix_hit_rate": eligible_prefix_hit_rate,
        "minimum_required_prefix_hit_rate": args.min_prefix_hit_rate,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if eligible_prefix_hit_rate < args.min_prefix_hit_rate:
        raise RuntimeError(
            "eligible prefix cache hit rate "
            f"{eligible_prefix_hit_rate:.3f} is below "
            f"{args.min_prefix_hit_rate:.3f}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send growing token-exact prompts under one cache affinity key and "
            "measure SGLang prefix-cache reuse."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TINKER_BASE_URL"),
        required="TINKER_BASE_URL" not in os.environ,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TINKER_API_KEY"),
        required="TINKER_API_KEY" not in os.environ,
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--affinity-key",
        default=f"multiturn-validation-{uuid.uuid4().hex}",
    )
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--initial-prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--initial-token", type=int, default=100)
    parser.add_argument("--token-span", type=int, default=1000)
    parser.add_argument("--turn-separator", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-prefix-hit-rate", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=3000)
    args = parser.parse_args()
    if args.turns < 2:
        parser.error("--turns must be at least 2")
    if args.initial_prompt_tokens < 1:
        parser.error("--initial-prompt-tokens must be positive")
    if args.output_tokens < 1:
        parser.error("--output-tokens must be positive")
    if args.token_span < 1:
        parser.error("--token-span must be positive")
    if not 0 <= args.min_prefix_hit_rate <= 1:
        parser.error("--min-prefix-hit-rate must be between 0 and 1")
    return args


def main() -> None:
    asyncio.run(validate(parse_args()))


if __name__ == "__main__":
    main()
