"""Smoke test multi-LoRA engine definitions on a deployed Lilo control plane.

For each definition this creates a LoRA training client, runs a cross-entropy
step, publishes the adapter and samples from it, then runs an
importance-sampling step on the sampled response and samples once more. The
model is unloaded afterwards so the trainer slot is released.

Usage::

    export TINKER_BASE_URL=https://...modal.run
    export TINKER_API_KEY=tml-lilo-...
    uv run scripts/lora_smoke.py \
        --definition-id qwen3_5_9b_miles_lora_16k \
        --definition-id qwen3_8_27b_miles_lora_64k

The definition id is passed as ``base_model`` so non-cataloged definitions can
be targeted directly. Definitions run sequentially unless ``--parallel`` is set.

Miles LoRA deployments fix their target modules (attention + MLP) at deploy
time and reject models whose ``train_unembed`` does not match, so the client is
created with ``train_unembed=False`` unless ``--train-unembed`` is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import tinker
from tinker import types

DEFAULT_DEFINITIONS = (
    "qwen3_5_9b_miles_lora_16k",
    "qwen3_8_27b_miles_lora_64k",
)
TIMEOUT = 60 * 60
PROMPT = "Question: What is two plus two?\nAnswer:"


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float)
    }
    if not numeric or any(not math.isfinite(value) for value in numeric.values()):
        raise RuntimeError(f"non-finite metrics: {metrics}")
    return numeric


def _sft_datum(tokenizer) -> types.Datum:
    prompt = tokenizer.encode(PROMPT, add_special_tokens=True)
    completion = tokenizer.encode(" 4", add_special_tokens=False)
    tokens = prompt + completion
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [0.0] * (len(prompt) - 1) + [1.0] * len(completion),
        },
    )


def _rl_datum(
    prompt: list[int],
    response: list[int],
    logprobs: list[float],
    reward: float,
) -> types.Datum:
    prompt_targets = len(prompt) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(prompt + response[:-1]),
        loss_fn_inputs={
            "target_tokens": prompt[1:] + response,
            "logprobs": [0.0] * prompt_targets + logprobs,
            "advantages": [0.0] * prompt_targets + [reward] * len(response),
        },
    )


def _step(training, data: list[types.Datum], loss_fn: str, lr: float) -> dict:
    started = time.perf_counter()
    forward = training.forward_backward(data, loss_fn)
    optimizer = training.optim_step(types.AdamParams(learning_rate=lr))
    forward_result = forward.result(timeout=TIMEOUT)
    forward_done = time.perf_counter()
    optimizer_result = optimizer.result(timeout=TIMEOUT)
    finished = time.perf_counter()
    if len(forward_result.loss_fn_outputs) != len(data):
        raise RuntimeError(
            f"expected {len(data)} loss outputs, got "
            f"{len(forward_result.loss_fn_outputs)}"
        )
    return {
        "loss_fn": loss_fn,
        "metrics": _finite_metrics(forward_result.metrics),
        "optimizer_metrics": _finite_metrics(optimizer_result.metrics),
        "forward_backward_seconds": forward_done - started,
        "optimizer_seconds": finished - forward_done,
    }


def _publish_and_sample(
    training, tokenizer, prompt: list[int], max_tokens: int
) -> tuple[dict, list[int], list[float]]:
    started = time.perf_counter()
    sampling = training.save_weights_and_get_sampling_client()
    published = time.perf_counter()
    result = sampling.sample(
        prompt=types.ModelInput.from_ints(prompt),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=1.0),
    ).result(timeout=TIMEOUT)
    finished = time.perf_counter()
    if len(result.sequences) != 1:
        raise RuntimeError(f"expected one sequence, got {len(result.sequences)}")
    sequence = result.sequences[0]
    tokens = list(sequence.tokens)
    logprobs = [float(value) for value in sequence.logprobs or ()]
    if not tokens or len(logprobs) != len(tokens):
        raise RuntimeError("sample did not return matching tokens and logprobs")
    if any(not math.isfinite(value) for value in logprobs):
        raise RuntimeError(f"non-finite sample logprobs: {logprobs}")
    text = tokenizer.decode(tokens)
    return (
        {
            "publish_seconds": published - started,
            "sample_seconds": finished - published,
            "output_tokens": len(tokens),
            "text": text,
        },
        tokens,
        logprobs,
    )


def _unload(base_url: str, api_key: str, model_id: str) -> None:
    headers = {"X-API-Key": api_key}
    with httpx.Client(base_url=base_url, headers=headers, timeout=60) as client:
        response = client.post("/api/v1/unload_model", json={"model_id": model_id})
        if response.status_code == 404:
            return
        response.raise_for_status()
        request_id = response.json()["request_id"]
        while True:
            response = client.post(
                "/api/v1/retrieve_future", json={"request_id": request_id}
            )
            if response.status_code != 408:
                response.raise_for_status()
                return
            time.sleep(1)


def _run_definition(
    definition_id: str,
    *,
    base_url: str,
    api_key: str,
    rank: int,
    train_unembed: bool,
    max_tokens: int,
) -> dict:
    report: dict[str, Any] = {
        "definition_id": definition_id,
        "status": "running",
        "started_at": time.time(),
        "phases": {},
    }
    training = None
    try:
        service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
        started = time.perf_counter()
        training = service.create_lora_training_client(
            base_model=definition_id, rank=rank, train_unembed=train_unembed
        )
        info = training.get_info()
        if not info.is_lora:
            raise RuntimeError(f"expected a LoRA model, got {info}")
        tokenizer = training.get_tokenizer()
        report["phases"]["provision"] = {
            "seconds": time.perf_counter() - started,
            "model_id": str(training.model_id),
            "base_model": info.model_name,
            "lora_rank": info.lora_rank,
        }
        print(json.dumps({"definition": definition_id, **report["phases"]}))

        report["phases"]["sft_step"] = _step(
            training, [_sft_datum(tokenizer)], "cross_entropy", 1e-4
        )
        print(json.dumps({"definition": definition_id, "sft_step": "ok"}))

        prompt = tokenizer.encode(PROMPT, add_special_tokens=True)
        sample, tokens, logprobs = _publish_and_sample(
            training, tokenizer, prompt, max_tokens
        )
        report["phases"]["sample_1"] = sample
        print(json.dumps({"definition": definition_id, "sample_1": sample["text"]}))

        reward = 1.0 if "4" in sample["text"] else -1.0
        report["phases"]["rl_step"] = _step(
            training,
            [_rl_datum(prompt, tokens, logprobs, reward)],
            "importance_sampling",
            1e-5,
        )
        report["phases"]["rl_step"]["reward"] = reward
        print(json.dumps({"definition": definition_id, "rl_step": "ok"}))

        sample, _, _ = _publish_and_sample(training, tokenizer, prompt, max_tokens)
        report["phases"]["sample_2"] = sample
        print(json.dumps({"definition": definition_id, "sample_2": sample["text"]}))

        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - report every failure per definition
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - report["started_at"]
        if training is not None:
            try:
                _unload(base_url, api_key, str(training.model_id))
                report["cleanup"] = {"model_unloaded": True}
            except (httpx.HTTPError, RuntimeError, TimeoutError, ValueError) as exc:
                report["cleanup"] = {
                    "model_unloaded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--definition-id",
        action="append",
        dest="definition_ids",
        help="engine definition id (repeatable); defaults to "
        + ", ".join(DEFAULT_DEFINITIONS),
    )
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--train-unembed", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/results/lora_smoke.json"),
    )
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or TINKER_BASE_URL is required")
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        parser.error("TINKER_API_KEY is required")
    definition_ids = args.definition_ids or list(DEFAULT_DEFINITIONS)

    def run(definition_id: str) -> dict:
        return _run_definition(
            definition_id,
            base_url=args.base_url,
            api_key=api_key,
            rank=args.rank,
            train_unembed=args.train_unembed,
            max_tokens=args.max_tokens,
        )

    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(definition_ids)) as pool:
            reports = list(pool.map(run, definition_ids))
    else:
        reports = [run(definition_id) for definition_id in definition_ids]

    stamp = time.strftime("%Y%m%d%H%M%S")
    output = args.output.with_name(f"{args.output.stem}.{stamp}{args.output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"base_url": args.base_url, "results": reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    for report in reports:
        line = {
            "definition": report["definition_id"],
            "status": report["status"],
            "elapsed_seconds": round(report["elapsed_seconds"], 1),
        }
        if report["status"] != "passed":
            line["error"] = report.get("error")
        print(json.dumps(line))
    print(output)
    if any(report["status"] != "passed" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
