from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import httpx
import modal
import tinker
from tinker import types

from lilo.providers.modal.app import app, cleaner, module_for, server

DEFAULT_DEFINITION = "qwen3_5_4b_full_64k"
TIMEOUT = 3 * 60 * 60


def _definition(definition_id: str) -> tuple[Any, str]:
    module = module_for(definition_id)
    if not module.CATALOG_VISIBLE:
        raise ValueError(f"definition is not cataloged: {definition_id}")
    return module, module.PARAMETERIZATION


def _timestamped(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.stem}.{stamp}{path.suffix}")


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _datum(length: int, vocab_size: int, seed: int) -> types.Datum:
    rng = random.Random(seed)
    tokens = [rng.randrange(vocab_size) for _ in range(length + 1)]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * length,
        },
    )


def _sft_datum(tokenizer) -> tuple[types.Datum, int, int]:
    prompt = tokenizer.encode(
        "Question: What is one plus one?\nAnswer:",
        add_special_tokens=True,
    )
    completion = tokenizer.encode(" 2", add_special_tokens=False)
    tokens = prompt + completion
    return (
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tokens[1:],
                "weights": [0.0] * (len(prompt) - 1) + [1.0] * len(completion),
            },
        ),
        len(tokens) - 1,
        len(completion),
    )


def _rl_datum(
    prompt: list[int],
    response: list[int],
    logprobs: list[float],
) -> types.Datum:
    prefix = len(prompt) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(prompt + response[:-1]),
        loss_fn_inputs={
            "target_tokens": [0] * prefix + response,
            "logprobs": [0.0] * prefix + logprobs,
            "advantages": [0.0] * prefix + [1.0] * len(response),
        },
    )


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float)
    }
    if not numeric or any(not math.isfinite(value) for value in numeric.values()):
        raise RuntimeError(f"non-finite metrics: {metrics}")
    return numeric


def _validate_forward(
    result,
    lengths: list[int],
    expected_tokens: int | None = None,
) -> dict[str, float]:
    metrics = _finite_metrics(result.metrics)
    outputs = result.loss_fn_outputs
    if len(outputs) != len(lengths):
        raise RuntimeError(f"expected {len(lengths)} outputs, got {len(outputs)}")
    actual = [len(output["logprobs"].data) for output in outputs]
    if actual != lengths:
        raise RuntimeError(f"expected output lengths {lengths}, got {actual}")
    token_count = float(sum(lengths) if expected_tokens is None else expected_tokens)
    if metrics.get("tokens:sum") != token_count:
        raise RuntimeError(
            f"expected tokens:sum={token_count}, got {metrics.get('tokens:sum')}"
        )
    return metrics


def _forward_step(
    training,
    data: list[types.Datum],
    lengths: list[int],
    expected_tokens: int | None = None,
) -> dict:
    started = time.perf_counter()
    forward = training.forward_backward(data, "cross_entropy")
    optimizer = training.optim_step(types.AdamParams(learning_rate=1e-4))
    forward_result = forward.result(timeout=TIMEOUT)
    forward_finished = time.perf_counter()
    optimizer_result = optimizer.result(timeout=TIMEOUT)
    finished = time.perf_counter()
    return {
        "lengths": lengths,
        "metrics": _validate_forward(forward_result, lengths, expected_tokens),
        "optimizer_metrics": _finite_metrics(optimizer_result.metrics),
        "forward_backward_seconds": forward_finished - started,
        "optimizer_seconds": finished - forward_finished,
        "step_seconds": finished - started,
    }


def _forward_logprobs(
    training,
    datum: types.Datum,
    expected_tokens: int,
) -> list[float]:
    result = training.forward([datum], "cross_entropy").result(timeout=TIMEOUT)
    _validate_forward(
        result,
        [len(datum.model_input.to_ints())],
        expected_tokens,
    )
    return [float(value) for value in result.loss_fn_outputs[0]["logprobs"].data]


def _max_error(actual: list[float], expected: list[float]) -> float:
    return max(
        (
            abs(value - reference)
            for value, reference in zip(actual, expected, strict=True)
        ),
        default=0.0,
    )


def _checkpoint_roundtrip(
    training,
    tokenizer,
    *,
    service,
    module,
    parameterization: str,
    base_url: str,
    api_key: str,
) -> tuple[dict, Any]:
    datum, length, trained_tokens = _sft_datum(tokenizer)
    warmup = [
        _forward_step(training, [datum], [length], trained_tokens) for _ in range(2)
    ]
    before = _forward_logprobs(training, datum, trained_tokens)
    started = time.perf_counter()
    saved = training.save_state(
        f"e2e-{uuid.uuid4().hex[:12]}",
        overwrite=True,
    ).result(timeout=TIMEOUT)
    save_finished = time.perf_counter()

    reference_step = _forward_step(training, [datum], [length], trained_tokens)
    reference = _forward_logprobs(training, datum, trained_tokens)
    if reference == before:
        raise RuntimeError("optimizer step did not change checkpoint test output")

    original_model_id = str(training.model_id)
    _unload(base_url, api_key, original_model_id)
    resumed = None
    load_started = time.perf_counter()
    try:
        resumed = _create_training(service, module, parameterization)
        resumed.load_state_with_optimizer(saved.path).result(timeout=TIMEOUT)
        restored = _forward_logprobs(resumed, datum, trained_tokens)
        load_finished = time.perf_counter()
        restore_error = _max_error(restored, before)
        if restore_error > 1e-5:
            raise RuntimeError(
                f"checkpoint restore max logprob error: {restore_error}"
            )

        resumed_step = _forward_step(resumed, [datum], [length], trained_tokens)
        continued = _forward_logprobs(resumed, datum, trained_tokens)
        continuation_error = _max_error(continued, reference)
        if continuation_error != 0:
            raise RuntimeError(
                "fresh-engine optimizer continuation differs from reference: "
                f"max logprob error {continuation_error}"
            )
        return (
            {
                "path": saved.path,
                "source_model_id": original_model_id,
                "resumed_model_id": str(resumed.model_id),
                "warmup": warmup,
                "reference_step": reference_step,
                "resumed_step": resumed_step,
                "save_seconds": save_finished - started,
                "load_seconds": load_finished - load_started,
                "restore_max_logprob_error": restore_error,
                "continuation_max_logprob_error": continuation_error,
            },
            resumed,
        )
    except BaseException:
        if resumed is not None:
            _unload(base_url, api_key, str(resumed.model_id))
        raise


def _hf_checkpoint_roundtrip(training, tokenizer) -> dict:
    datum, length, trained_tokens = _sft_datum(tokenizer)
    warmup = _forward_step(training, [datum], [length], trained_tokens)
    expected = _forward_logprobs(training, datum, trained_tokens)
    save_started = time.perf_counter()
    saved = training.save_state(
        f"hf-roundtrip-{uuid.uuid4().hex[:12]}",
        overwrite=True,
    ).result(timeout=TIMEOUT)
    save_seconds = time.perf_counter() - save_started

    mutation = _forward_step(training, [datum], [length], trained_tokens)
    mutated = _forward_logprobs(training, datum, trained_tokens)
    mutation_error = _max_error(mutated, expected)
    if mutation_error == 0:
        raise RuntimeError("training did not change HF roundtrip output")

    load_started = time.perf_counter()
    training.load_state(saved.path).result(timeout=TIMEOUT)
    restored = _forward_logprobs(training, datum, trained_tokens)
    load_seconds = time.perf_counter() - load_started
    restore_error = _max_error(restored, expected)
    if restore_error != 0:
        raise RuntimeError(f"HF restore max logprob error: {restore_error}")

    return {
        "checkpoint_path": saved.path,
        "warmup": warmup,
        "mutation": mutation,
        "mutation_max_logprob_error": mutation_error,
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "restore_max_logprob_error": restore_error,
    }


def _sample(sampling, prompt: list[int], max_tokens: int, seed: int) -> dict:
    started = time.perf_counter()
    result = sampling.sample(
        prompt=types.ModelInput.from_ints(prompt),
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
        ),
    ).result(timeout=TIMEOUT)
    elapsed = time.perf_counter() - started
    if len(result.sequences) != 1:
        raise RuntimeError(
            f"expected one sampled sequence, got {len(result.sequences)}"
        )
    sequence = result.sequences[0]
    tokens = list(sequence.tokens)
    logprobs = list(sequence.logprobs or ())
    if not tokens or len(logprobs) != len(tokens):
        raise RuntimeError("sample did not return matching tokens and logprobs")
    return {
        "prompt_tokens": len(prompt),
        "output_tokens": len(tokens),
        "tokens": tokens,
        "logprobs": logprobs,
        "seconds": elapsed,
    }


def _restored_sample_error(
    actual: dict,
    expected: dict,
    *,
    label: str,
) -> float:
    if actual["tokens"] != expected["tokens"]:
        raise RuntimeError(f"{label} produced different tokens after restore")
    error = _max_error(actual["logprobs"], expected["logprobs"])
    if error > 1e-4:
        raise RuntimeError(f"{label} sampler logprob error: {error}")
    return error


def _sampler_recovery_roundtrip(training, tokenizer) -> dict:
    datum, length, trained_tokens = _sft_datum(tokenizer)
    warmup = _forward_step(training, [datum], [length], trained_tokens)
    checkpoint_logprobs = _forward_logprobs(training, datum, trained_tokens)
    prompt = tokenizer.encode(
        "Question: What is two plus two?\nAnswer:",
        add_special_tokens=True,
    )

    publish_started = time.perf_counter()
    baseline_client = training.save_weights_and_get_sampling_client()
    baseline = _sample(baseline_client, prompt, 8, 1)
    baseline_publish_seconds = time.perf_counter() - publish_started

    save_started = time.perf_counter()
    saved = training.save_state(
        f"sampler-recovery-{uuid.uuid4().hex[:12]}",
        overwrite=True,
    ).result(timeout=TIMEOUT)
    save_seconds = time.perf_counter() - save_started

    mutation = _forward_step(training, [datum], [length], trained_tokens)
    mutated_logprobs = _forward_logprobs(training, datum, trained_tokens)
    trainer_mutation_error = _max_error(mutated_logprobs, checkpoint_logprobs)
    if trainer_mutation_error == 0:
        raise RuntimeError("training did not change checkpoint test output")
    mutated_client = training.save_weights_and_get_sampling_client()
    mutated = _sample(mutated_client, prompt, 8, 1)
    sampler_changed = (
        mutated["tokens"] != baseline["tokens"]
        or _max_error(mutated["logprobs"], baseline["logprobs"]) > 1e-4
    )
    if not sampler_changed:
        raise RuntimeError("mutated sampler output did not change")

    weights_load_started = time.perf_counter()
    training.load_state(saved.path).result(timeout=TIMEOUT)
    weights_load_seconds = time.perf_counter() - weights_load_started
    weights_restored = _forward_logprobs(training, datum, trained_tokens)
    weights_restore_error = _max_error(weights_restored, checkpoint_logprobs)
    if weights_restore_error > 1e-5:
        raise RuntimeError(
            f"weights-only restore max logprob error: {weights_restore_error}"
        )
    weights_publish_started = time.perf_counter()
    weights_client = training.save_weights_and_get_sampling_client()
    weights_sample = _sample(weights_client, prompt, 8, 1)
    weights_publish_seconds = time.perf_counter() - weights_publish_started
    weights_sampler_error = _restored_sample_error(
        weights_sample,
        baseline,
        label="weights-only restore",
    )

    optimizer_load_started = time.perf_counter()
    training.load_state_with_optimizer(saved.path).result(timeout=TIMEOUT)
    optimizer_load_seconds = time.perf_counter() - optimizer_load_started
    optimizer_restored = _forward_logprobs(training, datum, trained_tokens)
    optimizer_restore_error = _max_error(optimizer_restored, checkpoint_logprobs)
    if optimizer_restore_error > 1e-5:
        raise RuntimeError(
            f"optimizer restore max logprob error: {optimizer_restore_error}"
        )
    optimizer_publish_started = time.perf_counter()
    optimizer_client = training.save_weights_and_get_sampling_client()
    optimizer_sample = _sample(optimizer_client, prompt, 8, 1)
    optimizer_publish_seconds = time.perf_counter() - optimizer_publish_started
    optimizer_sampler_error = _restored_sample_error(
        optimizer_sample,
        baseline,
        label="optimizer restore",
    )

    return {
        "checkpoint_path": saved.path,
        "warmup": warmup,
        "mutation": mutation,
        "trainer_mutation_max_logprob_error": trainer_mutation_error,
        "baseline_publish_seconds": baseline_publish_seconds,
        "checkpoint_save_seconds": save_seconds,
        "weights_only": {
            "load_seconds": weights_load_seconds,
            "publish_seconds": weights_publish_seconds,
            "trainer_max_logprob_error": weights_restore_error,
            "sampler_max_logprob_error": weights_sampler_error,
        },
        "with_optimizer": {
            "load_seconds": optimizer_load_seconds,
            "publish_seconds": optimizer_publish_seconds,
            "trainer_max_logprob_error": optimizer_restore_error,
            "sampler_max_logprob_error": optimizer_sampler_error,
        },
    }


def _create_training(service, module, parameterization: str):
    if parameterization == "full":
        from lilo.client import create_full_training_client

        return create_full_training_client(service, module.MODEL_NAME)
    return service.create_lora_training_client(
        base_model=module.MODEL_NAME,
        rank=module.LORA_RANK,
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
                "/api/v1/retrieve_future",
                json={"request_id": request_id},
            )
            if response.status_code != 408:
                response.raise_for_status()
                return
            time.sleep(1)


def _warm(
    training,
    tokenizer,
    definition_id: str,
) -> tuple[Any, dict]:
    train = _forward_step(
        training,
        [_datum(128, tokenizer.vocab_size, 1)],
        [128],
    )
    export_started = time.perf_counter()
    sampling = training.save_weights_and_get_sampling_client()
    prompt = tokenizer.encode(
        "Question: What is two plus two?\nAnswer:",
        add_special_tokens=True,
    )
    rollout = _sample(sampling, prompt, 8, 1)
    return sampling, {
        "definition_id": definition_id,
        "trainer": train,
        "sampler_ready_seconds": time.perf_counter() - export_started,
        "rollout": rollout,
    }


def _correctness(
    training,
    sampling,
    tokenizer,
    *,
    context_length: int,
    packed_capacity: int,
    skip_max_context: bool,
) -> dict:
    packed_lengths = [
        32,
        max(64, packed_capacity // 8),
        max(128, packed_capacity // 4),
    ]
    near_capacity_lengths = [
        packed_capacity // divisor for divisor in (2, 4, 8, 16, 32, 64)
    ]
    multiple_lengths = [
        max(256, packed_capacity * 3 // 4),
        max(256, packed_capacity // 2),
        97,
    ]
    cases = {
        "packed_together": _forward_step(
            training,
            [
                _datum(length, tokenizer.vocab_size, 100 + index)
                for index, length in enumerate(packed_lengths)
            ],
            packed_lengths,
        ),
        "near_capacity_packed": _forward_step(
            training,
            [
                _datum(length, tokenizer.vocab_size, 150 + index)
                for index, length in enumerate(near_capacity_lengths)
            ],
            near_capacity_lengths,
        ),
        "multiple_packed_bins": _forward_step(
            training,
            [
                _datum(length, tokenizer.vocab_size, 200 + index)
                for index, length in enumerate(multiple_lengths)
            ],
            multiple_lengths,
        ),
    }
    if not skip_max_context:
        cases["max_context"] = _forward_step(
            training,
            [_datum(context_length, tokenizer.vocab_size, 300)],
            [context_length],
        )

    oversized = _datum(context_length + 1, tokenizer.vocab_size, 301)
    try:
        training.forward_backward([oversized], "cross_entropy").result(timeout=TIMEOUT)
    except (tinker.TinkerError, RuntimeError, TimeoutError, ValueError) as exc:
        cases["over_context_rejected"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        raise RuntimeError("over-context sequence was accepted")

    sft, sft_length, sft_tokens = _sft_datum(tokenizer)
    before = training.forward([sft], "cross_entropy").result(timeout=TIMEOUT)
    before_metrics = _validate_forward(
        before,
        [sft_length],
        sft_tokens,
    )
    steps = [
        _forward_step(
            training,
            [sft, sft, sft, sft],
            [sft_length] * 4,
            sft_tokens * 4,
        )
        for _ in range(3)
    ]
    after = training.forward([sft], "cross_entropy").result(timeout=TIMEOUT)
    after_metrics = _validate_forward(
        after,
        [sft_length],
        sft_tokens,
    )
    cases["sft"] = {
        "before": before_metrics,
        "steps": steps,
        "after": after_metrics,
    }

    prompts = [
        tokenizer.encode(text, add_special_tokens=True)
        for text in (
            "The capital of France is",
            "Count from one to five:",
            "Write one short sentence about the moon.",
        )
    ]
    rollout = [
        _sample(sampling, prompt, max_tokens, 400 + index)
        for index, (prompt, max_tokens) in enumerate(
            zip(prompts, (8, 16, 32), strict=True)
        )
    ]
    cases["rollout"] = rollout

    source = rollout[0]
    rl = _rl_datum(prompts[0], source["tokens"], source["logprobs"])
    started = time.perf_counter()
    forward = training.forward_backward([rl], "importance_sampling")
    optimizer = training.optim_step(types.AdamParams(learning_rate=1e-5))
    forward_result = forward.result(timeout=TIMEOUT)
    forward_finished = time.perf_counter()
    optimizer_result = optimizer.result(timeout=TIMEOUT)
    cases["rollout_to_rl"] = {
        "metrics": _finite_metrics(forward_result.metrics),
        "optimizer_metrics": _finite_metrics(optimizer_result.metrics),
        "forward_backward_seconds": forward_finished - started,
        "step_seconds": time.perf_counter() - started,
    }
    return cases


def _run(args: argparse.Namespace, base_url: str, output: Path) -> dict:
    module, parameterization = _definition(args.definition_id)
    api_key = os.environ["TINKER_API_KEY"]
    context_length = int(module.MAX_CONTEXT_LENGTH)
    packed_capacity = int(module.MAX_TOKENS_PER_MICROBATCH)
    report: dict[str, Any] = {
        "status": "running",
        "definition": {
            "definition_id": args.definition_id,
            "base_model": module.MODEL_NAME,
            "parameterization": parameterization,
            "context_length": context_length,
            "packed_capacity": packed_capacity,
            "micro_batch_size": module.MICRO_BATCH_SIZE,
            "gpu_type": module.GPU_TYPE,
            "gpus": module.GPUS,
        },
        "base_url": base_url,
        "started_at": time.time(),
        "phases": {},
    }
    _write(output, report)
    training = None
    try:
        service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
        provisioned_at = time.perf_counter()
        training = _create_training(service, module, parameterization)
        info = training.get_info()
        if bool(info.is_lora) != (parameterization == "lora"):
            raise RuntimeError(f"unexpected parameterization: {info}")
        tokenizer = training.get_tokenizer()
        report["phases"]["provision"] = {
            "seconds": time.perf_counter() - provisioned_at,
            "model_id": training.model_id,
            "is_lora": info.is_lora,
        }
        _write(output, report)
        if args.hf_roundtrip_only:
            if parameterization != "full":
                raise ValueError("--hf-roundtrip-only requires an FFT definition")
            report["phases"]["hf_roundtrip"] = _hf_checkpoint_roundtrip(
                training,
                tokenizer,
            )
            report["status"] = "passed"
            return report
        if args.sampler_recovery_only:
            if parameterization != "full":
                raise ValueError("--sampler-recovery-only requires an FFT definition")
            report["phases"]["sampler_recovery"] = _sampler_recovery_roundtrip(
                training,
                tokenizer,
            )
            report["status"] = "passed"
            return report
        if args.checkpoint_only:
            if parameterization != "full":
                raise ValueError("--checkpoint-only requires an FFT definition")
            checkpoint, training = _checkpoint_roundtrip(
                training,
                tokenizer,
                service=service,
                module=module,
                parameterization=parameterization,
                base_url=base_url,
                api_key=api_key,
            )
            report["phases"]["checkpoint"] = checkpoint
            report["status"] = "passed"
            return report

        sampling, warmup = _warm(
            training,
            tokenizer,
            args.definition_id,
        )
        report["phases"]["warmup"] = warmup
        _write(output, report)

        report["phases"]["correctness"] = _correctness(
            training,
            sampling,
            tokenizer,
            context_length=context_length,
            packed_capacity=packed_capacity,
            skip_max_context=args.skip_max_context,
        )
        _write(output, report)

        report["status"] = "passed"
        return report
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = time.time()
        if training is not None:
            try:
                _unload(base_url, api_key, str(training.model_id))
                report["cleanup"] = {"model_unloaded": True}
            except (httpx.HTTPError, RuntimeError, TimeoutError, ValueError) as exc:
                report["cleanup"] = {
                    "model_unloaded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        _write(output, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition-id", default=DEFAULT_DEFINITION)
    parser.add_argument("--base-url")
    parser.add_argument("--skip-max-context", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--hf-roundtrip-only", action="store_true")
    parser.add_argument("--sampler-recovery-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/results/engine_definition_e2e.json"),
    )
    args = parser.parse_args()
    if sum(
        (
            args.checkpoint_only,
            args.hf_roundtrip_only,
            args.sampler_recovery_only,
        )
    ) > 1:
        parser.error(
            "--checkpoint-only, --hf-roundtrip-only, and "
            "--sampler-recovery-only are exclusive"
        )

    output = _timestamped(args.output)
    context = nullcontext(args.base_url)
    if args.base_url is None:
        context = app.run(name=f"tinker-e2e-{uuid.uuid4().hex[:12]}")
    try:
        with modal.enable_output(), context:
            base_url = args.base_url or server.get_web_url()
            if not base_url:
                raise RuntimeError("Modal did not provide a control-plane URL")
            report = _run(args, base_url, output)
            if args.base_url is None:
                cleaner.remote()
    finally:
        print(output)
    summary = {
        "status": report["status"],
        "definition": report["definition"]["definition_id"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
