# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["tinker>=0.24,<0.25", "datasets", "wandb"]
# ///
"""Smoke-test a deployed Lilo server the way a customer would: pure Tinker SDK.

Runs a short LongRLVR GRPO loop against an already-deployed endpoint and
exercises the whole client path end to end: LoRA client creation, sampling at
the definition's context length, `forward_backward`, `optim_step`, `save_state`,
and resuming that checkpoint on a fresh client for one more step. Prompts are
padded with distractor documents so the run actually reaches the long-context
regime the definition was built for.

    export TINKER_BASE_URL=https://<deployment>.modal.run
    export TINKER_API_KEY=...
    uv run scripts/deployment_smoke_longrlvr.py \\
        --base-model qwen3_8_27b_miles_lora_64k \\
        --target-prompt-tokens 48000 --steps 2

Add `--wandb-project <project>` to log rewards, step times and backend metrics.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import tinker
import wandb
from datasets import load_dataset
from tinker import TrainingClient, types

DATASET_NAME = "Guanzheng/LongRLVR-Data"
TIMEOUT = 3 * 60 * 60

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
CHUNKS = re.compile(r"<useful_chunks>(.*?)</useful_chunks>", re.DOTALL | re.IGNORECASE)
CHUNK_ID = re.compile(r"<CHUNK_(\d+)>", re.IGNORECASE)
WORD = re.compile(r"\w+", re.UNICODE)
QUESTION_MARKER = "\n\nQuestion:"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def token_f1(candidate: str, reference: str) -> float:
    predicted = Counter(WORD.findall(candidate.casefold()))
    expected = Counter(WORD.findall(reference.casefold()))
    overlap = sum((predicted & expected).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def score(response: str, ground_truth: str, reference_chunks: Sequence[int]) -> float:
    """LongRLVR reward with a lexical stand-in for the LLM answer judge."""
    answers, chunk_sections = ANSWER.findall(response), CHUNKS.findall(response)
    if len(answers) != 1 or len(chunk_sections) != 1:
        return 0.0
    predicted = {int(chunk) for chunk in CHUNK_ID.findall(chunk_sections[0])}
    reference = set(reference_chunks)
    overlap = len(predicted & reference)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference) if reference else 0.0
    denominator = 4 * precision + recall
    chunk_f2 = 5 * precision * recall / denominator if denominator else 0.0
    answer_f1 = token_f1(answers[0].strip(), ground_truth)
    return answer_f1 + 0.1 * chunk_f2 + 0.9 * answer_f1 * chunk_f2


def load_rows(count: int, seed: int) -> list[dict[str, Any]]:
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True).shuffle(
        seed=seed, buffer_size=max(256, 4 * count)
    )
    rows: list[dict[str, Any]] = []
    for row in dataset:
        reward_model = row.get("reward_model") or {}
        extra_info = row.get("extra_info") or {}
        ground_truth = str(reward_model.get("ground_truth") or "")
        reference_chunks = extra_info.get("ref_chunks") or []
        if not isinstance(row.get("prompt"), list) or not ground_truth:
            continue
        if not reference_chunks:
            continue
        rows.append(
            {
                "prompt": [dict(message) for message in row["prompt"]],
                "ground_truth": ground_truth,
                "reference_chunks": [int(chunk) for chunk in reference_chunks],
            }
        )
        if len(rows) == count:
            return rows
    raise RuntimeError(f"found only {len(rows)} usable rows; expected {count}")


def pad_prompt(
    rows: Sequence[dict[str, Any]], index: int, tokenizer: Any, target_tokens: int
) -> list[dict[str, str]]:
    """Grow one prompt toward `target_tokens` with documents from other rows."""
    messages = [dict(message) for message in rows[index]["prompt"]]
    head, separator, tail = messages[-1]["content"].partition(QUESTION_MARKER)
    total = len(tokenizer.encode(messages[-1]["content"]))
    distractors: list[str] = []
    offset = 1
    while total < target_tokens and offset < len(rows):
        donor = rows[(index + offset) % len(rows)]["prompt"][-1]["content"]
        documents = donor.partition(QUESTION_MARKER)[0]
        distractors.append(documents)
        total += len(tokenizer.encode(documents))
        offset += 1
    messages[-1]["content"] = head + "\n" + "\n".join(distractors) + separator + tail
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="qwen3_8_27b_miles_lora_64k")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=65_536)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--target-prompt-tokens", type=int, default=48_000)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--skip-resume",
        action="store_true",
        help="Skip the save_state/load_state leg at the end of the run.",
    )
    return parser.parse_args()


def train_step(
    training: TrainingClient,
    prompts: Sequence[tuple[list[int], dict[str, Any]]],
    tokenizer: Any,
    args: argparse.Namespace,
    label: str,
) -> tuple[float, dict[str, Any], dict[str, Any], float]:
    """Sample a GRPO batch, run forward_backward + optim_step, return metrics."""
    started = time.time()
    log(f"{label}: saving weights for the sampler")
    sampling = training.save_weights_and_get_sampling_client()
    futures = [
        sampling.sample(
            prompt=types.ModelInput.from_ints(tokens),
            num_samples=args.group_size,
            sampling_params=types.SamplingParams(
                max_tokens=args.max_tokens, temperature=1.0, top_p=1.0
            ),
        )
        for tokens, _ in prompts
    ]
    batch: list[types.Datum] = []
    rewards: list[float] = []
    for (tokens, row), future in zip(prompts, futures):
        sequences = []
        group_rewards = []
        for sequence in future.result(timeout=TIMEOUT).sequences:
            response = list(sequence.tokens)
            sequences.append((response, list(sequence.logprobs or ())))
            group_rewards.append(
                score(
                    tokenizer.decode(response),
                    str(row["ground_truth"]),
                    row["reference_chunks"],
                )
            )
        mean = sum(group_rewards) / len(group_rewards)
        variance = sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)
        spread = variance**0.5
        for (response, logprobs), reward in zip(sequences, group_rewards):
            advantage = (reward - mean) / (spread + 1e-6)
            prefix = len(tokens) - 1
            batch.append(
                types.Datum(
                    # target_tokens must be model_input shifted by one, so
                    # the prompt is masked through advantages, not targets.
                    model_input=types.ModelInput.from_ints(tokens + response[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tokens[1:] + response,
                        "logprobs": [0.0] * prefix + logprobs,
                        "advantages": [0.0] * prefix + [advantage] * len(response),
                    },
                )
            )
        rewards.extend(group_rewards)

    log(f"{label}: forward_backward on {len(batch)} sequences")
    forward_backward = training.forward_backward(batch, "importance_sampling")
    optim = training.optim_step(types.AdamParams(learning_rate=args.learning_rate))
    forward_backward_result = forward_backward.result(timeout=TIMEOUT)
    optim_result = optim.result(timeout=TIMEOUT)
    elapsed = time.time() - started
    reward_mean = sum(rewards) / len(rewards)
    log(f"{label}: reward_mean={reward_mean:.4f} step_time={elapsed:.0f}s")
    log(f"{label}: forward_backward {forward_backward_result.metrics}")
    log(f"{label}: optim {optim_result.metrics}")
    return reward_mean, forward_backward_result.metrics, optim_result.metrics, elapsed


def log_wandb(
    step: int,
    reward_mean: float,
    forward_backward_metrics: dict[str, Any],
    optim_metrics: dict[str, Any],
    elapsed: float,
) -> None:
    wandb.log(
        {
            "reward/mean": reward_mean,
            "time/step_s": elapsed,
            **{
                f"fb/{key}": value
                for key, value in forward_backward_metrics.items()
                if isinstance(value, (int, float))
            },
            **{
                f"optim/{key}": value
                for key, value in optim_metrics.items()
                if isinstance(value, (int, float))
            },
        },
        step=step,
    )


def create_client(service: tinker.ServiceClient, args: argparse.Namespace):
    # Miles applies LoRA deployment-wide to attention and MLP; the SDK defaults
    # to train_unembed=True, which the backend rejects.
    return service.create_lora_training_client(
        base_model=args.base_model, rank=args.rank, train_unembed=False
    )


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"lilo-deploy-smoke-{int(time.time())}"
    run = None
    if args.wandb_project:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=run_name,
            config=vars(args),
        )
        log(f"wandb: {run.url}")

    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
    )
    log(f"creating LoRA training client rank={args.rank} on {args.base_model}")
    started = time.time()
    training = create_client(service, args)
    log(f"training client ready in {time.time() - started:.0f}s")
    tokenizer = training.get_tokenizer()

    rows = load_rows(max(args.groups * 4, 32), args.seed)
    budget = args.context_length - args.max_tokens
    prompts: list[tuple[list[int], dict[str, Any]]] = []
    for index in range(args.groups):
        messages = pad_prompt(rows, index, tokenizer, args.target_prompt_tokens)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) > budget:
            raise RuntimeError(f"prompt {index} exceeds the {budget}-token budget")
        prompts.append((tokens, rows[index]))
        log(f"prompt {index}: {len(tokens)} tokens")

    for step in range(args.steps):
        results = train_step(training, prompts, tokenizer, args, f"step {step}")
        if run is not None:
            log_wandb(step, *results)

    log("save_state")
    state = training.save_state(name=run_name).result(timeout=TIMEOUT)
    log(f"checkpoint: {state.path}")
    if run is not None:
        run.summary["checkpoint"] = state.path

    if not args.skip_resume:
        # A second client resuming the checkpoint is what a customer does after
        # a preemption, so the smoke test covers it rather than trusting the
        # save alone.
        log(f"resuming {state.path} on a fresh training client")
        started = time.time()
        resumed = create_client(service, args)
        resumed.load_state_with_optimizer(state.path).result(timeout=TIMEOUT)
        log(f"resumed in {time.time() - started:.0f}s")
        results = train_step(resumed, prompts, tokenizer, args, "resumed step")
        if run is not None:
            log_wandb(args.steps, *results)
            run.summary["resumed"] = True

    if run is not None:
        run.finish()
    log("smoke test passed")


if __name__ == "__main__":
    main()
