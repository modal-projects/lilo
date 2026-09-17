# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["tinker>=0.24,<0.25", "wandb"]
# ///
"""Minimal Tinker RL loop against Lilo with per-step Weights & Biases logging.

W&B is purely client-side: the Lilo server never sees it. Each step samples one
group, scores it, runs forward_backward + optim_step through the Tinker SDK, and
logs reward, response length, the loss metrics Lilo returns, and step timing.

    export TINKER_BASE_URL=https://your-modal-server-url
    export TINKER_API_KEY=...
    export WANDB_API_KEY=...   # or `wandb login`
    uv run scripts/wandb_rl_example.py --steps 5
"""

from __future__ import annotations

import argparse
import os
import time

import tinker
import wandb
from tinker import types

TIMEOUT = 60 * 60
PROMPTS = [
    "What is 2 + 2? Answer with only the number.",
    "What is 7 * 6? Answer with only the number.",
    "What is 10 - 3? Answer with only the number.",
    "What is 9 + 8? Answer with only the number.",
]
ANSWERS = ["4", "42", "7", "17"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--wandb-project", default="lilo-examples")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
    )
    training = service.create_lora_training_client(
        base_model=args.base_model, rank=args.lora_rank, train_unembed=False
    )
    tokenizer = training.get_tokenizer()

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "base_model": args.base_model,
            "lora_rank": args.lora_rank,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "loss_fn": "importance_sampling",
            "base_url": os.environ["TINKER_BASE_URL"],
        },
    )

    for step in range(args.steps):
        step_start = time.time()
        prompt_text = PROMPTS[step % len(PROMPTS)]
        answer = ANSWERS[step % len(PROMPTS)]
        prompt = tokenizer.encode(prompt_text, add_special_tokens=True)

        sampling = training.save_weights_and_get_sampling_client()
        sampled = sampling.sample(
            prompt=types.ModelInput.from_ints(prompt),
            num_samples=args.group_size,
            sampling_params=types.SamplingParams(max_tokens=16, temperature=1.0),
        ).result(timeout=TIMEOUT)
        sample_time = time.time() - step_start

        rewards = [
            1.0 if answer in tokenizer.decode(list(seq.tokens)) else 0.0
            for seq in sampled.sequences
        ]
        mean_reward = sum(rewards) / len(rewards)
        datums = []
        for seq, reward in zip(sampled.sequences, rewards):
            tokens = list(seq.tokens)
            logprobs = list(seq.logprobs or ())
            if not tokens or len(logprobs) != len(tokens):
                raise RuntimeError("sampling returned invalid tokens or logprobs")
            advantage = reward - mean_reward
            prompt_targets = len(prompt) - 1
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(prompt + tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": prompt[1:] + tokens,
                        "logprobs": [0.0] * prompt_targets + logprobs,
                        "advantages": [0.0] * prompt_targets
                        + [advantage] * len(tokens),
                    },
                )
            )

        train_start = time.time()
        forward = training.forward_backward(datums, "importance_sampling")
        optimizer = training.optim_step(
            types.AdamParams(learning_rate=args.learning_rate)
        )
        forward_metrics = forward.result(timeout=TIMEOUT).metrics
        optim_metrics = optimizer.result(timeout=TIMEOUT).metrics
        train_time = time.time() - train_start

        metrics = {
            "reward/mean": mean_reward,
            "reward/max": max(rewards),
            "response_len/mean": sum(len(s.tokens) for s in sampled.sequences)
            / len(sampled.sequences),
            "time/sample_s": sample_time,
            "time/train_s": train_time,
            "time/step_s": time.time() - step_start,
            **{f"train/{k}": v for k, v in forward_metrics.items()},
            **{f"optim/{k}": v for k, v in optim_metrics.items()},
        }
        run.log(metrics, step=step)
        print(
            f"step {step}: reward={mean_reward:.2f} step_s={metrics['time/step_s']:.1f}"
        )

    run.finish()


if __name__ == "__main__":
    main()
