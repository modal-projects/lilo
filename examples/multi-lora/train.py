# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["tinker>=0.24.1,<0.25", "datasets>=3,<5"]
# ///
"""Train independent LoRA adapters on GSM8K through the Tinker SDK."""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import tinker
from datasets import load_dataset
from tinker import types

TIMEOUT = 3600


def final_number(text: str) -> float | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def train(client_id, service, data, args):
    training = service.create_lora_training_client(base_model=args.base_model, rank=16)
    tokenizer = training.get_tokenizer()

    for step in range(args.steps):
        started = time.perf_counter()
        row = data[(client_id * args.steps + step) % len(data)]
        target = final_number(row["answer"].split("####")[-1])
        if target is None:
            raise ValueError("GSM8K answer has no numerical target")
        prompt = tokenizer.encode(
            row["question"] + "\nGive your final numerical answer last.\nAnswer:",
            add_special_tokens=True,
        )

        sampling = training.save_weights_and_get_sampling_client()
        samples = sampling.sample(
            prompt=types.ModelInput.from_ints(prompt),
            num_samples=args.group_size,
            sampling_params=types.SamplingParams(
                max_tokens=args.max_tokens, temperature=1.0
            ),
        ).result(timeout=TIMEOUT)
        if len(samples.sequences) != args.group_size:
            raise RuntimeError("sampling returned an incomplete group")
        rewards = [
            float(final_number(tokenizer.decode(seq.tokens)) == target)
            for seq in samples.sequences
        ]
        mean_reward = sum(rewards) / len(rewards)

        batch = []
        for seq, reward in zip(samples.sequences, rewards):
            tokens = list(seq.tokens)
            logprobs = list(seq.logprobs or ())
            if not tokens or len(logprobs) != len(tokens):
                raise RuntimeError("sampling returned invalid tokens or logprobs")
            prefix = len(prompt) - 1
            batch.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(prompt + tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": prompt[1:] + tokens,
                        "logprobs": [0.0] * prefix + logprobs,
                        "advantages": [0.0] * prefix
                        + [reward - mean_reward] * len(tokens),
                    },
                )
            )

        forward = training.forward_backward(batch, "importance_sampling")
        optimizer = training.optim_step(types.AdamParams(learning_rate=1e-5))
        forward.result(timeout=TIMEOUT)
        optimizer.result(timeout=TIMEOUT)
        mean_tokens = sum(len(seq.tokens) for seq in samples.sequences) / len(rewards)
        print(
            f"client={client_id} step={step + 1} reward={mean_reward:.3f} "
            f"tokens={mean_tokens:.0f} step_s={time.perf_counter() - started:.1f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    args = parser.parse_args()
    if min(args.clients, args.steps, args.max_tokens) < 1 or args.group_size < 2:
        parser.error("clients, steps and max-tokens must be positive; group-size >= 2")

    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
    )
    data = load_dataset("openai/gsm8k", "main", split="train")
    with ThreadPoolExecutor(max_workers=args.clients) as pool:
        futures = [
            pool.submit(train, client_id, service, data, args)
            for client_id in range(args.clients)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
