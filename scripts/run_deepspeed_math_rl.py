# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "fastapi>=0.141.1",
#   "httpx>=0.28.1",
#   "jinja2>=3.1",
#   "modal>=1.5.3",
#   "pydantic>=2.13.4",
#   "stitch @ git+https://github.com/modal-projects/stitch.git@375a9396a7b05770dc4ed9cc5fe34fc4d5a472d5",
#   "tinker>=0.24,<0.25",
#   "uvicorn>=0.52.0",
#   "xxhash>=3.8.1",
#   "zstandard>=0.25.0",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import modal
import tinker
from tinker import types

MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
DEFAULT_LEARNING_RATE = 1e-6
TIMEOUT = 3 * 60 * 60
PROBLEMS = (
    ("Nora has 12 boxes with 8 pencils each. She gives away 17. How many remain?", 79),
    ("A bus has 23 passengers. 18 board and 9 leave. How many are on the bus?", 32),
    ("A farmer fills 7 baskets with 24 apples each and sells 53. How many remain?", 115),
    ("Notebooks cost $6 and pens cost $4. What do 7 notebooks and 5 pens cost?", 62),
    ("A tank has 350 liters. It drains 87 liters and gains 46. How many remain?", 309),
    ("There are 4 trays of 18 cookies. Children eat 29. How many remain?", 43),
    ("A library has 215 books, receives 68, and lends 97. How many remain?", 186),
    ("A runner completes 6 laps of 400 meters plus 350 meters. How far in meters?", 2750),
)
ANSWER = re.compile(r"Answer:\s*(-?\d+)", re.IGNORECASE)


def timestamped(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.stem}.{stamp}{path.suffix}")


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prompt_tokens(tokenizer, question: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "Solve the arithmetic problem. End with exactly "
                    "`Answer: <integer>`."
                ),
            },
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return list(encoded)


def score(text: str, expected: int) -> float:
    matches = ANSWER.findall(text)
    return 1.0 if matches and int(matches[-1]) == expected else -1.0


def rl_datum(
    prompt: list[int],
    response: list[int],
    logprobs: list[float],
    reward: float,
) -> types.Datum:
    prefix = len(prompt) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(prompt + response[:-1]),
        loss_fn_inputs={
            "target_tokens": [0] * prefix + response,
            "logprobs": [0.0] * prefix + logprobs,
            "advantages": [0.0] * prefix + [reward] * len(response),
        },
    )


def unload(base_url: str, api_key: str, model_id: str) -> None:
    headers = {"X-API-Key": api_key}
    with httpx.Client(base_url=base_url, headers=headers, timeout=60) as client:
        response = client.post("/api/v1/unload_model", json={"model_id": model_id})
        response.raise_for_status()


def run(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from lilo.client import create_full_training_client
    from lilo.providers.modal.app import app, server

    output = timestamped(Path(args.output))
    report = {
        "status": "running",
        "model": MODEL_NAME,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "trajectories_per_step": args.batch_size * args.group_size,
            "max_tokens": args.max_tokens,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "steps": [],
    }
    write_report(output, report)
    training = None
    api_key = os.environ["TINKER_API_KEY"]
    try:
        with modal.enable_output(), app.run(
            name=f"deepspeed-math-rl-{time.strftime('%H%M%S')}"
        ):
            base_url = server.get_web_url()
            if not base_url:
                raise RuntimeError("Modal did not provide a control-plane URL")
            service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
            training = create_full_training_client(
                service,
                MODEL_NAME,
                rollout={"min_containers": 1, "max_containers": 1},
            )
            tokenizer = training.get_tokenizer()
            sampling = training.save_weights_and_get_sampling_client()

            for step in range(args.steps):
                data = []
                samples = []
                rollouts = []
                for batch_index in range(args.batch_size):
                    problem_index = (
                        step * args.batch_size + batch_index
                    ) % len(PROBLEMS)
                    question, expected = PROBLEMS[problem_index]
                    prompt = prompt_tokens(tokenizer, question)
                    future = sampling.sample(
                        prompt=types.ModelInput.from_ints(prompt),
                        num_samples=args.group_size,
                        sampling_params=types.SamplingParams(
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            seed=args.seed + step * args.batch_size + batch_index,
                        ),
                    )
                    rollouts.append((question, expected, prompt, future))

                for question, expected, prompt, future in rollouts:
                    sampled = future.result(timeout=TIMEOUT)
                    for sequence in sampled.sequences:
                        response = list(sequence.tokens)
                        logprobs = list(sequence.logprobs or ())
                        if not response or len(logprobs) != len(response):
                            raise RuntimeError("sample has missing tokens or logprobs")
                        text = tokenizer.decode(response)
                        reward = score(text, expected)
                        data.append(rl_datum(prompt, response, logprobs, reward))
                        samples.append(
                            {
                                "question": question,
                                "expected": expected,
                                "text": text,
                                "reward": reward,
                            }
                        )

                forward = training.forward_backward(data, "importance_sampling")
                optimizer = training.optim_step(
                    types.AdamParams(
                        learning_rate=args.learning_rate,
                        grad_clip_norm=1.0,
                    )
                )
                forward_result = forward.result(timeout=TIMEOUT)
                optimizer_result = optimizer.result(timeout=TIMEOUT)
                sampling = training.save_weights_and_get_sampling_client()
                record = {
                    "step": step,
                    "reward": sum(item["reward"] for item in samples) / len(samples),
                    "correct_rate": sum(
                        item["reward"] > 0 for item in samples
                    )
                    / len(samples),
                    "samples": samples,
                    "forward_metrics": dict(forward_result.metrics),
                    "optimizer_metrics": dict(optimizer_result.metrics),
                }
                report["steps"].append(record)
                write_report(output, report)
                print(
                    f"rl_step={step} correct={record['correct_rate']:.3f} "
                    f"reward={record['reward']:.3f} "
                    f"loss={record['forward_metrics'].get('loss:sum')} "
                    f"grad_norm={record['optimizer_metrics'].get('grad_norm:mean')}",
                    flush=True,
                )

            window = min(5, len(report["steps"]))
            first_reward = sum(
                item["reward"] for item in report["steps"][:window]
            ) / window
            last_reward = sum(
                item["reward"] for item in report["steps"][-window:]
            ) / window
            report["summary"] = {
                "window_steps": window,
                "first_reward": first_reward,
                "last_reward": last_reward,
                "reward_gain": last_reward - first_reward,
            }
            report["status"] = "passed"
            write_report(output, report)
            unload(base_url, api_key, str(training.model_id))
            training = None
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report(output, report)
        raise
    finally:
        if training is not None:
            try:
                unload(base_url, api_key, str(training.model_id))
            except Exception:
                pass
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=len(PROBLEMS))
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="scripts/results/deepspeed_qwen3_5_9b_math_rl.json",
    )
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.group_size < 1
        or args.max_tokens < 1
        or args.learning_rate <= 0
        or args.temperature <= 0
    ):
        parser.error(
            "steps, batch-size, group-size, max-tokens, learning-rate, and "
            "temperature must be positive"
        )

    if args.detach:
        log = timestamped(Path(args.output).with_suffix(".log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--group-size",
            str(args.group_size),
            "--max-tokens",
            str(args.max_tokens),
            "--learning-rate",
            str(args.learning_rate),
            "--temperature",
            str(args.temperature),
            "--seed",
            str(args.seed),
            "--output",
            args.output,
        ]
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        print(f"detached pid={process.pid} log={log}")
        return

    print(run(args))


if __name__ == "__main__":
    main()
