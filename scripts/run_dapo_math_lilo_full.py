# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dapo_math_comparison_common import (
    GRPO_STD_NORMALIZATION,
    MAX_STEPS,
    MAX_TOKENS,
    SAVE_EVERY,
    run_benchmark,
    timestamped_output,
    write_result,
)

BASE_URL = "https://modal-labs-kailash-dev--lilo-server.us-west.modal.run"
OUTPUT = "scripts/results/dapo_math_32k_qwen3_5_9b_lilo_full.json"
LOG = "scripts/results/dapo_math_32k_qwen3_5_9b_lilo_full.log"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY,
        help="persist a full state checkpoint and publish sampler weights every N steps",
    )
    parser.add_argument(
        "--grpo-std-normalization",
        action=argparse.BooleanOptionalAction,
        default=GRPO_STD_NORMALIZATION,
    )
    parser.add_argument(
        "--rollout-mode",
        choices=("cohort", "cookbook_async"),
        default="cohort",
    )
    parser.add_argument(
        "--remove-constant-reward-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rollout-min-containers", type=int)
    parser.add_argument("--rollout-max-containers", type=int)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run in a detached process and write output to a timestamped log",
    )
    args = parser.parse_args()

    if args.detach:
        log = timestamped_output(LOG)
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--steps",
            str(args.steps),
            "--max-tokens",
            str(args.max_tokens),
            "--save-every",
            str(args.save_every),
            "--base-url",
            args.base_url,
            "--output",
            args.output,
            (
                "--grpo-std-normalization"
                if args.grpo_std_normalization
                else "--no-grpo-std-normalization"
            ),
            "--rollout-mode",
            args.rollout_mode,
            (
                "--remove-constant-reward-groups"
                if args.remove_constant_reward_groups
                else "--no-remove-constant-reward-groups"
            ),
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

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    output = timestamped_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_result(
        output,
        run_benchmark(
            backend="lilo",
            parameterization="full",
            base_url=args.base_url,
            full=True,
            max_steps=args.steps,
            max_tokens=args.max_tokens,
            save_every=args.save_every,
            grpo_std_normalization=args.grpo_std_normalization,
            rollout_mode=args.rollout_mode,
            remove_constant_reward_groups=args.remove_constant_reward_groups,
            rollout={
                key: value
                for key, value in {
                    "min_containers": args.rollout_min_containers,
                    "max_containers": args.rollout_max_containers,
                }.items()
                if value is not None
            }
            or None,
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
