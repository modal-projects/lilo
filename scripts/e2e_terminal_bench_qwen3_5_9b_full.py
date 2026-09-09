# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook[modal] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

MODEL_NAME = "Qwen/Qwen3.5-9B"
RENDERER_NAME = "qwen3_5"
DATASET = "terminal-bench-2.0/terminal-bench"
BASE_URL = "https://modal-labs-kailash-dev--lilo-server.us-west.modal.run"
MAX_TURNS = 20
MAX_GENERATION_TOKENS = 8192
MAX_TRAJECTORY_TOKENS = 120 * 1024
GROUP_SIZE = 4
GROUPS_PER_BATCH = 8
ROLLOUT_WORKERS = 4 * GROUPS_PER_BATCH
LEARNING_RATE = 1e-6
MAX_STEPS_OFF_POLICY = 1
GRAD_CLIP_NORM = 1.0


def _timestamped(path: Path) -> Path:
    import time

    stamp = time.strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.stem}.{stamp}{path.suffix}")


def _with_rollout_worker_count(original, worker_count: int):
    """Compile the pinned cookbook async loop with independent rollout workers."""
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(original)))
    replacements = {
        "maxsize=config.async_config.groups_per_batch": f"maxsize={worker_count}",
        "_AsyncCounter(config.async_config.groups_per_batch)": (
            f"_AsyncCounter({worker_count})"
        ),
        "range(config.async_config.groups_per_batch)": f"range({worker_count})",
    }
    expected = {
        "maxsize=config.async_config.groups_per_batch": 1,
        "_AsyncCounter(config.async_config.groups_per_batch)": 1,
        "range(config.async_config.groups_per_batch)": 2,
    }
    for old, new in replacements.items():
        if source.count(old) != expected[old]:
            raise RuntimeError(f"unexpected cookbook async loop shape for {old!r}")
        source = source.replace(old, new)
    namespace = dict(inspect.unwrap(original).__globals__)
    filename = inspect.getsourcefile(original) or "<cookbook>"
    exec(compile(source, filename, "exec"), namespace)
    return namespace["do_async_training"]


async def _run(
    log_path: str,
    *,
    steps: int,
    rollout_workers: int,
    base_url: str,
) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

    import tinker
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.recipes.harbor_rl.harbor_env import (
        HarborDatasetBuilder,
        default_sandbox_factory,
        load_harbor_tasks,
    )
    from tinker_cookbook.recipes.harbor_rl import train as harbor_train
    from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main
    from tinker_cookbook.rl import train as rl_train

    tasks = load_harbor_tasks(DATASET)
    if not tasks:
        raise RuntimeError(
            "no Terminal-Bench tasks found; download terminal-bench@2.0 "
            f"to ~/.cache/harbor/tasks/{DATASET}"
        )

    create_lora = tinker.ServiceClient.create_lora_training_client_async
    create_adam_params = tinker.AdamParams
    save_checkpoint = checkpoint_utils.save_checkpoint_async

    async def create_training(
        service,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ):
        del rank, seed, train_mlp, train_attn, train_unembed
        from lilo.client import create_full_training_client_async

        return await create_full_training_client_async(
            service,
            base_model,
            user_metadata=user_metadata,
        )

    def clipped_adam_params(*args, **kwargs):
        kwargs.setdefault("grad_clip_norm", GRAD_CLIP_NORM)
        return create_adam_params(*args, **kwargs)

    async def save_supported_checkpoint(*args, **kwargs):
        # Cookbook's kind="both" waits for state persistence and publication.
        # This benchmark publishes sampler weights only, without recovery
        # checkpoints. To retain state saves while overlapping persistence, see:
        # docs/full-fine-tunes.md#persist-checkpoints-without-blocking-every-training-step
        if kwargs.get("kind") == "both":
            kwargs["kind"] = "sampler"
        return await save_checkpoint(*args, **kwargs)

    async def skip_final_checkpoint(manager, loop_state):
        await manager.finalize_async()
        return {}

    do_async_training = _with_rollout_worker_count(
        rl_train.do_async_training,
        rollout_workers,
    )

    def harbor_dataset_with_headroom(*args, batch_size: int, **kwargs):
        if batch_size != GROUPS_PER_BATCH:
            raise ValueError(
                f"expected training batch of {GROUPS_PER_BATCH} groups, got {batch_size}"
            )
        return HarborDatasetBuilder(
            *args,
            batch_size=rollout_workers,
            **kwargs,
        )

    config = CLIConfig(
        model_name=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        lora_rank=32,
        max_tokens=MAX_GENERATION_TOKENS,
        temperature=1.0,
        max_turns=MAX_TURNS,
        sandbox_timeout=7200,
        command_timeout=300,
        grader_timeout=300,
        max_trajectory_tokens=MAX_TRAJECTORY_TOKENS,
        max_generation_tokens=MAX_GENERATION_TOKENS,
        group_size=GROUP_SIZE,
        groups_per_batch=GROUPS_PER_BATCH,
        learning_rate=LEARNING_RATE,
        num_substeps=1,
        log_path=log_path,
        eval_every=0,
        save_every=0,
        base_url=base_url,
        behavior_if_log_dir_exists="delete",
        max_steps_off_policy=MAX_STEPS_OFF_POLICY,
        max_steps=steps,
    )
    with (
        patch.object(
            tinker.ServiceClient,
            "create_lora_training_client_async",
            create_training,
        ),
        patch.object(tinker, "AdamParams", clipped_adam_params),
        patch.object(
            checkpoint_utils,
            "save_checkpoint_async",
            save_supported_checkpoint,
        ),
        patch.object(
            checkpoint_utils.CheckpointManager,
            "save_final_async",
            skip_final_checkpoint,
        ),
        patch.object(
            harbor_train,
            "HarborDatasetBuilder",
            harbor_dataset_with_headroom,
        ),
        patch.object(
            rl_train,
            "do_async_training",
            do_async_training,
        ),
    ):
        await cli_main(
            config,
            tasks,
            sandbox_factory=default_sandbox_factory,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=ROLLOUT_WORKERS,
        help="concurrent trajectory-group workers; each launches group_size trajectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "scripts/results/e2e_terminal_bench_qwen3_5_9b_full.json"
        ),
    )
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()

    if args.detach:
        log = _timestamped(args.output.with_suffix(".log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--steps",
            str(args.steps),
            "--rollout-workers",
            str(args.rollout_workers),
            "--base-url",
            args.base_url,
            "--output",
            str(args.output),
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

    with tempfile.TemporaryDirectory(prefix="qwen35-terminal-bench-") as temp:
        log_path = str(Path(temp) / "run")
        asyncio.run(
            _run(
                log_path,
                steps=args.steps,
                rollout_workers=args.rollout_workers,
                base_url=args.base_url,
            )
        )
        metrics_path = Path(log_path) / "metrics.jsonl"
        metrics = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]

    output = _timestamped(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "dataset": DATASET,
                "max_turns": MAX_TURNS,
                "max_generation_tokens": MAX_GENERATION_TOKENS,
                "max_trajectory_tokens": MAX_TRAJECTORY_TOKENS,
                "group_size": GROUP_SIZE,
                "groups_per_batch": GROUPS_PER_BATCH,
                "rollout_workers": args.rollout_workers,
                "max_concurrent_trajectories": (
                    args.rollout_workers * GROUP_SIZE
                ),
                "max_steps_off_policy": MAX_STEPS_OFF_POLICY,
                "steps": args.steps,
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
