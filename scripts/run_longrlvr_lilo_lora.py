# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets",
#   "wandb",
#   "tinker>=0.24,<0.25",
#   "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@c8ed9c764b59161391156f980102d82f05014765",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import inspect
import math
import os
import subprocess
import sys
import textwrap
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import tinker
from grouped_tinker_completer import GroupedTinkerTokenCompleter
from longrlvr_comparison_common import (
    ADAM_BETAS,
    ADAM_EPS,
    CONTEXT_LENGTH,
    DATASET_SEED,
    GRAD_CLIP,
    GROUP_SIZE,
    GROUPS_PER_BATCH,
    LEARNING_RATE,
    LOSS_FN,
    LOSS_FN_CONFIG,
    MAX_STEPS,
    MAX_STEPS_OFF_POLICY,
    MAX_TOKENS,
    MODEL_NAME,
    RENDERER_NAME,
    SAVE_EVERY,
    SOURCE_GROUPS_PER_BATCH,
    TEMPERATURE,
    WANDB_GROUP,
    WANDB_PROJECT,
    WEIGHT_DECAY,
)
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl.train import AsyncConfig, Config
from tinker_cookbook.rl.train import main as train_main

BASE_URL = os.environ.get("TINKER_BASE_URL", "")
SCRIPT_PATH = Path(__file__).resolve()
TRAINER_GPUS = 4


def _comparison_metrics(
    metrics: dict[str, object], trainer_gpus: int
) -> dict[str, float]:
    """Return metrics shared with the Miles baseline namespace."""

    def get(*names: str) -> float | None:
        for name in names:
            value = metrics.get(name)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    reward = get("env/all/reward/total")
    response_len = get("env/all/ac_tokens_per_turn")
    step_time = get("time/total")
    train_time = get("time/train_step")
    sampling_time = get("time/sampling_time_mean")
    samples = get("env/all/total_episodes")
    prompt_len = get("env/all/ob_tokens_per_turn")
    loss = get("loss:mean", "loss", "loss/mean", "train/loss")
    entropy = get("optim/entropy")
    truncated = get(
        "env/all/truncated_ratio",
        "env/all/truncated",
        "env/all/frac_truncated",
    )
    common: dict[str, float] = {}
    for key, value in (
        ("cmp/reward_mean", reward),
        ("cmp/response_len_mean", response_len),
        ("cmp/step_time_s", step_time),
        ("cmp/train_time_s", train_time),
        ("cmp/rollout_time_s", sampling_time),
        ("cmp/loss", loss),
        ("cmp/entropy", entropy),
        ("cmp/truncated_ratio", truncated),
    ):
        if value is not None:
            common[key] = value
    if step_time and samples:
        common["cmp/samples_per_s"] = samples / step_time
    if (
        step_time
        and samples
        and trainer_gpus
        and prompt_len is not None
        and response_len is not None
    ):
        common["cmp/tokens_per_gpu_per_s"] = (
            samples * (prompt_len + response_len) / step_time / trainer_gpus
        )
    return common


class _ComparisonLogger:
    def __init__(self, wrapped, trainer_gpus: int):
        self._wrapped = wrapped
        self._trainer_gpus = trainer_gpus

    @property
    def store(self):
        return self._wrapped.store

    def log_hparams(self, config):
        return self._wrapped.log_hparams(config)

    def log_metrics(self, metrics, step=None):
        combined = dict(metrics)
        combined.update(_comparison_metrics(metrics, self._trainer_gpus))
        return self._wrapped.log_metrics(combined, step)

    def log_long_text(self, key, text):
        return self._wrapped.log_long_text(key, text)

    def close(self):
        return self._wrapped.close()

    def sync(self):
        return self._wrapped.sync()

    def get_logger_url(self):
        return self._wrapped.get_logger_url()


def _with_rollout_worker_count(original, worker_count: int):
    """Compile the pinned Cookbook async loop with independent rollout workers."""
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
            raise RuntimeError(f"unexpected Cookbook async loop shape for {old!r}")
        source = source.replace(old, new)
    stale_requeue_marker = """\
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > config.async_config.max_steps_off_policy
                ):
                    if dataloader_done_event.is_set():
                        logger.info(
                            f"[training_loop] Step {i_batch}: Samples are too stale, "
                            "discarding (dataloader done)"
                        )
                    else:
                        logger.info(
                            f"[training_loop] Step {i_batch}: Samples are too stale, requeuing"
                        )
                        asyncio.create_task(
                            env_group_builders_queue.put(
                                wrapped_trajectory_group.env_group_builder
                            ),
                            name="requeue_stale_sample_task",
                        )
                    return False
"""
    if source.count(stale_requeue_marker) != 1:
        raise RuntimeError("unexpected Cookbook stale-sample shape")
    source = source.replace(
        stale_requeue_marker,
        """\
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > config.async_config.max_steps_off_policy
                ):
                    logger.info(
                        f"[training_loop] Step {i_batch}: Samples are too stale, "
                        "discarding"
                    )
                    return False
""",
    )
    valid_group_marker = """\
            else:
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=worker_metrics,
                    )
                )
"""
    if source.count(valid_group_marker) != 1:
        raise RuntimeError("unexpected Cookbook valid-group worker shape")
    source = source.replace(
        valid_group_marker,
        valid_group_marker
        + """\
                # Keep at most one valid group per worker and sampler version.
                # Invalid/constant groups retry immediately to preserve headroom.
                while (
                    sampling_client_step == sampling_client_step_copy
                    and not dataloader_done_event.is_set()
                ):
                    await asyncio.sleep(0.1)
""",
    )
    namespace = dict(inspect.unwrap(original).__globals__)
    filename = inspect.getsourcefile(original) or "<cookbook>"
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace["do_async_training"]


from longrlvr_dataset import LongRLVRDatasetBuilder


async def run(
    *,
    steps: int,
    group_size: int,
    groups_per_batch: int,
    source_group_multiplier: float,
    max_tokens: int,
    base_url: str,
    log_path: str,
    seed: int,
    wandb_group: str,
    run_name: str,
    trainer_gpus: int,
) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from tinker_cookbook.rl import rollouts as rl_rollouts
    from tinker_cookbook.rl import train as rl_train

    create_adam_params = tinker.AdamParams
    create_lora = tinker.ServiceClient.create_lora_training_client_async
    save_checkpoint = checkpoint_utils.save_checkpoint_async

    def clipped_adam_params(*args, **kwargs):
        kwargs.setdefault("beta1", ADAM_BETAS[0])
        kwargs.setdefault("beta2", ADAM_BETAS[1])
        kwargs.setdefault("eps", ADAM_EPS)
        kwargs.setdefault("weight_decay", WEIGHT_DECAY)
        kwargs.setdefault("grad_clip_norm", GRAD_CLIP)
        return create_adam_params(*args, **kwargs)

    async def create_lora_training(
        service,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = False,
        user_metadata: dict[str, str] | None = None,
    ):
        return await create_lora(
            service,
            base_model,
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            user_metadata=user_metadata,
        )

    async def save_supported_checkpoint(*args, **kwargs):
        if kwargs.get("kind") == "both":
            kwargs["kind"] = "sampler"
        return await save_checkpoint(*args, **kwargs)

    source_groups_per_batch = math.ceil(groups_per_batch * source_group_multiplier)
    do_async_training = _with_rollout_worker_count(
        rl_train.do_async_training,
        source_groups_per_batch,
    )
    original_setup_logging = rl_train.ml_log.setup_logging

    def setup_comparison_logging(*args, **kwargs):
        return _ComparisonLogger(
            original_setup_logging(*args, **kwargs),
            trainer_gpus,
        )

    dataset_builder = LongRLVRDatasetBuilder(
        batch_size=source_groups_per_batch,
        group_size=group_size,
        num_groups=steps * source_groups_per_batch,
        model_name=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        max_prompt_tokens=CONTEXT_LENGTH - max_tokens,
        seed=seed,
    )
    os.environ["WANDB_RUN_GROUP"] = wandb_group
    config = Config(
        learning_rate=LEARNING_RATE,
        dataset_builder=dataset_builder,
        model_name=MODEL_NAME,
        recipe_name="longrlvr",
        renderer_name=RENDERER_NAME,
        max_tokens=max_tokens,
        log_path=log_path,
        eval_every=0,
        save_every=SAVE_EVERY,
        base_url=base_url,
        wandb_project=WANDB_PROJECT,
        wandb_name=run_name,
        loss_fn=LOSS_FN,
        loss_fn_config=LOSS_FN_CONFIG,
        lora_rank=32,
        temperature=TEMPERATURE,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=True,
        compute_post_kl=False,
        num_groups_to_log=1,
        rollout_json_export=True,
        async_config=AsyncConfig(
            max_steps_off_policy=MAX_STEPS_OFF_POLICY,
            groups_per_batch=groups_per_batch,
        ),
        max_steps=steps,
    )
    with (
        patch.object(
            tinker.ServiceClient,
            "create_lora_training_client_async",
            create_lora_training,
        ),
        patch.object(tinker, "AdamParams", clipped_adam_params),
        patch.object(
            checkpoint_utils,
            "save_checkpoint_async",
            save_supported_checkpoint,
        ),
        patch.object(
            rl_rollouts,
            "TinkerTokenCompleter",
            partial(GroupedTinkerTokenCompleter, group_size=group_size),
        ),
        patch.object(
            rl_train,
            "do_async_training",
            do_async_training,
        ),
        patch.object(
            rl_train.ml_log,
            "setup_logging",
            setup_comparison_logging,
        ),
    ):
        await train_main(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 16K LongRLVR on the Lilo Qwen3.5-9B LoRA backend."
    )
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--groups-per-batch", type=int, default=GROUPS_PER_BATCH)
    parser.add_argument(
        "--source-group-multiplier",
        type=float,
        default=SOURCE_GROUPS_PER_BATCH / GROUPS_PER_BATCH,
    )
    parser.add_argument(
        "--max-generation-tokens",
        type=int,
        default=MAX_TOKENS,
    )
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--run-name")
    parser.add_argument("--trainer-gpus", type=int, default=TRAINER_GPUS)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run in a detached process and write stdout/stderr beside the log directory",
    )
    args = parser.parse_args()

    if not 0 < args.max_generation_tokens < CONTEXT_LENGTH:
        parser.error(
            f"--max-generation-tokens must be between 1 and {CONTEXT_LENGTH - 1}"
        )
    if args.steps <= 0 or args.group_size <= 1 or args.groups_per_batch <= 0:
        parser.error(
            "steps/groups-per-batch must be positive and group-size must exceed 1"
        )
    if args.source_group_multiplier < 1:
        parser.error("--source-group-multiplier must be at least 1")
    if not args.base_url:
        parser.error("--base-url or TINKER_BASE_URL is required")

    stamp = time.strftime("%Y%m%d%H%M%S")
    run_name = args.run_name or f"{args.wandb_group}-{stamp}"
    log_path = args.log_path or Path(
        f"scripts/results/longrlvr_qwen3_5_9b_lilo_lora_16k.{stamp}"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.detach:
        output_log = log_path.with_name(f"{log_path.name}.log")
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--steps",
            str(args.steps),
            "--group-size",
            str(args.group_size),
            "--groups-per-batch",
            str(args.groups_per_batch),
            "--source-group-multiplier",
            str(args.source_group_multiplier),
            "--max-generation-tokens",
            str(args.max_generation_tokens),
            "--base-url",
            args.base_url,
            "--seed",
            str(args.seed),
            "--wandb-group",
            args.wandb_group,
            "--run-name",
            run_name,
            "--trainer-gpus",
            str(args.trainer_gpus),
            "--log-path",
            str(log_path),
        ]
        with output_log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        print(f"detached pid={process.pid} log={output_log} run={log_path}")
        return

    asyncio.run(
        run(
            steps=args.steps,
            group_size=args.group_size,
            groups_per_batch=args.groups_per_batch,
            source_group_multiplier=args.source_group_multiplier,
            max_tokens=args.max_generation_tokens,
            base_url=args.base_url,
            log_path=str(log_path),
            seed=args.seed,
            wandb_group=args.wandb_group,
            run_name=run_name,
            trainer_gpus=args.trainer_gpus,
        )
    )
    print(log_path)


if __name__ == "__main__":
    main()
