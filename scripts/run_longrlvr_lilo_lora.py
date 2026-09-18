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
import contextlib
import hashlib
import inspect
import json
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
import torch
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
from tinker_cookbook.rl.train import AsyncConfig, Config
from tinker_cookbook.rl.train import main as train_main
from tinker_cookbook.tokenizer_utils import get_tokenizer, register_tokenizer

BASE_URL = os.environ.get("TINKER_BASE_URL", "")
SCRIPT_PATH = Path(__file__).resolve()
TRAINER_GPUS = 4


def _matched_advantage_stats(
    trajectory_groups_P,
    *,
    std_normalize: bool,
    per_token_scale: bool,
    sample_mean: bool = False,
):
    """Per-trajectory scalar advantages, scaled to the requested loss weighting.

    The server loss is a plain sum over response tokens, so the scale applied
    here is the effective per-token weight:
      token-mean  (per_token_scale): adv_i / T,          T = total tokens in batch
      sample-mean (sample_mean):     adv_i / (n * len_i), n = trajectories in batch
    """
    advantages_P = []
    total_tokens = 0
    num_trajectories = 0
    traj_lens_P = []
    first_before_std = None
    first_after_std = None
    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(
            traj_group.get_total_rewards(),
            dtype=torch.float32,
        )
        adv_G = rewards_G - rewards_G.mean()
        if first_before_std is None:
            first_before_std = float(adv_G.std().item()) if len(rewards_G) > 1 else 0.0
        if std_normalize and len(rewards_G) > 1:
            std = rewards_G.std()
            if std > 0:
                adv_G = adv_G / (std + 1e-6)
        if first_after_std is None:
            first_after_std = float(adv_G.std().item()) if len(adv_G) > 1 else 0.0
        advantages_P.append(adv_G)
        lens_G = torch.tensor(
            [
                sum(len(transition.ac.tokens) for transition in traj.transitions)
                for traj in traj_group.trajectories_G
            ],
            dtype=torch.float32,
        )
        traj_lens_P.append(lens_G)
        total_tokens += int(lens_G.sum().item())
        num_trajectories += len(lens_G)
    if sample_mean and num_trajectories > 0:
        advantages_P = [
            advantages / (num_trajectories * lens_G.clamp(min=1.0))
            for advantages, lens_G in zip(advantages_P, traj_lens_P, strict=True)
        ]
    elif per_token_scale and total_tokens > 0:
        advantages_P = [advantages / total_tokens for advantages in advantages_P]
    return advantages_P, total_tokens, first_before_std, first_after_std


def matched_compute_advantages(
    trajectory_groups_P,
    *,
    std_normalize: bool,
    per_token_scale: bool,
    sample_mean: bool = False,
):
    return _matched_advantage_stats(
        trajectory_groups_P,
        std_normalize=std_normalize,
        per_token_scale=per_token_scale,
        sample_mean=sample_mean,
    )[0]


def _comparison_metrics(
    metrics: dict[str, object],
    trainer_gpus: int,
    advantage_diagnostics: dict[str, float | bool],
    std_normalize_advantages: bool,
    per_token_loss_scale: bool,
    sample_mean_advantages: bool = False,
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
        ("cmp/prompt_len_mean", prompt_len),
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
    if std_normalize_advantages or per_token_loss_scale or sample_mean_advantages:
        common.update(
            {
                "cmp/adv_total_tokens": float(advantage_diagnostics["total_tokens"]),
                "cmp/std_normalize_advantages": float(std_normalize_advantages),
                "cmp/per_token_loss_scale": float(per_token_loss_scale),
                "cmp/sample_mean_advantages": float(sample_mean_advantages),
            }
        )
    return common


def _pinned_prompt_metadata(prompt_file: str | None) -> dict[str, object]:
    if prompt_file is None:
        return {}
    path = Path(prompt_file)
    payload = path.read_bytes()
    document = json.loads(payload)
    rows = document if isinstance(document, list) else document["rows"]
    if not isinstance(rows, list):
        raise TypeError(f"prompt file must contain a JSON list: {path}")
    return {
        "pinned_prompt_file": str(path),
        "pinned_prompt_sha256": hashlib.sha256(payload).hexdigest(),
        "pinned_prompt_rows": len(rows),
    }


class _ComparisonLogger:
    def __init__(
        self,
        wrapped,
        trainer_gpus: int,
        advantage_diagnostics: dict[str, float],
        std_normalize_advantages: bool,
        per_token_loss_scale: bool,
        sample_mean_advantages: bool = False,
    ):
        self._wrapped = wrapped
        self._trainer_gpus = trainer_gpus
        self._advantage_diagnostics = advantage_diagnostics
        self._std_normalize_advantages = std_normalize_advantages
        self._per_token_loss_scale = per_token_loss_scale
        self._sample_mean_advantages = sample_mean_advantages

    @property
    def store(self):
        return self._wrapped.store

    def log_hparams(self, config):
        return self._wrapped.log_hparams(config)

    def log_metrics(self, metrics, step=None):
        combined = dict(metrics)
        combined.update(
            _comparison_metrics(
                metrics,
                self._trainer_gpus,
                self._advantage_diagnostics,
                self._std_normalize_advantages,
                self._per_token_loss_scale,
                self._sample_mean_advantages,
            )
        )
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
    max_steps_off_policy: int,
    std_normalize_advantages: bool,
    per_token_loss_scale: bool,
    base_model: str,
    prompt_file: str | None,
    sample_mean_advantages: bool = False,
    context_length: int,
    pad_to_tokens: int = 0,
    engine_model: str | None = None,
    save_every: int = SAVE_EVERY,
) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from tinker_cookbook.rl import rollouts as rl_rollouts
    from tinker_cookbook.rl import train as rl_train

    if base_model != MODEL_NAME:
        register_tokenizer(base_model, lambda: get_tokenizer(MODEL_NAME))

    create_adam_params = tinker.AdamParams
    create_lora = tinker.ServiceClient.create_lora_training_client_async

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
            engine_model or base_model,
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            user_metadata=user_metadata,
        )


    source_groups_per_batch = math.ceil(groups_per_batch * source_group_multiplier)
    advantage_diagnostics = {"total_tokens": 0.0}
    pinned_prompt_metadata = _pinned_prompt_metadata(prompt_file)

    def compute_matched_advantages(trajectory_groups_P):
        (
            advantages_P,
            total_tokens,
            first_before_std,
            first_after_std,
        ) = _matched_advantage_stats(
            trajectory_groups_P,
            std_normalize=std_normalize_advantages,
            per_token_scale=per_token_loss_scale,
            sample_mean=sample_mean_advantages,
        )
        advantage_diagnostics["total_tokens"] = float(total_tokens)
        if first_before_std is not None and not advantage_diagnostics.get(
            "sanity_logged", False
        ):
            print(
                "[matched_advantages] step=0 "
                f"std_before={first_before_std:.6f} "
                f"std_after_normalize={first_after_std:.6f} "
                f"total_tokens={total_tokens} "
                f"std_normalize={std_normalize_advantages} "
                f"per_token_scale={per_token_loss_scale} "
                f"sample_mean={sample_mean_advantages}",
                flush=True,
            )
            advantage_diagnostics["sanity_logged"] = True
        return advantages_P

    do_async_training = _with_rollout_worker_count(
        rl_train.do_async_training,
        source_groups_per_batch,
    )
    original_setup_logging = rl_train.ml_log.setup_logging

    def setup_comparison_logging(*args, **kwargs):
        wrapped = original_setup_logging(*args, **kwargs)
        if pinned_prompt_metadata:
            wrapped.log_hparams(pinned_prompt_metadata)
        return _ComparisonLogger(
            wrapped,
            trainer_gpus,
            advantage_diagnostics,
            std_normalize_advantages,
            per_token_loss_scale,
            sample_mean_advantages,
        )

    dataset_builder = LongRLVRDatasetBuilder(
        batch_size=source_groups_per_batch,
        group_size=group_size,
        num_groups=steps * source_groups_per_batch,
        model_name=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        max_prompt_tokens=context_length - max_tokens,
        seed=seed,
        prompt_file=prompt_file,
        pad_to_tokens=pad_to_tokens,
    )
    os.environ["WANDB_RUN_GROUP"] = wandb_group
    config = Config(
        learning_rate=LEARNING_RATE,
        dataset_builder=dataset_builder,
        model_name=base_model,
        recipe_name="longrlvr",
        renderer_name=RENDERER_NAME,
        max_tokens=max_tokens,
        log_path=log_path,
        eval_every=0,
        save_every=save_every,
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
            max_steps_off_policy=max_steps_off_policy,
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
            rl_rollouts,
            "TinkerTokenCompleter",
            partial(GroupedTinkerTokenCompleter, group_size=group_size),
        ),
        patch.object(
            rl_train,
            "do_async_training",
            do_async_training,
        ),
        (
            patch.object(
                rl_train,
                "compute_advantages",
                compute_matched_advantages,
            )
            if std_normalize_advantages
            or per_token_loss_scale
            or sample_mean_advantages
            else contextlib.nullcontext()
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
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
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
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--prompt-file")
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument(
        "--pad-to-tokens",
        type=int,
        default=0,
        help="pad prompts with unrelated LongRLVR chunks up to this token count",
    )
    parser.add_argument(
        "--engine-model",
        default=None,
        help="model identifier sent to the engine for LoRA creation "
        "(e.g. a definition id); defaults to --base-model",
    )
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--run-name")
    parser.add_argument("--trainer-gpus", type=int, default=TRAINER_GPUS)
    parser.add_argument(
        "--max-steps-off-policy",
        type=int,
        default=MAX_STEPS_OFF_POLICY,
    )
    parser.add_argument(
        "--std-normalize-advantages",
        action="store_true",
    )
    parser.add_argument(
        "--per-token-loss-scale",
        action="store_true",
    )
    parser.add_argument(
        "--sample-mean-advantages",
        action="store_true",
        help="weight each trajectory equally: per-token advantage adv_i/(n*len_i)",
    )
    parser.add_argument("--log-path", type=Path)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run in a detached process and write stdout/stderr beside the log directory",
    )
    args = parser.parse_args()

    if args.per_token_loss_scale and args.sample_mean_advantages:
        parser.error(
            "--per-token-loss-scale and --sample-mean-advantages are exclusive"
        )
    if not 0 < args.max_generation_tokens < args.context_length:
        parser.error(
            f"--max-generation-tokens must be between 1 and {args.context_length - 1}"
        )
    if args.steps <= 0 or args.group_size <= 1 or args.groups_per_batch <= 0:
        parser.error(
            "steps/groups-per-batch must be positive and group-size must exceed 1"
        )
    if args.source_group_multiplier < 1:
        parser.error("--source-group-multiplier must be at least 1")
    if args.max_steps_off_policy < 0:
        parser.error("--max-steps-off-policy must be non-negative")
    if not args.base_url:
        parser.error("--base-url or TINKER_BASE_URL is required")

    stamp = time.strftime("%Y%m%d%H%M%S")
    run_name = args.run_name or f"{args.wandb_group}-{stamp}"
    log_path = args.log_path or Path(
        f"scripts/results/longrlvr_qwen3_5_9b_lilo_lora_16k.{stamp}"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
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
            "--base-model",
            args.base_model,
            *(["--prompt-file", args.prompt_file] if args.prompt_file else []),
            *(["--engine-model", args.engine_model] if args.engine_model else []),
            "--max-steps-off-policy",
            str(args.max_steps_off_policy),
            "--context-length",
            str(args.context_length),
            *(
                ["--pad-to-tokens", str(args.pad_to_tokens)]
                if args.pad_to_tokens
                else []
            ),
            *(["--std-normalize-advantages"] if args.std_normalize_advantages else []),
            *(["--per-token-loss-scale"] if args.per_token_loss_scale else []),
            *(["--sample-mean-advantages"] if args.sample_mean_advantages else []),
            "--save-every",
            str(args.save_every),
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
            max_steps_off_policy=args.max_steps_off_policy,
            std_normalize_advantages=args.std_normalize_advantages,
            per_token_loss_scale=args.per_token_loss_scale,
            sample_mean_advantages=args.sample_mean_advantages,
            base_model=args.base_model,
            prompt_file=args.prompt_file,
            context_length=args.context_length,
            pad_to_tokens=args.pad_to_tokens,
            engine_model=args.engine_model,
            save_every=args.save_every,
        )
    )
    print(log_path)


if __name__ == "__main__":
    main()
