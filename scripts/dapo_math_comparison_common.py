from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from unittest.mock import patch

MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
COOKBOOK_REVISION = "c8ed9c764b59161391156f980102d82f05014765"
ENVIRONMENT = "dapo-math-17k"
DATASET = "open-r1/DAPO-Math-17k-Processed"
CONTEXT_LENGTH = 32_768
MAX_TOKENS = 20_480
DATASET_SEED = 0
LORA_RANK = 32
LORA_SEED = 4242
GROUP_SIZE = 8
GROUPS_PER_BATCH = 32
SOURCE_GROUPS_PER_BATCH = 4 * GROUPS_PER_BATCH
ROLLOUT_WORKERS = SOURCE_GROUPS_PER_BATCH
MAX_STEPS = 30
MAX_STEPS_OFF_POLICY = 1
SAVE_EVERY = 5
LEARNING_RATE = 1e-6
GRAD_CLIP_NORM = 1.0
TEMPERATURE = 1.0
GRPO_STD_NORMALIZATION = True
GRPO_STD_EPSILON = 1e-6
LOSS_FN = "ppo"
LOSS_FN_CONFIG = {
    "clip_low_threshold": 0.8,
    "clip_high_threshold": 1.28,
}


def timestamped_output(base: str) -> Path:
    path = Path(base)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return path.with_name(f"{path.stem}.{timestamp}{path.suffix or '.json'}")


def write_result(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def experiment_config(
    *,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
    save_every: int = SAVE_EVERY,
    model_name: str = MODEL_NAME,
    renderer_name: str | None = None,
    rollout_mode: str = "cohort",
    grpo_std_normalization: bool = GRPO_STD_NORMALIZATION,
    remove_constant_reward_groups: bool = True,
    rollout: dict[str, int] | None = None,
) -> dict:
    if rollout_mode not in {"cohort", "cookbook_async"}:
        raise ValueError(f"unsupported rollout mode: {rollout_mode}")
    cohort_mode = rollout_mode == "cohort"
    source_groups = SOURCE_GROUPS_PER_BATCH if cohort_mode else GROUPS_PER_BATCH
    return {
        "base_model": model_name,
        "renderer_name": renderer_name,
        "cookbook_revision": COOKBOOK_REVISION,
        "environment": ENVIRONMENT,
        "dataset": DATASET,
        "execution": "async",
        "algorithm": "dapo",
        "context_length": CONTEXT_LENGTH,
        "dataset_seed": DATASET_SEED,
        "group_size": GROUP_SIZE,
        "groups_per_batch": GROUPS_PER_BATCH,
        "source_groups_per_batch": source_groups,
        "rollout_workers": ROLLOUT_WORKERS if cohort_mode else GROUPS_PER_BATCH,
        "rollout_selection": (
            "batch_scoped_first_completed" if cohort_mode else "continuous_fifo"
        ),
        "surplus_rollouts": (
            "drain_and_discard" if cohort_mode else "retain_until_trained"
        ),
        "surplus_drain": "best_effort_background" if cohort_mode else None,
        "rollout_pool": rollout,
        "trajectories_per_step": GROUP_SIZE * GROUPS_PER_BATCH,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
        "max_steps_off_policy": MAX_STEPS_OFF_POLICY,
        "save_every": save_every,
        "checkpoint_kind": "sampler" if save_every == 0 else "both",
        "temperature": TEMPERATURE,
        "learning_rate": LEARNING_RATE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "loss_fn": LOSS_FN,
        "loss_fn_config": LOSS_FN_CONFIG,
        "grpo_std_normalization": grpo_std_normalization,
        "reward": "correct if terminal Answer line + User stop, else -0.1",
        "remove_constant_reward_groups": remove_constant_reward_groups,
    }


def _compute_miles_grpo_advantages(trajectory_groups):
    """Match Miles' group-centered, sample-std-normalized GRPO advantages."""
    import torch

    advantages = []
    for trajectory_group in trajectory_groups:
        rewards = torch.tensor(
            trajectory_group.get_total_rewards(),
            dtype=torch.float32,
        )
        centered = rewards - rewards.mean()
        advantages.append(centered / (centered.std() + GRPO_STD_EPSILON))
    return advantages


async def _select_first_valid(tasks, target_count: int):
    """Return first-completed non-None results and leave stragglers pending."""
    pending = set(tasks)
    selected = []
    completed_discarded = 0
    while pending and len(selected) < target_count:
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            result = task.result()
            if result is not None and len(selected) < target_count:
                selected.append(result)
            else:
                completed_discarded += 1
    if len(selected) != target_count:
        await asyncio.gather(*pending, return_exceptions=True)
        raise RuntimeError(
            f"expected {target_count} valid results, got {len(selected)}"
        )
    return selected, pending, completed_discarded


async def _drain_discarded_tasks(tasks, cohort_index: int) -> None:
    """Consume durable Tinker futures without making them training candidates."""
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if errors:
        raise RuntimeError(
            f"cohort {cohort_index} surplus drain raised {len(errors)} errors"
        ) from errors[0]


async def _run_recipe(
    log_path: str,
    *,
    model_name: str,
    renderer_name: str | None,
    base_url: str | None,
    full: bool,
    max_steps: int,
    max_tokens: int,
    save_every: int,
    rollout_mode: str,
    grpo_std_normalization: bool,
    remove_constant_reward_groups: bool,
    rollout: dict[str, int] | None,
) -> None:
    import tinker
    from dapo_math_dataset import DAPOMathDatasetBuilder
    from grouped_tinker_completer import GroupedTinkerTokenCompleter
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.recipes.math_rl import train as math_train
    from tinker_cookbook.rl import data_processing as rl_data_processing
    from tinker_cookbook.rl import rollouts as rl_rollouts
    from tinker_cookbook.rl import train as rl_train
    from tinker_cookbook.utils import ml_log

    CLIConfig = math_train.CLIConfig
    cli_main = math_train.cli_main
    create_config = math_train.Config
    create_lora = tinker.ServiceClient.create_lora_training_client_async
    create_adam_params = tinker.AdamParams
    save_checkpoint = checkpoint_utils.save_checkpoint_async
    benchmark_started = time.perf_counter()
    json_log_metrics = ml_log.JsonLogger.log_metrics

    async def do_cohort_async_training(
        start_batch,
        end_batch,
        num_batches,
        config,
        training_client,
        kl_reference_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer,
        error_counter=None,
        strategy=None,
        checkpoint_mgr=None,
    ):
        """Run batch-scoped rollout cohorts without reusing stragglers."""
        if config.async_config is None:
            raise ValueError("cohort async training requires async_config")
        if config.stream_minibatch_config is not None:
            raise ValueError(
                "cohort async training does not support streaming minibatches"
            )
        if evaluators and config.eval_every:
            raise ValueError("cohort async training does not support evaluators")

        # With full-state saves enabled, kind="both" waits for checkpoint
        # persistence AND sampler publication before returning. The wrapper
        # below uses sampler-only saves when full and save_every == 0; these
        # do not create recovery checkpoints. To overlap state persistence:
        # docs/full-fine-tunes.md#persist-checkpoints-without-blocking-every-training-step
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{start_batch:06d}",
            log_path=config.log_path,
            loop_state={"batch": start_batch},
            kind="both",
            ttl_seconds=config.ttl_seconds,
            store=ml_logger.store,
        )
        sampling_client = training_client.create_sampling_client(
            path_dict["sampler_path"]
        )
        sampling_client_step = start_batch
        target_groups = config.async_config.groups_per_batch

        async def generate_cohort(
            cohort_index,
            cohort_sampling_client,
            cohort_sampling_step,
        ):
            builders = list(dataset.get_batch(cohort_index))
            if len(builders) != ROLLOUT_WORKERS:
                raise RuntimeError(
                    f"cohort {cohort_index} expected {ROLLOUT_WORKERS} "
                    f"candidate groups, got {len(builders)}"
                )

            async def generate_one(builder):
                started = time.time()
                trajectory_group = (
                    await rl_train.do_group_rollout_and_filter_constant_reward(
                        cohort_sampling_client,
                        builder,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        do_remove_constant_reward_groups=(
                            config.remove_constant_reward_groups
                        ),
                        strategy=strategy,
                        termination=config.effective_termination(),
                    )
                )
                if error_counter is not None:
                    error_counter.ingest(trajectory_group)
                if trajectory_group is None:
                    return None
                return rl_train.WrappedTrajectoryGroup(
                    trajectory_group=trajectory_group,
                    env_group_builder=builder,
                    sampling_client_step=cohort_sampling_step,
                    metrics={
                        "time/trajectory_group_worker_loop/total": (
                            time.time() - started
                        )
                    },
                )

            tasks = {
                asyncio.create_task(
                    generate_one(builder),
                    name=f"cohort_{cohort_index}_group_{group_index}",
                )
                for group_index, builder in enumerate(builders)
            }
            selected, pending, completed_discarded = await _select_first_valid(
                tasks,
                target_groups,
            )
            pending_at_selection = len(pending)

            drain_task = asyncio.create_task(
                _drain_discarded_tasks(pending, cohort_index),
                name=f"cohort_{cohort_index}_surplus_drain",
            )
            return (
                selected,
                drain_task,
                {
                    "rollout_cohort/candidate_groups": len(builders),
                    "rollout_cohort/selected_groups": len(selected),
                    "rollout_cohort/completed_discarded_at_selection": (
                        completed_discarded
                    ),
                    "rollout_cohort/pending_discarded_at_selection": (
                        pending_at_selection
                    ),
                },
            )

        cohort_task = asyncio.create_task(
            generate_cohort(
                start_batch,
                sampling_client,
                sampling_client_step,
            ),
            name=f"rollout_cohort_{start_batch}",
        )
        background_drain_tasks = set()

        def drain_finished(task):
            background_drain_tasks.discard(task)
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                rl_train.logger.exception(
                    "discarded rollout cohort drain raised an error"
                )

        for i_batch in range(start_batch, end_batch):
            (
                wrapped_groups,
                current_drain_task,
                cohort_metrics,
            ) = await cohort_task

            background_drain_tasks.add(current_drain_task)
            current_drain_task.add_done_callback(drain_finished)
            cohort_metrics["rollout_cohort/background_drains"] = len(
                background_drain_tasks
            )

            if i_batch + 1 < end_batch:
                cohort_task = asyncio.create_task(
                    generate_cohort(
                        i_batch + 1,
                        sampling_client,
                        sampling_client_step,
                    ),
                    name=f"rollout_cohort_{i_batch + 1}",
                )

            metrics = {
                "training_client/step": i_batch,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
                **cohort_metrics,
            }
            metrics.update(rl_train.compute_sampling_client_metrics(wrapped_groups))

            rl_train.logger.info(
                f"[cohort_training_loop] Step {i_batch}: Will train on "
                f"{len(wrapped_groups)} groups and discard the remaining cohort"
            )
            with rl_train.trace.trace_iteration(step=i_batch) as window:
                (
                    sampling_client,
                    train_step_metrics,
                ) = await rl_train.do_train_step_and_get_sampling_client(
                    config,
                    i_batch,
                    training_client,
                    checkpoint_mgr,
                    kl_reference_client,
                    tokenizer,
                    [group.env_group_builder for group in wrapped_groups],
                    [group.trajectory_group for group in wrapped_groups],
                )

            sampling_client_step = i_batch + 1
            if checkpoint_mgr is not None:
                await checkpoint_mgr.maybe_save_rolling_async(
                    step=i_batch + 1,
                    loop_state={"batch": i_batch + 1},
                )

            metrics.update(train_step_metrics)
            if error_counter is not None:
                metrics.update(error_counter.get_metrics())
            metrics.update(window.get_timing_metrics())
            window.save_timing(i_batch, store=ml_logger.store)
            ml_logger.log_metrics(metrics, step=i_batch)

        if background_drain_tasks:
            rl_train.logger.info(
                "abandoning %d local surplus-drain waiters; accepted Tinker "
                "requests remain durable",
                len(background_drain_tasks),
            )
            for task in background_drain_tasks:
                task.cancel()
            await asyncio.gather(
                *background_drain_tasks,
                return_exceptions=True,
            )

    def clipped_adam_params(*args, **kwargs):
        kwargs.setdefault("grad_clip_norm", GRAD_CLIP_NORM)
        return create_adam_params(*args, **kwargs)

    def dapo_config(*args, **kwargs):
        kwargs["remove_constant_reward_groups"] = remove_constant_reward_groups
        return create_config(*args, **kwargs)

    def dapo_dataset_with_dynamic_sampling_headroom(
        env,
        batch_size,
        model_name,
        renderer_name,
        group_size,
        seed=0,
    ):
        assert env == ENVIRONMENT
        assert batch_size == GROUPS_PER_BATCH
        return DAPOMathDatasetBuilder(
            batch_size=(
                SOURCE_GROUPS_PER_BATCH
                if rollout_mode == "cohort"
                else GROUPS_PER_BATCH
            ),
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )

    def timed_json_log_metrics(self, metrics, step=None):
        metrics = dict(metrics)
        if "training_client/step" in metrics:
            metrics["benchmark/wall_elapsed"] = time.perf_counter() - benchmark_started
        return json_log_metrics(self, metrics, step)

    async def skip_final_checkpoint(manager, loop_state):
        await manager.finalize_async()
        return {}

    async def save_supported_checkpoint(*args, **kwargs):
        if full and save_every == 0 and kwargs.get("kind") == "both":
            kwargs["kind"] = "sampler"
        return await save_checkpoint(*args, **kwargs)

    async def create_training(
        service,
        base_model: str,
        rank: int = LORA_RANK,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ):
        if full:
            from lilo.client import create_full_training_client_async

            return await create_full_training_client_async(
                service,
                base_model,
                user_metadata=user_metadata,
                rollout=rollout,
            )
        return await create_lora(
            service,
            base_model,
            rank=LORA_RANK,
            seed=LORA_SEED,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata=user_metadata,
        )

    config = CLIConfig(
        model_name=model_name,
        renderer_name=renderer_name,
        lora_rank=LORA_RANK,
        env=ENVIRONMENT,
        seed=DATASET_SEED,
        group_size=GROUP_SIZE,
        groups_per_batch=GROUPS_PER_BATCH,
        learning_rate=LEARNING_RATE,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        kl_penalty_coef=0.0,
        num_substeps=1,
        log_path=log_path,
        compute_post_kl=False,
        eval_every=0,
        save_every=save_every,
        base_url=base_url,
        behavior_if_log_dir_exists="delete",
        max_steps_off_policy=MAX_STEPS_OFF_POLICY,
        loss_fn=LOSS_FN,
        loss_fn_config=LOSS_FN_CONFIG,
        max_steps=max_steps,
    )
    do_async_patch = (
        patch.object(
            rl_train,
            "do_async_training",
            do_cohort_async_training,
        )
        if rollout_mode == "cohort"
        else contextlib.nullcontext()
    )
    data_advantage_patch = (
        patch.object(
            rl_data_processing,
            "compute_advantages",
            _compute_miles_grpo_advantages,
        )
        if grpo_std_normalization
        else contextlib.nullcontext()
    )
    train_advantage_patch = (
        patch.object(
            rl_train,
            "compute_advantages",
            _compute_miles_grpo_advantages,
        )
        if grpo_std_normalization
        else contextlib.nullcontext()
    )
    with (
        patch.object(
            tinker.ServiceClient,
            "create_lora_training_client_async",
            create_training,
        ),
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
            tinker,
            "AdamParams",
            clipped_adam_params,
        ),
        patch.object(
            ml_log.JsonLogger,
            "log_metrics",
            timed_json_log_metrics,
        ),
        patch.object(
            math_train,
            "Config",
            dapo_config,
        ),
        patch.object(
            math_train,
            "get_dataset_builder",
            dapo_dataset_with_dynamic_sampling_headroom,
        ),
        patch.object(
            rl_rollouts,
            "TinkerTokenCompleter",
            partial(GroupedTinkerTokenCompleter, group_size=GROUP_SIZE),
        ),
        do_async_patch,
        data_advantage_patch,
        train_advantage_patch,
    ):
        await cli_main(config)


def run_benchmark(
    *,
    backend: str,
    parameterization: str,
    base_url: str | None,
    full: bool,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
    save_every: int = SAVE_EVERY,
    model_name: str = MODEL_NAME,
    renderer_name: str | None = None,
    rollout_mode: str = "cohort",
    grpo_std_normalization: bool = GRPO_STD_NORMALIZATION,
    remove_constant_reward_groups: bool = True,
    rollout: dict[str, int] | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"qwen-dapo-math-async-{backend}-") as temp:
        log_path = str(Path(temp) / "run")
        started_at = time.perf_counter()
        asyncio.run(
            _run_recipe(
                log_path,
                model_name=model_name,
                renderer_name=renderer_name,
                base_url=base_url,
                full=full,
                max_steps=max_steps,
                max_tokens=max_tokens,
                save_every=save_every,
                rollout_mode=rollout_mode,
                grpo_std_normalization=grpo_std_normalization,
                remove_constant_reward_groups=remove_constant_reward_groups,
                rollout=rollout,
            )
        )
        process_wall_time = time.perf_counter() - started_at
        metrics = [
            json.loads(line)
            for line in (Path(log_path) / "metrics.jsonl").read_text().splitlines()
        ]

    training_metrics = [
        metric
        for metric in metrics
        if "env/all/reward/total" in metric and "benchmark/wall_elapsed" in metric
    ]
    previous_completion = 0.0
    steps = []
    for metric in training_metrics:
        completion = metric["benchmark/wall_elapsed"]
        steps.append(
            {
                "step": metric["step"],
                "reward": metric["env/all/reward/total"],
                "correct": metric["env/all/correct"],
                "format": metric["env/all/format"],
                "mean_output_tokens": metric["env/all/ac_tokens_per_turn"],
                "step_time:seconds": completion - previous_completion,
                "sampling_time:seconds": metric["time/sampling_time_max"],
                "sampling_time_mean:seconds": metric["time/sampling_time_mean"],
                "train_publish_time:seconds": metric[
                    "time/do_train_step_and_get_sampling_client"
                ],
                "weight_publish_time:seconds": metric[
                    "time/save_checkpoint_and_get_sampling_client"
                ],
                "sampling_policy_step_min": metric["sampling_client/step_min"],
                "sampling_policy_step_max": metric["sampling_client/step_max"],
            }
        )
        previous_completion = completion
    if len(steps) != max_steps:
        raise RuntimeError(f"expected {max_steps} RL steps, got {len(steps)}")
    return {
        "backend": backend,
        "parameterization": parameterization,
        "config": experiment_config(
            max_steps=max_steps,
            max_tokens=max_tokens,
            save_every=save_every,
            model_name=model_name,
            renderer_name=renderer_name,
            rollout_mode=rollout_mode,
            grpo_std_normalization=grpo_std_normalization,
            remove_constant_reward_groups=remove_constant_reward_groups,
            rollout=rollout,
        ),
        "wall_time:seconds": training_metrics[-1]["benchmark/wall_elapsed"],
        "process_wall_time:seconds": process_wall_time,
        "steps": steps,
    }


__all__ = [
    "MAX_STEPS",
    "MAX_TOKENS",
    "SAVE_EVERY",
    "run_benchmark",
    "timestamped_output",
    "write_result",
]
