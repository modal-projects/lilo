"""Code-only GRPO; historical reward-v3 remains available for comparisons."""

import dataclasses
import os

APP_NAME = os.environ.get("CODEGOLF_APP", "lilo-codegolf-example")
VOLUME_NAME = os.environ.get("CODEGOLF_VOLUME", APP_NAME)
DEFAULT_RUN = "golf"
DEFAULT_VARIANT = "async-v6"
DEFAULT_STEPS = 500
VARIANTS = ("async-v6", "async-v5", "reward-v3", "reward-v4")


@dataclasses.dataclass
class Config:
    steps: int = 500
    prompts_per_step: int = 4
    group_size: int = 8
    max_tokens: int = 16384
    learning_rate: float = 1e-6
    checkpoint_every: int = 50
    eval_every: int = 20
    eval_problems: int = 16
    seed: int = 42
    model: str = "Qwen/Qwen3.5-9B"
    reward_bonus: float = 0.15
    reward_scale: float = 2048
    output_token_penalty: float = 0.08
    output_token_scale: int = 16384
    advantage_std_floor: float = 0.5


@dataclasses.dataclass
class AsyncConfig(Config):
    async_rollouts: bool = True
    max_policy_lag: int = 4
    rollout_workers: int = 4
    buffer_batches: int = 4
    prefill_batches: int = 4
    rollout_min_replicas: int = 4
    rollout_max_replicas: int = 8
    judge_concurrency: int = 64


@dataclasses.dataclass
class TunedAsyncConfig(AsyncConfig):
    buffer_batches: int = 2
    prefill_batches: int = 2


def config_for(variant=DEFAULT_VARIANT, steps=DEFAULT_STEPS):
    if steps <= 0:
        raise ValueError("steps must be positive")
    if variant == "async-v6":
        return TunedAsyncConfig(steps=steps)
    if variant == "async-v5":
        return AsyncConfig(steps=steps)
    if variant == "reward-v4":
        return Config(steps=steps)
    if variant == "reward-v3":
        return Config(
            steps=steps, reward_bonus=0.1, reward_scale=256, output_token_penalty=0
        )
    raise ValueError(f"Unknown variant: {variant}")
