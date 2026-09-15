"""Code-only GRPO and TailRL configurations."""

import dataclasses
import os

APP_NAME = os.environ.get("CODEGOLF_APP", "lilo-codegolf-example")
VOLUME_NAME = os.environ.get("CODEGOLF_VOLUME", APP_NAME)
DEFAULT_RUN = "golf"
DEFAULT_VARIANT = "async-v6"
DEFAULT_STEPS = 500
VARIANTS = ("tailrl", "async-v7", "async-v6", "async-v5", "reward-v3", "reward-v4")
ADVANTAGE_ESTIMATORS = ("grpo", "tailrl")


def with_config_defaults(config):
    """Interpret specs saved before estimator selection and multi-sample eval."""
    return {"advantage_estimator": "grpo", "eval_samples": 1, **config}


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
    eval_samples: int = 1
    seed: int = 42
    model: str = "Qwen/Qwen3.5-9B"
    reward_bonus: float = 0.15
    reward_scale: float = 2048
    output_token_penalty: float = 0.08
    output_token_scale: int = 16384
    advantage_std_floor: float = 0.5
    advantage_estimator: str = "grpo"

    def __post_init__(self):
        if self.advantage_estimator not in ADVANTAGE_ESTIMATORS:
            raise ValueError(f"Unknown advantage estimator: {self.advantage_estimator}")
        if not isinstance(self.eval_samples, int) or self.eval_samples < 1:
            raise ValueError("eval_samples must be a positive integer")


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


@dataclasses.dataclass
class StrongGolfConfig(TunedAsyncConfig):
    reward_bonus: float = 0.30
    output_token_penalty: float = 0.20


@dataclasses.dataclass
class TailRLConfig(TunedAsyncConfig):
    advantage_estimator: str = "tailrl"
    eval_samples: int = 8


def config_for(variant=DEFAULT_VARIANT, steps=DEFAULT_STEPS, *, eval_samples=None):
    if steps <= 0:
        raise ValueError("steps must be positive")
    if variant == "tailrl":
        config = TailRLConfig(steps=steps)
    elif variant == "async-v7":
        config = StrongGolfConfig(steps=steps)
    elif variant == "async-v6":
        config = TunedAsyncConfig(steps=steps)
    elif variant == "async-v5":
        config = AsyncConfig(steps=steps)
    elif variant == "reward-v4":
        config = Config(steps=steps)
    elif variant == "reward-v3":
        config = Config(
            steps=steps, reward_bonus=0.1, reward_scale=256, output_token_penalty=0
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if eval_samples is not None:
        config = dataclasses.replace(config, eval_samples=eval_samples)
    return config
