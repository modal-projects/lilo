"""Complete engine recipes; importing these does not register a Modal App."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from lilo.backends.megatron_runtime.common.config import (
    EngineModelConfig,
    OptimizerConfig,
)


@dataclass(frozen=True)
class SamplingConfig:
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    memory_fraction: float = 0.85
    max_running_requests: int = 32
    max_queued_requests: int = 4
    target_concurrency: int = 16
    cpu_weight_cache_max_compile_group_gb: float = 16
    schedule_policy: str = "lpm"


@dataclass(frozen=True)
class Engine:
    """One explicit FFT recipe. Resource/context changes require a new recipe.

    Images are optional ordinary modal.Image objects. Custom runtime packages can
    be installed into them; no Lilo registry edit or repository checkout is needed.
    """

    name: str
    model: str
    trainer_gpu: str
    sampler_gpu: str
    training: EngineModelConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    revision: str | None = None
    trainer_image: Any = field(default=None, repr=False, compare=False)
    sampler_image: Any = field(default=None, repr=False, compare=False)
    trainer_cpu: float | tuple[float, float] | None = None
    trainer_memory: int | tuple[int, int] | None = None
    sampler_cpu: float | tuple[float, float] | None = None
    sampler_memory: int | tuple[int, int] | None = None
    trainer_timeout: int = 86_400
    backend_env: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.model:
            raise ValueError("engine name and model are required")
        self.training.validate(gpu_count(self.trainer_gpu))
        world = gpu_count(self.sampler_gpu)
        for value in (
            self.sampling.tensor_parallel_size,
            self.sampling.expert_parallel_size
            * self.sampling.expert_tensor_parallel_size,
        ):
            if value < 1 or world % value:
                raise ValueError("sampler parallelism must divide sampler GPU count")
        if self.trainer_timeout < 1 or self.trainer_timeout > 86_400:
            raise ValueError("trainer_timeout must be in 1..86400 seconds")
        if any(key.startswith("LILO_") for key in self.backend_env):
            raise ValueError("LILO_ environment variables are owned by the runtime")


def gpu_count(value: str) -> int:
    _, sep, count = value.partition(":")
    result = int(count) if sep else 1
    if result < 1:
        raise ValueError("GPU count must be positive")
    return result


def qwen3_5_4b_full_64k() -> Engine:
    return Engine(
        name="qwen3_5_4b_full_64k",
        model="Qwen/Qwen3.5-4B",
        trainer_gpu="H100:4",
        sampler_gpu="H100:1",
        training=EngineModelConfig(
            hf_checkpoint="",
            seq_length=65_536,
            max_tokens_per_microbatch=65_536,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
            defer_fp32_logits=True,
            fp32_lm_head=True,
            use_distributed_optimizer=True,
            provider_overrides={
                "mtp_num_layers": 0,
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
            optimizer=OptimizerConfig(loss_scale=1.0),
        ),
    )


def qwen3_6_27b_full_64k() -> Engine:
    original = qwen3_5_4b_full_64k()
    return replace(
        original,
        name="qwen3_6_27b_full_64k",
        model="Qwen/Qwen3.6-27B",
        trainer_gpu="H200:8",
        sampler_gpu="H200:4",
        training=replace(
            original.training,
            tensor_model_parallel_size=4,
            defer_fp32_logits=False,
            fp32_lm_head=False,
            gpu_memory_fraction=0.90,
            optimizer=OptimizerConfig(),
        ),
        sampling=SamplingConfig(
            tensor_parallel_size=4,
            memory_fraction=0.90,
            cpu_weight_cache_max_compile_group_gb=32,
        ),
    )


def qwen3_5_9b_full_64k() -> Engine:
    """8×H200 full fine-tuning with TP2, CP2, DP2, and 64K context."""
    original = qwen3_5_4b_full_64k()
    return replace(
        original,
        name="qwen3_5_9b_full_64k",
        model="Qwen/Qwen3.5-9B",
        trainer_gpu="H200:8",
        sampler_gpu="H200:1",
    )
