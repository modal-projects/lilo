from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeepSpeedOptimizerConfig:
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class DeepSpeedBackendConfig:
    hf_checkpoint: str
    zero_stage: int = 2
    micro_batch_size: int = 1
    max_sequence_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    optimizer: DeepSpeedOptimizerConfig = field(
        default_factory=DeepSpeedOptimizerConfig
    )

    def __post_init__(self) -> None:
        if self.zero_stage not in {1, 2}:
            raise ValueError("minimal DeepSpeed backend supports ZeRO stages 1 and 2")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")

    def engine_config(self, world_size: int) -> dict[str, Any]:
        return {
            "train_micro_batch_size_per_gpu": self.micro_batch_size,
            "train_batch_size": self.micro_batch_size * world_size,
            "gradient_accumulation_steps": 1,
            "managed_gradient_accumulation": False,
            "bf16": {"enabled": self.bf16},
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": self.zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": self.zero_stage == 2,
                "reduce_scatter": self.zero_stage == 2,
            },
        }


def parse_deepspeed_backend_config(
    value: dict[str, Any],
) -> tuple[DeepSpeedBackendConfig, Path]:
    raw = dict(value.get("deepspeed") or value)
    optimizer = DeepSpeedOptimizerConfig(**raw.pop("optimizer", {}))
    checkpoint_dir = Path(value.get("checkpoint_dir", "/checkpoints"))
    return DeepSpeedBackendConfig(**raw, optimizer=optimizer), checkpoint_dir
