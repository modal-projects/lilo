from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MILES_REVISION = "e1f608a6b335b09e966293757492627cf22cb5ec"

_PEFT_TARGETS = {
    "linear_qkv": ("q_proj", "k_proj", "v_proj"),
    "linear_q": ("q_proj",),
    "linear_k": ("k_proj",),
    "linear_v": ("v_proj",),
    "linear_proj": ("o_proj",),
    "linear_fc1": ("gate_proj", "up_proj"),
    "linear_fc1_gate": ("gate_proj",),
    "linear_fc1_up": ("up_proj",),
    "linear_fc2": ("down_proj",),
    "output_layer": ("lm_head",),
}


@dataclass(frozen=True, slots=True)
class MilesBackendConfig:
    """Stable Lilo configuration translated to Miles CLI arguments at startup."""

    hf_checkpoint: str
    model_type: str
    actor_num_gpus_per_node: int
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    max_lora_slots: int = 8
    max_lora_rank: int = 32
    default_lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "output_layer",
    )
    max_tokens_per_gpu: int = 8192
    extra_args: tuple[str, ...] = ()

    @property
    def world_size(self) -> int:
        return self.actor_num_gpus_per_node

    @property
    def peft_target_modules(self) -> tuple[str, ...]:
        targets: list[str] = []
        for target in self.target_modules:
            leaf = target.rsplit(".", 1)[-1]
            for name in _PEFT_TARGETS.get(leaf, (leaf,)):
                if name not in targets:
                    targets.append(name)
        return tuple(targets)

    def validate(self) -> None:
        positive = {
            "actor_num_gpus_per_node": self.actor_num_gpus_per_node,
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
            "max_lora_slots": self.max_lora_slots,
            "max_lora_rank": self.max_lora_rank,
            "max_tokens_per_gpu": self.max_tokens_per_gpu,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not self.hf_checkpoint:
            raise ValueError("hf_checkpoint is required")
        if not self.model_type:
            raise ValueError("model_type is required")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if (
            self.default_lora_alpha <= 0
            or not float(self.default_lora_alpha).is_integer()
        ):
            raise ValueError("default_lora_alpha must be a positive integer")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.world_size != self.tensor_model_parallel_size:
            raise ValueError(
                "MilesCommandBackend currently requires data parallel size 1; "
                "actor_num_gpus_per_node must equal tensor_model_parallel_size"
            )

    def miles_arguments(self) -> list[str]:
        """Arguments owned by the integration, excluding model architecture flags."""

        arguments = [
            "--train-backend",
            "megatron",
            "--hf-checkpoint",
            self.hf_checkpoint,
            "--load",
            self.hf_checkpoint,
            "--pretrained-checkpoint",
            self.hf_checkpoint,
            "--megatron-to-hf-mode",
            "bridge",
            "--debug-train-only",
            "--rollout-num-gpus",
            "0",
            "--actor-num-nodes",
            "1",
            "--actor-num-gpus-per-node",
            str(self.actor_num_gpus_per_node),
            "--multi-lora-n-adapters",
            str(self.max_lora_slots),
            "--lora-rank",
            str(self.max_lora_rank),
            "--lora-alpha",
            str(int(self.default_lora_alpha)),
            "--lora-dropout",
            str(self.lora_dropout),
            "--target-modules",
            ",".join(self.target_modules),
            "--no-gradient-accumulation-fusion",
            "--optimizer",
            "adam",
            "--lr",
            "1e-4",
            "--tensor-model-parallel-size",
            str(self.tensor_model_parallel_size),
            "--pipeline-model-parallel-size",
            "1",
            "--context-parallel-size",
            "1",
            "--expert-model-parallel-size",
            str(self.expert_model_parallel_size),
            "--expert-tensor-parallel-size",
            str(self.expert_tensor_parallel_size),
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu",
            str(self.max_tokens_per_gpu),
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend",
            "flash",
        ]
        if self.tensor_model_parallel_size > 1:
            arguments.append("--sequence-parallel")
        return [*arguments, *self.extra_args]


def parse_backend_config(
    value: dict[str, Any],
) -> tuple[MilesBackendConfig, Path, Path]:
    miles = dict(value.get("miles") or value)
    for name in ("target_modules", "extra_args"):
        if name in miles:
            miles[name] = tuple(miles[name])
    config = MilesBackendConfig(**miles)
    config.validate()
    checkpoint_dir = Path(value.get("checkpoint_dir") or "/tmp/lilo")
    capture_dir = Path(value.get("capture_dir") or "/tmp/lilo-miles-captures")
    return config, checkpoint_dir, capture_dir
