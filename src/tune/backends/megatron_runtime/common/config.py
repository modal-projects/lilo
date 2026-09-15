from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    optimizer: str = "adam"
    lr: float = 1e-4
    min_lr: float = 1e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    clip_grad: float = 1.0
    loss_scale: float | None = None
    lr_decay_style: str = "constant"
    lr_warmup_iters: int = 0
    lr_decay_iters: int | None = None


@dataclass(frozen=True, slots=True)
class EngineModelConfig:
    hf_checkpoint: str
    max_tokens_per_microbatch: int | None = None

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    sequence_parallel: bool = False

    max_lora_slots: int = 16
    dynamo_recompile_limit: int = 64
    max_lora_rank: int = 32
    default_lora_alpha: int = 32
    lora_dropout: float = 0.0
    split_qkv: bool = False
    split_gdn: bool = False
    split_mamba: bool = False
    target_modules: tuple[str, ...] = (
        "decoder.*.linear_qkv",
        "decoder.*.linear_proj",
        "decoder.*.linear_fc1",
        "decoder.*.linear_fc2",
        "output_layer",
    )

    seed: int = 1234
    micro_batch_size: int = 1
    seq_length: int = 4096
    bf16: bool = True
    fp16: bool = False
    gpu_memory_fraction: float | None = None
    calculate_per_token_loss: bool = True
    attention_backend: str = "flash"
    cross_entropy_loss_fusion: bool = False
    defer_fp32_logits: bool = False
    fp32_lm_head: bool = False
    provider_overrides: dict[str, object] = field(default_factory=dict)

    overlap_grad_reduce: bool = False
    align_grad_reduce: bool = True
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = False

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def data_parallel_size(self, world_size: int) -> int:
        model_parallel_size = (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
        )
        if world_size % model_parallel_size != 0:
            raise ValueError(
                f"world size {world_size} is not divisible by model parallel size "
                f"{model_parallel_size}"
            )
        return world_size // model_parallel_size

    def has_lora_target(self, module_name: str) -> bool:
        return any(
            target.rsplit(".", 1)[-1] == module_name for target in self.target_modules
        )

    def sequence_padding_multiple(self) -> int:
        context_multiple = (
            2 * self.context_parallel_size if self.context_parallel_size > 1 else 1
        )
        sequence_multiple = (
            self.tensor_model_parallel_size * self.context_parallel_size
            if self.sequence_parallel
            else 1
        )
        return math.lcm(context_multiple, sequence_multiple)

    def packed_token_capacity(self) -> int:
        return (
            self.seq_length
            if self.max_tokens_per_microbatch is None
            else self.max_tokens_per_microbatch
        )

    def validate(self, world_size: int) -> None:
        parallel_sizes = {
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
        }
        for name, size in parallel_sizes.items():
            if size < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        if self.gpu_memory_fraction is not None and not (
            0 < self.gpu_memory_fraction <= 1
        ):
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be at least 1")
        if self.seq_length < 1:
            raise ValueError("seq_length must be at least 1")
        if (
            self.max_tokens_per_microbatch is not None
            and self.max_tokens_per_microbatch < 1
        ):
            raise ValueError("max_tokens_per_microbatch must be at least 1")
        if self.max_lora_slots < 1:
            raise ValueError("max_lora_slots must be at least 1")
        if self.max_lora_rank < 1:
            raise ValueError("max_lora_rank must be at least 1")
        if self.attention_backend not in {
            "auto",
            "flash",
            "fused",
            "local",
            "unfused",
        }:
            raise ValueError(f"unsupported attention backend: {self.attention_backend}")

        self.data_parallel_size(world_size)
