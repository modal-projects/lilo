from __future__ import annotations

import torch
from megatron.bridge import AutoBridge
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.transformer.enums import AttnBackend

from .config import EngineModelConfig


def model_provider(config: EngineModelConfig):
    dtype = parameter_dtype(config)
    bridge = AutoBridge.from_hf_pretrained(
        config.hf_checkpoint,
        trust_remote_code=True,
    )
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = config.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = config.pipeline_model_parallel_size
    provider.virtual_pipeline_model_parallel_size = (
        config.virtual_pipeline_model_parallel_size
    )
    provider.context_parallel_size = config.context_parallel_size
    provider.expert_model_parallel_size = config.expert_model_parallel_size
    provider.expert_tensor_parallel_size = config.expert_tensor_parallel_size
    provider.sequence_parallel = config.sequence_parallel
    provider.variable_seq_lengths = True
    if getattr(provider, "moe_token_dispatcher_type", None) == "allgather":
        provider.moe_token_dispatcher_type = "alltoall"
    provider.calculate_per_token_loss = config.calculate_per_token_loss
    provider.attention_backend = AttnBackend[config.attention_backend]
    provider.cross_entropy_loss_fusion = config.cross_entropy_loss_fusion
    provider.params_dtype = dtype
    for name, value in config.provider_overrides.items():
        if not hasattr(provider, name):
            raise ValueError(f"unknown Megatron provider override: {name}")
        setattr(provider, name, value)
    return bridge, provider, dtype


def parameter_dtype(config: EngineModelConfig):
    return (
        torch.bfloat16
        if config.bf16
        else torch.float16
        if config.fp16
        else torch.float32
    )


def distributed_model(
    provider,
    config: EngineModelConfig,
    *,
    distributed_optimizer: bool,
):
    return provider.provide_distributed_model(
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=distributed_optimizer,
            overlap_grad_reduce=config.overlap_grad_reduce,
            overlap_param_gather=config.overlap_param_gather,
            align_param_gather=config.align_param_gather,
        ),
        bf16=config.bf16,
        fp16=config.fp16,
    )


def optimizer_config(
    config: EngineModelConfig,
    dtype,
    *,
    distributed_optimizer: bool,
):
    return MCoreOptimizerConfig(
        optimizer=config.optimizer.optimizer,
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        adam_beta1=config.optimizer.adam_beta1,
        adam_beta2=config.optimizer.adam_beta2,
        adam_eps=config.optimizer.adam_eps,
        clip_grad=config.optimizer.clip_grad,
        loss_scale=config.optimizer.loss_scale,
        bf16=config.bf16,
        fp16=config.fp16,
        params_dtype=dtype,
        use_distributed_optimizer=distributed_optimizer,
        overlap_param_gather=config.overlap_param_gather,
    )
