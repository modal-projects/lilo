from __future__ import annotations

from dataclasses import replace

import torch
from megatron.bridge import AutoBridge
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.transformer.enums import AttnBackend

from .config import EngineModelConfig
from .settings import distributed_settings, optimizer_settings, provider_settings


def model_provider(config: EngineModelConfig):
    dtype = parameter_dtype(config)
    bridge = AutoBridge.from_hf_pretrained(
        config.hf_checkpoint,
        trust_remote_code=True,
    )
    settings = provider_settings(config, dtype)
    provider = replace(
        bridge.to_megatron_provider(),
        **{**settings, "attention_backend": AttnBackend[config.attention_backend]},
    )
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
            **distributed_settings(config, distributed_optimizer)
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
        **optimizer_settings(config, dtype, distributed_optimizer)
    )
