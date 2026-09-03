from __future__ import annotations

from megatron.core.optimizer import get_megatron_optimizer

from ..common.config import EngineModelConfig
from ..common.modeling import (
    distributed_model,
    model_provider,
    optimizer_config,
    parameter_dtype,
)


def create_fft_model_and_optimizer(config: EngineModelConfig):
    bridge, provider, _ = model_provider(config)
    provider.finalize()
    model = distributed_model(
        provider,
        config,
        distributed_optimizer=config.use_distributed_optimizer,
    )
    optimizer = create_fft_optimizer(config, model)
    return model, optimizer, bridge


def create_fft_optimizer(config: EngineModelConfig, model):
    return get_megatron_optimizer(
        config=optimizer_config(
            config,
            parameter_dtype(config),
            distributed_optimizer=config.use_distributed_optimizer,
        ),
        model_chunks=model,
    )
