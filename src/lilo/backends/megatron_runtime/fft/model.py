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
    if config.fp32_lm_head:
        for chunk in model:
            apply_fp32_lm_head(chunk)
    optimizer = create_fft_optimizer(config, model)
    return model, optimizer, bridge


def apply_fp32_lm_head(module) -> None:
    for name, layer in module.named_modules():
        if not name.endswith("output_layer"):
            continue
        impl = layer._forward_impl

        def forward_impl(
            *, input, weight, bias, gradient_accumulation_fusion, _impl=impl, **kwargs
        ):
            return _impl(
                input=input.float(),
                weight=weight.float(),
                bias=None if bias is None else bias.float(),
                gradient_accumulation_fusion=False,
                **kwargs,
            )

        layer._forward_impl = forward_impl


def create_fft_optimizer(config: EngineModelConfig, model):
    return get_megatron_optimizer(
        config=optimizer_config(
            config,
            parameter_dtype(config),
            distributed_optimizer=config.use_distributed_optimizer,
        ),
        model_chunks=model,
    )
