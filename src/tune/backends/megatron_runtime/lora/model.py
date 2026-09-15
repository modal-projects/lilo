from __future__ import annotations

import torch
from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.utils import unwrap_model

from ..common.config import EngineModelConfig
from ..common.modeling import (
    distributed_model,
    model_provider,
    optimizer_config,
)


def create_model_and_optimizer(config: EngineModelConfig):
    bridge, provider, dtype = model_provider(config)
    source_ties_embeddings = provider.share_embeddings_and_output_weights
    needs_separate_output = source_ties_embeddings and config.has_lora_target(
        "output_layer"
    )
    provider.share_embeddings_and_output_weights = (
        source_ties_embeddings and not needs_separate_output
    )

    multi_lora_kwargs = {
        "target_modules": list(config.target_modules),
        "n_adapters": config.max_lora_slots,
        "dim": config.max_lora_rank,
        "alpha": config.default_lora_alpha,
        "dropout": config.lora_dropout,
        "lora_A_init_method": "kaiming",
        "split_qkv": config.split_qkv,
        "split_gdn": config.split_gdn,
        "split_mamba": config.split_mamba,
    }
    provider.register_pre_wrap_hook(MultiLoRA(**multi_lora_kwargs))

    provider.finalize()
    model = distributed_model(provider, config, distributed_optimizer=False)
    if needs_separate_output:
        _copy_embeddings_to_output(model)

    mcore_optimizer_config = optimizer_config(
        config,
        dtype,
        distributed_optimizer=False,
    )

    optimizers = {}
    for slot in range(config.max_lora_slots):
        with expose_adapter_slot(model, slot):
            optimizers[slot] = get_megatron_optimizer(
                config=mcore_optimizer_config,
                model_chunks=model,
            )

    return model, optimizers, bridge


def _copy_embeddings_to_output(
    model,
) -> None:
    for chunk in unwrap_model(model):
        roots = (chunk, getattr(chunk, "language_model", None))
        for root in roots:
            if root is None:
                continue
            embedding = getattr(
                getattr(root, "embedding", None),
                "word_embeddings",
                None,
            )
            output = getattr(root, "output_layer", None)
            output = getattr(output, "to_wrap", output)
            if embedding is not None and output is not None:
                with torch.no_grad():
                    output.weight.copy_(embedding.weight)
                return
    raise ValueError("untied output layer is not local to the embedding layer")
