from dataclasses import replace

from lilo.configs.qwen35_9b_lora_16k import config as base

config = replace(
    base,
    name="qwen35-9b-lora-64k",
    model=replace(base.model, max_context_length=65536),
    trainer=replace(
        base.trainer,
        compute=replace(base.trainer.compute, gpu="H200", gpus_per_node=8),
        config={
            **base.trainer.config,
            "tensor_model_parallel_size": 8,
            "max_tokens_per_gpu": 65536,
        },
    ),
    routing=replace(base.routing, default=False),
)
