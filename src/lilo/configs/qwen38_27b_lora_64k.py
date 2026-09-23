from dataclasses import replace

from lilo.configs.qwen38_27b_lora_16k import config as base

config = replace(
    base,
    name="qwen38-27b-lora-64k",
    model=replace(base.model, max_context_length=65536),
    trainer=replace(
        base.trainer,
        config={
            **base.trainer.config,
            "max_tokens_per_gpu": 32768,
            "context_parallel_size": 2,
        },
    ),
    inference=replace(
        base.inference,
        target_concurrency=8,
        config={**base.inference.config, "max_running_requests": 16},
    ),
    routing=replace(base.routing, default=False),
)
