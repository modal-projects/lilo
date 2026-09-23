from dataclasses import replace

from lilo.configs.qwen38_27b_lora_16k import config as base

config = replace(
    base,
    name="qwen38-27b-lora-128k",
    model=replace(base.model, max_context_length=131072),
    trainer=replace(
        base.trainer,
        config={
            **base.trainer.config,
            "tensor_model_parallel_size": 2,
            "max_tokens_per_gpu": 32768,
            "context_parallel_size": 4,
        },
    ),
    inference=replace(
        base.inference,
        compute=replace(base.inference.compute, gpus_per_node=2),
        max_replicas=4,
        target_concurrency=4,
        config={**base.inference.config, "tp_size": 2, "max_running_requests": 8},
    ),
    routing=replace(base.routing, default=False),
)
