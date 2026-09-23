from dataclasses import replace

from lilo.configs.qwen38_27b_lora_16k import config as base

config = replace(
    base,
    name="qwen38-27b-lora-256k",
    model=replace(base.model, max_context_length=262144),
    routing=replace(base.routing, default=False),
    trainer=replace(
        base.trainer,
        compute=replace(base.trainer.compute, nodes=2),
        config={
            **base.trainer.config,
            "tensor_model_parallel_size": 2,
            "context_parallel_size": 8,
            "max_tokens_per_gpu": 32768,
            "cli_options": {
                **base.trainer.config["cli_options"],
                "distributed_timeout_minutes": 120,
            },
        },
    ),
    inference=replace(
        base.inference,
        compute=replace(base.inference.compute, gpus_per_node=4),
        min_replicas=2,
        max_replicas=2,
        target_concurrency=2,
        config={
            **base.inference.config,
            "tp_size": 4,
            "max_running_requests": 4,
            "max_queued_requests": 8,
            "max_loaded_loras": 256,
        },
    ),
)
