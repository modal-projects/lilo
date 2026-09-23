from dataclasses import replace

from lilo.configs.qwen35_9b_instruct_lora_16k import config as base

config = replace(
    base,
    name="qwen35-9b-instruct-lora-16k-dp2",
    trainer=replace(
        base.trainer, config={**base.trainer.config, "tensor_model_parallel_size": 4}
    ),
    routing=replace(base.routing, default=False),
)
