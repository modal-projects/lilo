from dataclasses import replace

from lilo.configs.qwen35_9b_lora_16k import config as base

config = replace(
    base,
    name="qwen35-9b-lora-16k-single",
    trainer=replace(base.trainer, max_clients_per_instance=1),
    routing=replace(base.routing, default=False),
)
