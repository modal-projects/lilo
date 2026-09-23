from dataclasses import replace

from lilo.configuration import Routing
from lilo.configs.qwen35_9b_lora_16k import config as base

config = replace(
    base,
    name="qwen35-9b-lora-2k",
    model=replace(base.model, max_context_length=2048),
    trainer=replace(
        base.trainer,
        compute=replace(base.trainer.compute, gpu="H200", cpu=8, memory_mib=32768),
        max_clients_per_instance=4,
        config={**base.trainer.config, "max_tokens_per_gpu": 2048, "max_lora_slots": 4},
    ),
    inference=replace(
        base.inference,
        config={
            **base.inference.config,
            "ep_size": 1,
            "max_loaded_loras": 32,
            "schedule_policy": "lpm",
        },
    ),
    routing=Routing(),
)
