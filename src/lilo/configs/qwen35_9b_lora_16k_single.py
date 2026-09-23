from dataclasses import dataclass
from lilo.configs.qwen35_9b_lora_16k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    name: str = "qwen35-9b-lora-16k-single"

    def __post_init__(self):
        self.routing.default = False
        self.trainer.engine.max_clients_per_instance = 1
