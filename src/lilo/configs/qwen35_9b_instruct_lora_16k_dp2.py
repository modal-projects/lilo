from dataclasses import dataclass
from lilo.configs.qwen35_9b_instruct_lora_16k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    name: str = "qwen35-9b-instruct-lora-16k-dp2"

    def __post_init__(self):
        self.routing.default = False
        self.trainer.config["options"]["tensor_model_parallel_size"] = 4
