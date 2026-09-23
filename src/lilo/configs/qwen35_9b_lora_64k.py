from dataclasses import dataclass
from lilo.configs.qwen35_9b_lora_16k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    name: str = "qwen35-9b-lora-64k"

    def __post_init__(self):
        self.model.max_context_length = 65536
        self.routing.default = False
        self.trainer.resources.gpu = "H200:8"
        self.trainer.config["options"]["tensor_model_parallel_size"] = 8
        self.trainer.config["options"]["max_tokens_per_gpu"] = 65536
