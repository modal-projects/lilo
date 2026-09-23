from dataclasses import dataclass
from lilo.configs.qwen38_27b_lora_16k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    name: str = "qwen38-27b-lora-128k"

    def __post_init__(self):
        self.model.max_context_length = 131072
        self.routing.default = False
        self.trainer.config["options"]["tensor_model_parallel_size"] = 2
        self.trainer.config["options"]["context_parallel_size"] = 4
        self.trainer.config["options"]["max_tokens_per_gpu"] = 32768
        self.inference.resources.gpu = "H200:2"
        self.inference.scaling.max_replicas = 4
        self.inference.scaling.target_concurrency = 4
        self.inference.config["tp_size"] = 2
        self.inference.config["max_running_requests"] = 8
