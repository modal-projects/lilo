from dataclasses import dataclass
from lilo.configs.qwen35_9b_lora_16k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    def __post_init__(self):
        self.model.revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
