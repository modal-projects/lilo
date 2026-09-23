from dataclasses import dataclass
from lilo.configs.qwen35_9b_lora_64k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    def __post_init__(self):
        super().__post_init__()
        self.model.revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
