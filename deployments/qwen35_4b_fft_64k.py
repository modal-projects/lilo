from dataclasses import dataclass
from lilo.configs.qwen35_4b_fft_64k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    def __post_init__(self):
        self.model.revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
