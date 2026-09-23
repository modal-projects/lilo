from dataclasses import dataclass
from lilo.configs.qwen35_35b_a3b_fft_64k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    name: str = "qwen36-35b-a3b-fft-64k"

    def __post_init__(self):
        self.model.id = "Qwen/Qwen3.6-35B-A3B"
        self.inference.config["dp_size"] = 1
        self.inference.config["enable_dp_attention"] = False
