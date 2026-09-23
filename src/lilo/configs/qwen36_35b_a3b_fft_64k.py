from dataclasses import replace

from lilo.configs.qwen35_35b_a3b_fft_64k import config as base

config = replace(
    base,
    name="qwen36-35b-a3b-fft-64k",
    model=replace(base.model, id="Qwen/Qwen3.6-35B-A3B"),
    inference=replace(
        base.inference,
        config={**base.inference.config, "dp_size": 1, "enable_dp_attention": False},
    ),
)
