from lilo.configs.qwen35_35b_a3b_fft_64k import Config as ParentConfig


class Config(ParentConfig):
    name = "qwen36-35b-a3b-fft-64k"
    overrides = {
        "model.id": "Qwen/Qwen3.6-35B-A3B",
        "inference.config.dp_size": 1,
        "inference.config.enable_dp_attention": False,
    }
