from lilo.configs.qwen35_35b_a3b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen36-35b-a3b-fft-64k"
    model = "Qwen/Qwen3.6-35B-A3B"
    overrides = {"sglang_cfg.dp_size": 1, "sglang_cfg.enable_dp_attention": False}


config = Config()
