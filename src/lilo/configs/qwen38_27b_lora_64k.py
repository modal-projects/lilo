from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-64k"
    max_context_length = 65536
    overrides = {
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 2,
        "inference_target_concurrency": 8,
        "sglang_cfg.max_running_requests": 16,
    }


config = Config()
