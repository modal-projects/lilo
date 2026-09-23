from lilo.configs.qwen38_27b_lora_16k import Config as ParentConfig


class Config(ParentConfig):
    name = "qwen38-27b-lora-64k"
    overrides = {
        "model.max_context_length": 65536,
        "routing.default": False,
        "trainer.config.context_parallel_size": 2,
        "trainer.config.max_tokens_per_gpu": 32768,
        "inference.scaling.target_concurrency": 8,
        "inference.config.max_running_requests": 16,
    }
