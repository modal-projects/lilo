from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-64k"
    max_context_length = 65536
    overrides = {
        "trainer.config.max_tokens_per_gpu": 32768,
        "trainer.config.context_parallel_size": 2,
        "inference.target_concurrency": 8,
        "inference.config.max_running_requests": 16,
    }


config = Config()
