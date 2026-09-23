from lilo.configs.qwen35_9b_lora_16k import Config as ParentConfig


class Config(ParentConfig):
    name = "qwen35-9b-lora-64k"
    overrides = {
        "model.max_context_length": 65536,
        "routing.default": False,
        "trainer.resources.gpu": "H200:8",
        "trainer.config.options.tensor_model_parallel_size": 8,
        "trainer.config.options.max_tokens_per_gpu": 65536,
    }
