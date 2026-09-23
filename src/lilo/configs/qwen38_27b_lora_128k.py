from lilo.configs.qwen38_27b_lora_16k import Config as ParentConfig


class Config(ParentConfig):
    name = "qwen38-27b-lora-128k"
    overrides = {
        "model.max_context_length": 131072,
        "routing.default": False,
        "trainer.config.options.tensor_model_parallel_size": 2,
        "trainer.config.options.context_parallel_size": 4,
        "trainer.config.options.max_tokens_per_gpu": 32768,
        "inference.resources.gpu": "H200:2",
        "inference.scaling.max_replicas": 4,
        "inference.scaling.target_concurrency": 4,
        "inference.config.tp_size": 2,
        "inference.config.max_running_requests": 8,
    }
