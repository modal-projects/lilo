from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-128k"
    max_context_length = 131072
    default = False
    overrides = {
        "trainer.config.tensor_model_parallel_size": 2,
        "trainer.config.max_tokens_per_gpu": 32768,
        "trainer.config.context_parallel_size": 4,
        "inference.gpus_per_node": 2,
        "inference.max_replicas": 4,
        "inference.target_concurrency": 4,
        "inference.config.tp_size": 2,
        "inference.config.max_running_requests": 8,
    }


config = Config()
