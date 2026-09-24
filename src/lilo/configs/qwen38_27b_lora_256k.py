from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-256k"
    max_context_length = 262144
    default = False
    overrides = {
        "trainer.nodes": 2,
        "trainer.config.tensor_model_parallel_size": 2,
        "trainer.config.context_parallel_size": 8,
        "trainer.config.max_tokens_per_gpu": 32768,
        "trainer.config.cli_options.distributed_timeout_minutes": 120,
        "inference.gpus_per_node": 4,
        "inference.min_replicas": 2,
        "inference.max_replicas": 2,
        "inference.target_concurrency": 2,
        "inference.config.tp_size": 4,
        "inference.config.max_running_requests": 4,
        "inference.config.max_queued_requests": 8,
        "inference.config.max_loaded_loras": 256,
    }


config = Config()
