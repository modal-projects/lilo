from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-256k"
    max_context_length = 262144
    overrides = {
        "trainer_nodes": 2,
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.context_parallel_size": 8,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.cli_options.distributed_timeout_minutes": 120,
        "inference_gpus_per_node": 4,
        "inference_min_replicas": 2,
        "inference_max_replicas": 2,
        "inference_target_concurrency": 2,
        "sglang_cfg.tp_size": 4,
        "sglang_cfg.max_running_requests": 4,
        "sglang_cfg.max_queued_requests": 8,
        "sglang_cfg.max_loaded_loras": 256,
    }


config = Config()
