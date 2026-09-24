from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-128k"
    max_context_length = 131072
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 4,
        "inference_gpus_per_node": 2,
        "inference_max_replicas": 4,
        "inference_target_concurrency": 4,
        "sglang_cfg.tp_size": 2,
        "sglang_cfg.max_running_requests": 8,
    }


config = Config()
