from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-256k"
    max_context_length = 262144
    default = False
    trainer = {
        **Parent.trainer,
        "nodes": 2,
        "config": {
            **Parent.trainer["config"],
            "tensor_model_parallel_size": 2,
            "context_parallel_size": 8,
            "max_tokens_per_gpu": 32768,
            "cli_options": {
                **Parent.trainer["config"]["cli_options"],
                "distributed_timeout_minutes": 120,
            },
        },
    }
    inference = {
        **Parent.inference,
        "gpus_per_node": 4,
        "min_replicas": 2,
        "max_replicas": 2,
        "target_concurrency": 2,
        "config": {
            **Parent.inference["config"],
            "tp_size": 4,
            "max_running_requests": 4,
            "max_queued_requests": 8,
            "max_loaded_loras": 256,
        },
    }


config = Config()
