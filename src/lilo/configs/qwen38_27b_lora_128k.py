from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-128k"
    max_context_length = 131072
    trainer = {
        **Parent.trainer,
        "config": {
            **Parent.trainer["config"],
            "tensor_model_parallel_size": 2,
            "max_tokens_per_gpu": 32768,
            "context_parallel_size": 4,
        },
    }
    inference = {
        **Parent.inference,
        "gpus_per_node": 2,
        "max_replicas": 4,
        "target_concurrency": 4,
        "config": {
            **Parent.inference["config"],
            "tp_size": 2,
            "max_running_requests": 8,
        },
    }
    default = False


config = Config()
