from lilo.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-64k"
    max_context_length = 65536
    trainer = {
        **Parent.trainer,
        "config": {
            **Parent.trainer["config"],
            "max_tokens_per_gpu": 32768,
            "context_parallel_size": 2,
        },
    }
    inference = {
        **Parent.inference,
        "target_concurrency": 8,
        "config": {**Parent.inference["config"], "max_running_requests": 16},
    }
    default = False


config = Config()
