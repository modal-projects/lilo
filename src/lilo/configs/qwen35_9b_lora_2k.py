from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-2k"
    max_context_length = 2048
    trainer = {
        **Parent.trainer,
        "gpu": "H200",
        "cpu": 8,
        "memory_mib": 32768,
        "max_clients_per_instance": 4,
        "config": {
            **Parent.trainer["config"],
            "max_tokens_per_gpu": 2048,
            "max_lora_slots": 4,
        },
    }
    inference = {
        **Parent.inference,
        "config": {
            **Parent.inference["config"],
            "ep_size": 1,
            "max_loaded_loras": 32,
            "schedule_policy": "lpm",
        },
    }
    default = False


config = Config()
