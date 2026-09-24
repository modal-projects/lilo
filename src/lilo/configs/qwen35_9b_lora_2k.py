from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-2k"
    max_context_length = 2048
    overrides = {
        "trainer_gpu": "H200",
        "trainer_cpu": 8,
        "trainer_memory_mib": 32768,
        "trainer_max_clients_per_instance": 4,
        "miles_cfg.max_tokens_per_gpu": 2048,
        "miles_cfg.max_lora_slots": 4,
        "sglang_cfg.ep_size": 1,
        "sglang_cfg.max_loaded_loras": 32,
        "sglang_cfg.schedule_policy": "lpm",
    }


config = Config()
