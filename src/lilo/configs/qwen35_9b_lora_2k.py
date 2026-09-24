from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-2k"
    max_context_length = 2048
    overrides = {
        "trainer.gpu": "H200",
        "trainer.cpu": 8,
        "trainer.memory_mib": 32768,
        "trainer.max_clients_per_instance": 4,
        "trainer.config.max_tokens_per_gpu": 2048,
        "trainer.config.max_lora_slots": 4,
        "inference.config.ep_size": 1,
        "inference.config.max_loaded_loras": 32,
        "inference.config.schedule_policy": "lpm",
    }


config = Config()
