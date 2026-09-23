from lilo.configs.qwen35_9b_lora_16k import Config as ParentConfig


class Config(ParentConfig):
    name = "qwen35-9b-lora-16k-single"
    overrides = {"routing.default": False, "trainer.engine.max_clients_per_instance": 1}
