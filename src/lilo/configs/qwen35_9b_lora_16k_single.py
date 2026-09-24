from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-16k-single"
    overrides = {"trainer.max_clients_per_instance": 1}


config = Config()
