from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-16k-single"
    trainer = {**Parent.trainer, "max_clients_per_instance": 1}
    default = False


config = Config()
