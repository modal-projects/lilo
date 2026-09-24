from lilo.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-instruct-lora-16k-dp2"
    default = False
    overrides = {"trainer.config.tensor_model_parallel_size": 4}


config = Config()
