from lilo.configs.qwen35_9b_instruct_lora_16k import Config as ParentConfig


class Config(ParentConfig):
    name = "qwen35-9b-instruct-lora-16k-dp2"
    overrides = {
        "routing.default": False,
        "trainer.config.options.tensor_model_parallel_size": 4,
    }
