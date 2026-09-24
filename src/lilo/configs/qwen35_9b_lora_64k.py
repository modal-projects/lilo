from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-64k"
    max_context_length = 65536
    overrides = {
        "trainer_gpu": "H200",
        "trainer_gpus_per_node": 8,
        "miles_cfg.tensor_model_parallel_size": 8,
        "miles_cfg.max_tokens_per_gpu": 65536,
    }


config = Config()
