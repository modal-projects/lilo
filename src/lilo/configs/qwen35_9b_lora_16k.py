from lilo.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen35-9b-lora-16k"
    model = "Qwen/Qwen3.5-9B-Base"
    max_context_length = 16384
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_cpu = 16
    trainer_memory_mib = 65536
    miles_cfg = {
        "model_type": "qwen3.5-9B",
        "tensor_model_parallel_size": 4,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "target_modules": [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "output_layer",
        ],
        "max_tokens_per_gpu": 16384,
        "cli_options": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    trainer_max_clients_per_instance = 6
    inference_gpu = "H200"
    sglang_cfg = {
        "tp_size": 1,
        "mem_fraction_static": 0.8,
        "max_running_requests": 32,
        "max_queued_requests": 8,
        "max_loaded_loras": 64,
        "max_loras_per_batch": 8,
    }


config = Config()
