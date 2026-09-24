from lilo.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen38-27b-lora-16k"
    model = "Qwen/Qwen3.8-27B"
    max_context_length = 16384
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    miles_cfg = {
        "model_type": "qwen3.8-27B",
        "tensor_model_parallel_size": 4,
        "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        "max_tokens_per_gpu": 16384,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
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
        "ep_size": 1,
        "mem_fraction_static": 0.8,
        "max_running_requests": 32,
        "max_queued_requests": 8,
        "max_loaded_loras": 256,
        "max_loras_per_batch": 8,
        "schedule_policy": "lpm",
    }


config = Config()
