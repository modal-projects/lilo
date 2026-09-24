from lilo.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen36-27b-fft-64k"
    parameterization = "full"
    model = "Qwen/Qwen3.6-27B"
    max_context_length = 65536
    trainer = {
        "gpu": "H200",
        "gpus_per_node": 8,
        "backend": "megatron",
        "config": {
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 2,
            "sequence_parallel": True,
            "micro_batch_size": 1,
            "max_tokens_per_microbatch": 65536,
            "bf16": True,
            "fp16": False,
            "gpu_memory_fraction": 0.9,
            "use_distributed_optimizer": True,
            "provider_overrides": {
                "mtp_num_layers": 0,
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
            "optimizer": {"optimizer": "adam", "lr": 0.0001, "min_lr": 0.0001},
        },
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        "sampler_persistence_concurrency": 1,
    }
    inference = {
        "gpu": "H200",
        "gpus_per_node": 4,
        "config": {
            "tp_size": 4,
            "ep_size": 1,
            "mem_fraction_static": 0.9,
            "max_running_requests": 32,
            "max_queued_requests": 4,
            "cpu_weight_cache_max_compile_group_gb": 32,
        },
    }


config = Config()
