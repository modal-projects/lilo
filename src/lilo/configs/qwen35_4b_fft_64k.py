from lilo.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen35-4b-fft-64k"
    parameterization = "full"
    model = "Qwen/Qwen3.5-4B"
    max_context_length = 65536
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    backend = "megatron"
    megatron_cfg = {
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "micro_batch_size": 1,
        "max_tokens_per_microbatch": 65536,
        "defer_fp32_logits": True,
        "fp32_lm_head": True,
        "use_distributed_optimizer": True,
        "provider_overrides": {
            "mtp_num_layers": 0,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
        "optimizer": {"lr": 0.0001, "min_lr": 0.0001, "loss_scale": 1.0},
    }
    sampler_persistence_concurrency = 1
    inference_gpu = "H100"
    sglang_cfg = {
        "tp_size": 1,
        "mem_fraction_static": 0.85,
        "max_running_requests": 32,
        "max_queued_requests": 4,
        "cpu_weight_cache_max_compile_group_gb": 16,
    }


config = Config()
