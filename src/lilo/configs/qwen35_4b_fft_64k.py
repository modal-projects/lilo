from lilo.deployments import BaseConfig


class Config(BaseConfig):
    name = "qwen35-4b-fft-64k"
    model = {
        "id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "parameterization": "full",
        "max_context_length": 65536,
    }
    routing = {"default": True}
    deployment = {"frontend": "lilo-yaml"}
    trainer = {
        "backend": "megatron",
        "resources": {"gpu": "H100:4"},
        "engine": {"max_clients_per_instance": 1, "sampler_persistence_concurrency": 1},
        "config": {
            "runtime": {
                "tensor_model_parallel_size": 2,
                "context_parallel_size": 2,
                "sequence_parallel": True,
                "micro_batch_size": 1,
                "max_tokens_per_microbatch": 65536,
                "defer_fp32_logits": True,
                "fp32_lm_head": True,
                "use_distributed_optimizer": True,
            },
            "provider": {
                "mtp_num_layers": 0,
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
            "optimizer": {"lr": 0.0001, "min_lr": 0.0001, "loss_scale": 1.0},
        },
    }
    inference = {
        "resources": {"gpu": "H100:1"},
        "config": {
            "tp_size": 1,
            "mem_fraction_static": 0.85,
            "max_running_requests": 32,
            "max_queued_requests": 4,
            "cpu_weight_cache_max_compile_group_gb": 16,
        },
    }
