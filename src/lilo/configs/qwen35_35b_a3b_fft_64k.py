from lilo.deployments import (
    BaseConfig,
    EngineOptions,
    Inference,
    InferenceScaling,
    Model,
    Resources,
    Routing,
    Trainer,
)


class Config(BaseConfig):
    name = "qwen35-35b-a3b-fft-64k"
    model = Model(
        id="Qwen/Qwen3.5-35B-A3B", parameterization="full", max_context_length=65536
    )
    routing = Routing(default=True)
    trainer = Trainer(
        backend="megatron",
        resources=Resources(gpu="H200:8"),
        engine=EngineOptions(
            max_clients_per_instance=1, sampler_persistence_concurrency=1
        ),
        env={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        config={
            "runtime": {
                "tensor_model_parallel_size": 4,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 2,
                "expert_model_parallel_size": 8,
                "expert_tensor_parallel_size": 1,
                "sequence_parallel": True,
                "micro_batch_size": 1,
                "max_tokens_per_microbatch": 65536,
                "bf16": True,
                "fp16": False,
                "gpu_memory_fraction": 0.9,
                "use_distributed_optimizer": True,
            },
            "provider": {
                "mtp_num_layers": 0,
                "recompute_granularity": "selective",
                "moe_layer_recompute": True,
                "moe_token_dispatcher_type": "alltoall",
                "moe_router_fusion": True,
                "moe_permute_fusion": True,
                "moe_grouped_gemm": True,
                "moe_shared_expert_overlap": False,
                "moe_aux_loss_coeff": 0.0,
            },
            "optimizer": {"optimizer": "adam", "lr": 0.0001, "min_lr": 0.0001},
        },
    )
    inference = Inference(
        resources=Resources(gpu="H200:4"),
        scaling=InferenceScaling(min_replicas=0, max_replicas=8, target_concurrency=16),
        config={
            "tp_size": 4,
            "ep_size": 4,
            "mem_fraction_static": 0.9,
            "max_running_requests": 32,
            "max_queued_requests": 4,
            "cpu_weight_cache_max_compile_group_gb": 32,
            "dp_size": 4,
            "enable_dp_attention": True,
        },
    )
