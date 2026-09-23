from dataclasses import dataclass, field
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


@dataclass(kw_only=True)
class Config(BaseConfig):
    name: str = "qwen35-9b-fft-64k"
    model: Model = field(
        default_factory=lambda: Model(
            id="Qwen/Qwen3.5-9B", parameterization="full", max_context_length=65536
        )
    )
    routing: Routing = field(default_factory=lambda: Routing(default=True))
    trainer: Trainer = field(
        default_factory=lambda: Trainer(
            backend="megatron",
            resources=Resources(gpu="H200:4"),
            engine=EngineOptions(
                max_clients_per_instance=1, sampler_persistence_concurrency=1
            ),
            env={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
            },
            config={
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
        )
    )
    inference: Inference = field(
        default_factory=lambda: Inference(
            resources=Resources(gpu="H200:1"),
            scaling=InferenceScaling(
                min_replicas=0, max_replicas=8, target_concurrency=16
            ),
            config={
                "tp_size": 1,
                "ep_size": 1,
                "mem_fraction_static": 0.85,
                "max_running_requests": 32,
                "max_queued_requests": 4,
                "cpu_weight_cache_max_compile_group_gb": 16,
            },
        )
    )
