from dataclasses import dataclass, field
from lilo.deployments import (
    BaseConfig,
    Deployment,
    EngineOptions,
    Inference,
    InferenceScaling,
    Model,
    Resources,
    Routing,
    Trainer,
    TrainerScaling,
)


@dataclass(kw_only=True)
class Config(BaseConfig):
    name: str = "qwen35-9b-lora-16k"
    model: Model = field(
        default_factory=lambda: Model(
            id="Qwen/Qwen3.5-9B-Base",
            revision="68c46c4b3498877f3ef123c856ecfde50c39f404",
            parameterization="lora",
            max_context_length=16384,
        )
    )
    routing: Routing = field(default_factory=lambda: Routing(default=True))
    deployment: Deployment = field(
        default_factory=lambda: Deployment(frontend="lilo-yaml", mode="shared")
    )
    trainer: Trainer = field(
        default_factory=lambda: Trainer(
            backend="miles",
            resources=Resources(gpu="H100:4", cpu=16, memory_mib=65536),
            scaling=TrainerScaling(max_instances=1),
            engine=EngineOptions(
                max_clients_per_instance=6, sampler_persistence_concurrency=8
            ),
            env={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
            },
            config={
                "model_args": "qwen3.5-9B",
                "options": {
                    "tensor_model_parallel_size": 4,
                    "multi_lora_n_adapters": 6,
                    "lora_rank": 32,
                    "lora_alpha": 32,
                    "target_modules": [
                        "linear_qkv",
                        "linear_proj",
                        "linear_fc1",
                        "linear_fc2",
                        "output_layer",
                    ],
                    "max_tokens_per_gpu": 16384,
                    "recompute_granularity": "full",
                    "recompute_method": "uniform",
                    "recompute_num_layers": 1,
                },
            },
        )
    )
    inference: Inference = field(
        default_factory=lambda: Inference(
            backend="sglang",
            resources=Resources(gpu="H200:1"),
            scaling=InferenceScaling(
                min_replicas=0, max_replicas=8, target_concurrency=16
            ),
            config={
                "tp_size": 1,
                "mem_fraction_static": 0.8,
                "max_running_requests": 32,
                "max_queued_requests": 8,
                "max_loaded_loras": 64,
                "max_loras_per_batch": 8,
            },
        )
    )
