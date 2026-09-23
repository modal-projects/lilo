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
    name = "qwen35-9b-instruct-lora-16k"
    model = Model(
        id="Qwen/Qwen3.5-9B", parameterization="lora", max_context_length=16384
    )
    routing = Routing(default=True)
    trainer = Trainer(
        backend="miles",
        resources=Resources(gpu="H100:8"),
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
                "tensor_model_parallel_size": 8,
                "target_modules": [
                    "linear_qkv",
                    "linear_proj",
                    "linear_fc1",
                    "linear_fc2",
                ],
                "max_tokens_per_gpu": 16384,
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
                "multi_lora_n_adapters": 6,
                "lora_rank": 32,
                "lora_alpha": 32,
            },
        },
    )
    inference = Inference(
        resources=Resources(gpu="H200:1"),
        scaling=InferenceScaling(min_replicas=0, max_replicas=8, target_concurrency=16),
        config={
            "tp_size": 1,
            "ep_size": 1,
            "mem_fraction_static": 0.8,
            "max_running_requests": 32,
            "max_queued_requests": 8,
            "max_loaded_loras": 256,
            "max_loras_per_batch": 8,
            "lora_target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "schedule_policy": "lpm",
        },
    )
