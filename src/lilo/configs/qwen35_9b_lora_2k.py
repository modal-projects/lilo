from lilo.configuration import Compute, Deployment, Inference, Model, Trainer

config = Deployment(
    name="qwen35-9b-lora-2k",
    model=Model(id="Qwen/Qwen3.5-9B-Base", max_context_length=2048),
    trainer=Trainer(
        compute=Compute(gpu="H200", gpus_per_node=4),
        config={
            "model_type": "qwen3.5-9B",
            "tensor_model_parallel_size": 4,
            "target_modules": [
                "linear_qkv",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
                "output_layer",
            ],
            "max_tokens_per_gpu": 2048,
            "max_lora_slots": 4,
            "max_lora_rank": 32,
            "default_lora_alpha": 32,
            "cli_options": {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
        },
        env={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        max_clients_per_instance=4,
    ),
    inference=Inference(
        compute=Compute(gpu="H200"),
        config={
            "tp_size": 1,
            "ep_size": 1,
            "mem_fraction_static": 0.8,
            "max_running_requests": 32,
            "max_queued_requests": 8,
            "max_loaded_loras": 32,
            "max_loras_per_batch": 8,
            "schedule_policy": "lpm",
        },
    ),
)
