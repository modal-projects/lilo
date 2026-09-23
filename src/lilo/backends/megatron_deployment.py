"""Build Lilo loop settings while preserving native Megatron configuration."""

from dataclasses import asdict, fields

from lilo.backend_options import native_options
from .megatron_runtime.common.config import EngineModelConfig, OptimizerConfig

# These values also control packing, collectives, and checkpoint metadata in Lilo.
# Configure them once under runtime so the provider and training loop agree.
PROVIDER_MANAGED = {
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "sequence_parallel",
    "variable_seq_lengths",
    "params_dtype",
    "seq_length",
}
DISTRIBUTED_MANAGED = {
    "use_distributed_optimizer",
    "overlap_grad_reduce",
    "overlap_param_gather",
    "align_param_gather",
}
OPTIMIZER_MANAGED = {
    "bf16",
    "fp16",
    "params_dtype",
    "use_distributed_optimizer",
    "overlap_param_gather",
}


def build_config(spec, asset_path):
    trainer = spec.trainer
    if spec.model.parameterization != "full":
        raise ValueError("Megatron YAML deployments require full parameterization")
    if trainer.engine.max_clients_per_instance != 1:
        raise ValueError("FFT trainers admit one client per instance")
    if trainer.engine.sampler_persistence_concurrency != 1:
        raise ValueError("Megatron requires sampler_persistence_concurrency: 1")
    sections = native_options(trainer.config, set())
    unknown = sections.keys() - {"runtime", "provider", "optimizer", "distributed"}
    if unknown:
        raise ValueError(f"unknown Megatron config sections: {sorted(unknown)}")
    runtime = native_options(
        sections.get("runtime", {}),
        {
            "hf_checkpoint",
            "seq_length",
            "max_lora_slots",
            "max_lora_rank",
            "optimizer",
            "provider_overrides",
            "optimizer_overrides",
            "distributed_overrides",
        },
    )
    provider = native_options(sections.get("provider", {}), PROVIDER_MANAGED)
    optimizer = native_options(sections.get("optimizer", {}), OPTIMIZER_MANAGED)
    distributed = native_options(sections.get("distributed", {}), DISTRIBUTED_MANAGED)
    if optimizer.get("optimizer", "adam") != "adam":
        raise ValueError("Tinker optim_step requires an Adam optimizer")
    # Keep scheduling and per-request optimizer settings available to Lilo. Every
    # other optimizer field goes to the installed Megatron constructor unchanged.
    loop_fields = {field.name for field in fields(OptimizerConfig)}
    loop_optimizer = {
        key: value for key, value in optimizer.items() if key in loop_fields
    }
    native_optimizer = {
        key: value for key, value in optimizer.items() if key not in loop_fields
    }
    try:
        config = EngineModelConfig(
            hf_checkpoint=asset_path,
            seq_length=spec.model.max_context_length,
            optimizer=OptimizerConfig(**loop_optimizer),
            provider_overrides=provider,
            optimizer_overrides=native_optimizer,
            distributed_overrides=distributed,
            **runtime,
        )
    except TypeError as exc:
        raise ValueError(f"invalid Megatron runtime options: {exc}") from exc
    config.validate(trainer.resources.gpu_count)
    return {"megatron": asdict(config), "checkpoint_dir": "/checkpoints"}
