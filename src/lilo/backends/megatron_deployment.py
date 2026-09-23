"""Construct the existing Megatron training config without a second schema."""

from dataclasses import asdict

from lilo.deployments import gpu_count
from lilo.config_validation import reject_managed_options
from .megatron_config import parse_backend_config

# These values also control packing, collectives, and checkpoint metadata in Lilo.
# Configure them once on EngineModelConfig so packing and the provider agree.
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
    if spec.model["parameterization"] != "full":
        raise ValueError("Megatron deployments require full parameterization")
    if trainer["engine"]["max_clients_per_instance"] != 1:
        raise ValueError("FFT trainers admit one client per instance")
    if trainer["engine"]["sampler_persistence_concurrency"] != 1:
        raise ValueError("Megatron requires sampler_persistence_concurrency: 1")
    settings = trainer["config"]
    reject_managed_options(settings, {"hf_checkpoint", "seq_length"})
    reject_managed_options(settings.get("provider_overrides", {}), PROVIDER_MANAGED)
    reject_managed_options(
        settings.get("optimizer_overrides", {}), OPTIMIZER_MANAGED | {"optimizer"}
    )
    reject_managed_options(
        settings.get("distributed_overrides", {}), DISTRIBUTED_MANAGED
    )
    config, _ = parse_backend_config(
        {
            "megatron": {
                **settings,
                "hf_checkpoint": asset_path,
                "seq_length": spec.model["max_context_length"],
            }
        }
    )
    if config.optimizer.optimizer != "adam":
        raise ValueError("Tinker optim_step requires an Adam optimizer")
    config.validate(gpu_count(trainer["resources"]))
    return {"megatron": asdict(config), "checkpoint_dir": "/checkpoints"}
