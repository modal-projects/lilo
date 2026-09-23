"""Miles deployment integration. Native options are validated by Miles at startup."""

from dataclasses import asdict

from lilo.deployments import gpu_count
from lilo.config_validation import reject_managed_options
from .miles_config import MilesBackendConfig

MILES_MANAGED = {
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "lora_alpha",
    "lora_dropout",
    "lora_rank",
    "max_tokens_per_gpu",
    "multi_lora_n_adapters",
    "target_modules",
    "tensor_model_parallel_size",
    "hf_checkpoint",
    "load",
    "pretrained_checkpoint",
    "train_backend",
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "rollout_num_gpus",
    "debug_train_only",
    "megatron_to_hf_mode",
    "seq_length",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "colocate",
    "custom_actor",
    "sglang_model_path",
    "use_dynamic_global_batch_size",
    "delay_split_train_data_by_dp",
    "use_dynamic_batch_size",
    "optimizer",
    "gradient_accumulation_fusion",
    "save",
    "save_interval",
    "ckpt_step",
    "rollout_num_gpus_per_engine",
    "num_gpus_per_node",
}


def build_config(spec, asset_path):
    """Pass trainer.config directly to MilesBackendConfig."""
    trainer = spec.trainer
    if spec.model["parameterization"] != "lora":
        raise ValueError("Miles requires lora parameterization")
    settings = trainer["config"]
    reject_managed_options(
        settings, {"hf_checkpoint", "actor_num_gpus_per_node", "extra_args"}
    )
    reject_managed_options(settings.get("cli_options", {}), MILES_MANAGED)
    config = MilesBackendConfig(
        hf_checkpoint=asset_path,
        actor_num_gpus_per_node=gpu_count(trainer["resources"]),
        extra_args=("--seq-length", str(spec.model["max_context_length"])),
        **settings,
    )
    config.validate()
    if config.world_size % (
        config.expert_model_parallel_size * config.expert_tensor_parallel_size
    ):
        raise ValueError("expert parallel sizes must divide the trainer GPU allocation")
    if trainer["engine"]["max_clients_per_instance"] > config.max_lora_slots:
        raise ValueError("max_clients_per_instance exceeds max_lora_slots")
    return {"miles": asdict(config), "checkpoint_dir": "/checkpoints"}
