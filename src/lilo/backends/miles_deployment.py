"""Miles deployment integration. Native options are validated by Miles at startup."""

from dataclasses import asdict

from lilo.backend_options import native_options
from .miles_config import MilesBackendConfig

MILES_MANAGED = {
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
    trainer = spec.trainer
    if spec.model.parameterization != "lora":
        raise ValueError("Miles requires lora parameterization")
    values = native_options(trainer.config, set())
    unknown = values.keys() - {"model_args", "options"}
    if unknown:
        raise ValueError(f"unknown Miles config sections: {sorted(unknown)}")
    options = native_options(values.get("options", {}), MILES_MANAGED)
    names = {
        "tensor_model_parallel_size": "tensor_model_parallel_size",
        "context_parallel_size": "context_parallel_size",
        "expert_model_parallel_size": "expert_model_parallel_size",
        "expert_tensor_parallel_size": "expert_tensor_parallel_size",
        "multi_lora_n_adapters": "max_lora_slots",
        "lora_rank": "max_lora_rank",
        "lora_alpha": "default_lora_alpha",
        "lora_dropout": "lora_dropout",
        "target_modules": "target_modules",
        "max_tokens_per_gpu": "max_tokens_per_gpu",
    }
    settings = {
        dest: options.pop(source) for source, dest in names.items() if source in options
    }
    for name, value in settings.items():
        if (
            name not in {"target_modules", "default_lora_alpha", "lora_dropout"}
            and type(value) is not int
        ):
            raise ValueError(f"Miles option {name} must be an integer")
    if "target_modules" in settings:
        value = settings["target_modules"]
        value = value.split(",") if isinstance(value, str) else value
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError("target_modules must be a nonempty list of module names")
        settings["target_modules"] = tuple(value)
    config = MilesBackendConfig(
        hf_checkpoint=asset_path,
        model_type=values.get("model_args") or "",
        actor_num_gpus_per_node=trainer.resources.gpu_count,
        native_options=options,
        extra_args=("--seq-length", str(spec.model.max_context_length)),
        **settings,
    )
    config.validate()
    if config.world_size % (
        config.expert_model_parallel_size * config.expert_tensor_parallel_size
    ):
        raise ValueError("expert parallel sizes must divide the trainer GPU allocation")
    if trainer.engine.max_clients_per_instance > config.max_lora_slots:
        raise ValueError("max_clients_per_instance exceeds multi_lora_n_adapters")
    return {"miles": asdict(config), "checkpoint_dir": "/checkpoints"}
