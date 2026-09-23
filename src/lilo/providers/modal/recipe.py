"""Translate a deployment recipe into backend and serving configurations."""

from __future__ import annotations

from dataclasses import asdict

from lilo.backends.miles_config import MilesBackendConfig

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
SGLANG_MANAGED = {
    "model_path",
    "model",
    "host",
    "port",
    "context_length",
    "enable_lora",
    "max_lora_rank",
    "enable_cpu_weight_cache",
    "api_key",
    "pp_size",
    "lora_paths",
    "dist_init_addr",
    "nnodes",
    "node_rank",
    "tokenizer_path",
    "tokenizer_revision",
    "revision",
    "grpc_mode",
    "smg_grpc_mode",
    "encoder_only",
    "use_ray",
    "disaggregation_mode",
    "skip_tokenizer_init",
}


def native_options(values, protected):
    result = {}
    for key, value in values.items():
        if key.startswith("-") or key.replace("_", "").isalnum() is False:
            raise ValueError(f"native option must use its underscore name: {key}")
        if key in protected:
            raise ValueError(f"option {key} is managed by Lilo")
        result[key] = value
    return result


def backend_config(spec, asset_path="/assets/pending"):
    trainer = spec.trainer
    if trainer.backend == "miles":
        options = native_options(trainer.miles.options, MILES_MANAGED)
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
            dest: options.pop(source)
            for source, dest in names.items()
            if source in options
        }
        for name, value in settings.items():
            if (
                name not in {"target_modules", "default_lora_alpha", "lora_dropout"}
                and type(value) is not int
            ):
                raise ValueError(f"trainer.miles option {name} must be an integer")
        if "target_modules" in settings:
            value = settings["target_modules"]
            value = value.split(",") if isinstance(value, str) else value
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(
                    "target_modules must be a nonempty list of module names"
                )
            settings["target_modules"] = tuple(value)
        config = MilesBackendConfig(
            hf_checkpoint=asset_path,
            model_type=trainer.miles.model_args or "",
            actor_num_gpus_per_node=trainer.resources.gpu_count,
            native_options=options,
            extra_args=("--seq-length", str(spec.model.max_context_length)),
            **settings,
        )
        config.validate()
        if config.world_size % (
            config.expert_model_parallel_size * config.expert_tensor_parallel_size
        ):
            raise ValueError(
                "expert parallel sizes must divide the trainer GPU allocation"
            )
        if trainer.engine.max_clients_per_instance > config.max_lora_slots:
            raise ValueError("max_clients_per_instance exceeds multi_lora_n_adapters")
        return {"miles": asdict(config), "checkpoint_dir": "/checkpoints"}
    from lilo.backends.megatron_runtime.common.config import (
        EngineModelConfig,
        OptimizerConfig,
    )

    options = native_options(
        trainer.megatron.options,
        {"hf_checkpoint", "seq_length", "max_lora_slots", "max_lora_rank"},
    )
    overrides = options.get("provider_overrides", {})
    protected = {
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
    native_options(overrides, protected)
    try:
        optimizer = OptimizerConfig(**options.pop("optimizer", {}))
    except TypeError as exc:
        raise ValueError(f"invalid trainer.megatron optimizer options: {exc}") from exc
    try:
        config = EngineModelConfig(
            hf_checkpoint=asset_path,
            seq_length=spec.model.max_context_length,
            optimizer=optimizer,
            **options,
        )
    except TypeError as exc:
        raise ValueError(f"invalid trainer.megatron options: {exc}") from exc
    config.validate(trainer.resources.gpu_count)
    return {"megatron": asdict(config), "checkpoint_dir": "/checkpoints"}


def serving_options(spec):
    options = native_options(spec.inference.sglang.options, SGLANG_MANAGED)
    tp = options.get("tp_size", spec.inference.resources.gpu_count)
    ep = options.get("ep_size", 1)
    if not isinstance(tp, int) or tp != spec.inference.resources.gpu_count:
        raise ValueError("sglang.tp_size must equal the replica GPU allocation")
    if not isinstance(ep, int) or ep < 1 or tp % ep:
        raise ValueError("sglang.ep_size must divide the replica GPU allocation")
    dp = options.get("dp_size", 1)
    dp_attention = options.get("enable_dp_attention", False)
    if type(dp) is not int or dp < 1 or tp % dp:
        raise ValueError("sglang.dp_size must divide the replica GPU allocation")
    if not isinstance(dp_attention, bool):
        raise ValueError("sglang.enable_dp_attention must be a boolean")
    if dp > 1 and not dp_attention:
        raise ValueError("sglang.dp_size > 1 requires enable_dp_attention")
    for key in (
        "max_loaded_loras",
        "max_loras_per_batch",
        "max_running_requests",
        "max_queued_requests",
    ):
        if key in options and (not isinstance(options[key], int) or options[key] < 1):
            raise ValueError(f"sglang.{key} must be positive")
    if not 0 < options.get("mem_fraction_static", 0.8) < 1:
        raise ValueError("sglang.mem_fraction_static must be between zero and one")
    if options.get("max_loaded_loras", 64) < options.get("max_loras_per_batch", 8):
        raise ValueError("max_loaded_loras must be >= max_loras_per_batch")
    return options
