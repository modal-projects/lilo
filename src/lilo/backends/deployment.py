"""Resolve backend settings before launch; reserve fields owned by Lilo."""

from dataclasses import asdict

from lilo.backends.megatron_config import parse_backend_config
from lilo.backends.miles_config import MilesBackendConfig
from lilo.config_validation import reject_managed_options

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


SGLANG_MANAGED = {
    "model_path",
    "model",
    "host",
    "port",
    "context_length",
    "enable_lora",
    "max_lora_rank",
    "lora_target_modules",
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


def backend_config(spec, asset_path="/assets/pending"):
    trainer = spec.trainer
    settings = trainer.config
    if trainer.backend == "megatron":
        reject_managed_options(settings, {"hf_checkpoint", "seq_length"})
        config, _ = parse_backend_config(
            {
                "megatron": {
                    **settings,
                    "hf_checkpoint": asset_path,
                    "seq_length": spec.max_context_length,
                }
            }
        )
        if config.optimizer.optimizer != "adam":
            raise ValueError("Tinker optim_step requires an Adam optimizer")
        config.validate(trainer.gpus_per_node)
        return {"megatron": asdict(config), "checkpoint_dir": "/checkpoints"}
    reject_managed_options(
        settings,
        {"hf_checkpoint", "actor_num_gpus_per_node", "actor_num_nodes", "extra_args"},
    )
    reject_managed_options(settings.get("cli_options", {}), MILES_MANAGED)
    config = MilesBackendConfig(
        hf_checkpoint=asset_path,
        actor_num_gpus_per_node=trainer.gpus_per_node,
        actor_num_nodes=trainer.nodes,
        extra_args=("--seq-length", str(spec.max_context_length)),
        **settings,
    )
    config.validate()
    if config.world_size % (
        config.expert_model_parallel_size * config.expert_tensor_parallel_size
    ):
        raise ValueError("expert parallel sizes must divide the trainer GPU allocation")
    if trainer.max_clients_per_instance > config.max_lora_slots:
        raise ValueError("max_clients_per_instance exceeds max_lora_slots")
    return {"miles": asdict(config), "checkpoint_dir": "/checkpoints"}


def serving_options(spec):
    options = dict(spec.inference.config)
    reject_managed_options(options, SGLANG_MANAGED)
    tp = options.get("tp_size", spec.inference.gpus_per_node)
    ep = options.get("ep_size", 1)
    if type(tp) is not int or tp != spec.inference.gpus_per_node:
        raise ValueError("sglang.tp_size must equal the replica GPU allocation")
    if type(ep) is not int or ep < 1 or tp % ep:
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
        if key in options and (type(options[key]) is not int or options[key] < 1):
            raise ValueError(f"sglang.{key} must be positive")
    if not 0 < options.get("mem_fraction_static", 0.8) < 1:
        raise ValueError("sglang.mem_fraction_static must be between zero and one")
    if options.get("max_loaded_loras", 64) < options.get("max_loras_per_batch", 8):
        raise ValueError("max_loaded_loras must be >= max_loras_per_batch")
    return options


def resolve_backend_settings(spec, asset_path):
    trainer = backend_config(spec, asset_path)
    inference = {
        "context_length": spec.max_context_length,
        "tp_size": spec.inference.gpus_per_node,
        "mem_fraction_static": 0.8,
        "max_running_requests": 32,
        "weight_loader_disable_mmap": True,
        **serving_options(spec),
    }
    if spec.parameterization == "lora":
        miles = MilesBackendConfig(**trainer["miles"])
        inference.update(
            enable_lora=True,
            max_lora_rank=miles.max_lora_rank,
            lora_target_modules=list(miles.peft_target_modules),
        )
        inference.setdefault("max_loaded_loras", 64)
        inference.setdefault("max_loras_per_batch", 8)
    else:
        inference["enable_cpu_weight_cache"] = True
    return trainer, inference
