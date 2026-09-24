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
    if spec.backend == "megatron":
        if spec.trainer_nodes != 1:
            raise ValueError("multi-node training currently requires Miles")
        if spec.parameterization != "full":
            raise ValueError("Megatron requires full parameterization")
        if spec.trainer_max_clients_per_instance != 1:
            raise ValueError("FFT trainers admit one client per instance")
        if spec.sampler_persistence_concurrency != 1:
            raise ValueError("Megatron requires sampler_persistence_concurrency: 1")
        settings = spec.megatron_cfg
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
        config.validate(spec.trainer_gpus_per_node)
        return {"megatron": asdict(config), "checkpoint_dir": "/checkpoints"}
    if spec.backend != "miles":
        raise ValueError(f"unknown backend: {spec.backend}")
    if spec.parameterization != "lora":
        raise ValueError("Miles requires lora parameterization")
    settings = spec.miles_cfg
    reject_managed_options(
        settings,
        {"hf_checkpoint", "actor_num_gpus_per_node", "actor_num_nodes", "extra_args"},
    )
    reject_managed_options(settings.get("cli_options", {}), MILES_MANAGED)
    config = MilesBackendConfig(
        hf_checkpoint=asset_path,
        actor_num_gpus_per_node=spec.trainer_gpus_per_node,
        actor_num_nodes=spec.trainer_nodes,
        extra_args=("--seq-length", str(spec.max_context_length)),
        **settings,
    )
    config.validate()
    if config.world_size % (
        config.expert_model_parallel_size * config.expert_tensor_parallel_size
    ):
        raise ValueError("expert parallel sizes must divide the trainer GPU allocation")
    if spec.trainer_max_clients_per_instance > config.max_lora_slots:
        raise ValueError("max_clients_per_instance exceeds max_lora_slots")
    return {"miles": asdict(config), "checkpoint_dir": "/checkpoints"}


def serving_options(spec):
    options = dict(spec.sglang_cfg)
    reject_managed_options(options, SGLANG_MANAGED)
    tp = options.get("tp_size", spec.inference_gpus_per_node)
    if tp != spec.inference_gpus_per_node:
        raise ValueError("sglang.tp_size must equal the replica GPU allocation")
    return options


def resolve_backend_settings(spec, asset_path):
    trainer = backend_config(spec, asset_path)
    inference = {
        "context_length": spec.max_context_length,
        "tp_size": spec.inference_gpus_per_node,
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
