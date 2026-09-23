"""Resolve lightweight backend settings once, before creating worker apps."""

from lilo.backends.megatron_deployment import build_config as megatron_config
from lilo.backends.miles_config import MilesBackendConfig
from lilo.backends.miles_deployment import build_config as miles_config
from lilo.inference.sglang_deployment import build_config as sglang_config

TRAINERS = {"miles": miles_config, "megatron": megatron_config}


def backend_config(spec, asset_path="/assets/pending"):
    return TRAINERS[spec.trainer.backend](spec, asset_path)


def serving_options(spec):
    return sglang_config(spec)


def resolve_backend_settings(spec, asset_path):
    trainer = backend_config(spec, asset_path)
    inference = {
        "context_length": spec.model.max_context_length,
        "tp_size": spec.inference.compute.gpus_per_node,
        "mem_fraction_static": 0.8,
        "max_running_requests": 32,
        "weight_loader_disable_mmap": True,
        **serving_options(spec),
    }
    if spec.model.parameterization == "lora":
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
