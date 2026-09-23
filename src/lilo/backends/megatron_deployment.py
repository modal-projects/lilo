"""Construct the existing Megatron training config without a second schema."""

from dataclasses import asdict

from lilo.config_validation import reject_managed_options

from .megatron_config import parse_backend_config


def build_config(spec, asset_path):
    trainer = spec.trainer
    settings = trainer.config
    reject_managed_options(settings, {"hf_checkpoint", "seq_length"})
    config, _ = parse_backend_config(
        {
            "megatron": {
                **settings,
                "hf_checkpoint": asset_path,
                "seq_length": spec.model.max_context_length,
            }
        }
    )
    if config.optimizer.optimizer != "adam":
        raise ValueError("Tinker optim_step requires an Adam optimizer")
    config.validate(trainer.compute.gpus_per_node)
    return {"megatron": asdict(config), "checkpoint_dir": "/checkpoints"}
