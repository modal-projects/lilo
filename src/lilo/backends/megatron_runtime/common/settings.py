"""One ownership rule: backend extras may add fields, never replace Lilo settings."""

from dataclasses import asdict

from lilo.config_validation import reject_managed_options


def provider_settings(config, dtype):
    owned = {
        "tensor_model_parallel_size": config.tensor_model_parallel_size,
        "pipeline_model_parallel_size": config.pipeline_model_parallel_size,
        "virtual_pipeline_model_parallel_size": config.virtual_pipeline_model_parallel_size,
        "context_parallel_size": config.context_parallel_size,
        "expert_model_parallel_size": config.expert_model_parallel_size,
        "expert_tensor_parallel_size": config.expert_tensor_parallel_size,
        "sequence_parallel": config.sequence_parallel,
        "variable_seq_lengths": True,
        "calculate_per_token_loss": config.calculate_per_token_loss,
        "attention_backend": config.attention_backend,
        "cross_entropy_loss_fusion": config.cross_entropy_loss_fusion,
        "params_dtype": dtype,
    }
    reject_managed_options(config.provider_overrides, owned.keys() | {"seq_length"})
    return {**owned, **config.provider_overrides}


def optimizer_settings(config, dtype, distributed_optimizer):
    owned = {
        "optimizer": config.optimizer.optimizer,
        "lr": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
        "adam_beta1": config.optimizer.adam_beta1,
        "adam_beta2": config.optimizer.adam_beta2,
        "adam_eps": config.optimizer.adam_eps,
        "clip_grad": config.optimizer.clip_grad,
        "loss_scale": config.optimizer.loss_scale,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "params_dtype": dtype,
        "use_distributed_optimizer": distributed_optimizer,
        "overlap_param_gather": config.overlap_param_gather,
    }
    reject_managed_options(
        config.optimizer_overrides, owned.keys() | asdict(config.optimizer).keys()
    )
    return {**owned, **config.optimizer_overrides}


def distributed_settings(config, distributed_optimizer):
    owned = {
        "use_distributed_optimizer": distributed_optimizer,
        "overlap_grad_reduce": config.overlap_grad_reduce,
        "overlap_param_gather": config.overlap_param_gather,
        "align_param_gather": config.align_param_gather,
    }
    reject_managed_options(config.distributed_overrides, owned.keys())
    return {**owned, **config.distributed_overrides}
