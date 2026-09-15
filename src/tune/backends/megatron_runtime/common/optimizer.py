from __future__ import annotations


def set_adam_params(optimizer, adam) -> None:
    if optimizer.config.optimizer != "adam":
        raise ValueError("optim_step requires an Adam optimizer")
    if adam.learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    if not 0 <= adam.beta1 < 1 or not 0 <= adam.beta2 < 1:
        raise ValueError("adam betas must be in [0, 1)")
    if adam.eps <= 0:
        raise ValueError("adam eps must be positive")
    if adam.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if adam.grad_clip_norm < 0:
        raise ValueError("grad_clip_norm must be non-negative")

    for group in optimizer.param_groups:
        group["lr"] = float(adam.learning_rate) * float(group.get("lr_mult", 1.0))
        group["betas"] = (float(adam.beta1), float(adam.beta2))
        group["eps"] = float(adam.eps)
        group["weight_decay"] = float(adam.weight_decay) * float(
            group.get("wd_mult", 1.0)
        )
    optimizer.config.clip_grad = float(adam.grad_clip_norm)
