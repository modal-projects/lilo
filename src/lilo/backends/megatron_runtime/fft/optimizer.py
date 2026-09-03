from __future__ import annotations

from megatron.core.distributed import finalize_model_grads

from ..common.optimizer import set_adam_params


def run_fft_optimizer_step(optimizer, model, *, adam) -> tuple[bool, float]:
    set_adam_params(optimizer, adam)
    try:
        finalize_model_grads(model, None)
        successful, grad_norm, _ = optimizer.step()
        return bool(successful), float(grad_norm or 0.0)
    finally:
        for chunk in model:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()
