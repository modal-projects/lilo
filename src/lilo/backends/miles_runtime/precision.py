"""Optional higher-precision tensor-parallel reductions for parity experiments.

BF16 reduce-scatter can round differently when packing moves a token to another
sequence-parallel rank. Accumulate partial results in the configured dtype and
cast only once. This improves reduction accuracy; it does not guarantee
deterministic GEMMs or backward kernels.
"""

from functools import wraps
import sys


class _CastAfterWait:
    def __init__(self, work, output, reduced):
        self.work = work
        self.output = output
        self.reduced = reduced
        self.done = False

    def wait(self):
        result = self.work.wait()
        if not self.done:
            self.output.copy_(self.reduced)
            self.done = True
        return result


_accumulation_dtype = "float32"


def install_tp_reductions(dtype="float32") -> None:
    import torch
    from megatron.core.tensor_parallel import layers, mappings
    import transformer_engine.pytorch.distributed as te_distributed

    global _accumulation_dtype
    if dtype not in ("float32", "float64"):
        raise ValueError("Reduction dtype must be float32 or float64")
    _accumulation_dtype = dtype

    def accumulation_dtype():
        return getattr(torch, _accumulation_dtype)

    def should_cast(value):
        return value.dtype in (torch.float16, torch.bfloat16) or (
            value.dtype == torch.float32 and _accumulation_dtype == "float64"
        )

    def install(module, name, factory):
        original = getattr(module, name)
        if not getattr(original, "_lilo_precision_reduction", False):
            replacement = factory(original)
            replacement._lilo_precision_reduction = True
            setattr(module, name, replacement)

    def mapping_scatter(original):
        @wraps(original)
        def scatter(value, *args, **kwargs):
            if not should_cast(value):
                return original(value, *args, **kwargs)
            return original(value.to(accumulation_dtype()), *args, **kwargs).to(
                value.dtype
            )

        return scatter

    def mapping_reduce(original):
        @wraps(original)
        def reduce(value, group, fp32=False):
            if not should_cast(value):
                return original(value, group, fp32=fp32)
            reduced = original(value.to(accumulation_dtype()), group, fp32=False)
            value.copy_(reduced)
            return value

        return reduce

    def native_scatter(original):
        @wraps(original)
        def scatter(output, value, *args, **kwargs):
            if not should_cast(value):
                return original(output, value, *args, **kwargs)
            reduced = torch.empty_like(output, dtype=accumulation_dtype())
            work = original(reduced, value.to(accumulation_dtype()), *args, **kwargs)
            if work is None:
                output.copy_(reduced)
                return None
            return _CastAfterWait(work, output, reduced)

        return scatter

    def te_scatter(original):
        @wraps(original)
        def scatter(value, *args, **kwargs):
            if not should_cast(value):
                return original(value, *args, **kwargs)
            reduced, work = original(value.to(accumulation_dtype()), *args, **kwargs)
            if work is None:
                return reduced.to(value.dtype), None
            output = torch.empty_like(reduced, dtype=value.dtype)
            return output, _CastAfterWait(work, output, reduced)

        return scatter

    install(mappings, "_reduce_scatter_along_first_dim", mapping_scatter)
    install(mappings, "_reduce", mapping_reduce)
    install(layers, "dist_reduce_scatter_func", native_scatter)
    original = te_distributed.reduce_scatter_along_first_dim
    if not getattr(original, "_lilo_precision_reduction", False):
        replacement = te_scatter(original)
        replacement._lilo_precision_reduction = True
        # TE imports the collective into its linear/MLP implementation modules.
        for name, module in list(sys.modules.items()):
            if (
                name.startswith("transformer_engine.pytorch")
                and getattr(module, "reduce_scatter_along_first_dim", None) is original
            ):
                module.reduce_scatter_along_first_dim = replacement


def configure_deterministic_attention() -> None:
    """Select FA2, which supports deterministic backward for Qwen3.5's head size.

    The FA3 build in the pinned Miles image rejects deterministic backward at
    head dimension 256. TE 2.17 predates per-version environment switches, so
    also update its shared availability flags before constructing attention.
    """
    import os
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        FlashAttentionUtils,
    )

    if not FlashAttentionUtils.is_installed:
        raise RuntimeError("Deterministic Miles attention requires FlashAttention 2")
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["NVTE_FLASH_ATTN_V3"] = "0"
    os.environ["NVTE_FLASH_ATTN_V4"] = "0"
    FlashAttentionUtils.v3_is_installed = False
    if hasattr(FlashAttentionUtils, "v4_is_installed"):
        FlashAttentionUtils.v4_is_installed = False


def configure_deterministic_losses() -> None:
    """Use one dynamic-shape CE path from the first sample.

    Megatron's default jit_fuser specializes the first response length, then
    recompiles when another length arrives. Those kernels can round differently,
    so replaying the first sample changes its logprobs despite identical logits.
    Compile the existing arithmetic dynamically from the start; no warmup or
    change to the cross-entropy objective is needed.
    """
    import torch
    from megatron.core.fusions import fused_cross_entropy

    for name in (
        "calculate_logits_max",
        "calculate_predicted_logits",
        "calculate_cross_entropy_loss",
        "calculate_gradients",
    ):
        current = getattr(fused_cross_entropy, name)
        if getattr(current, "_lilo_dynamic_shapes", False):
            continue
        original = getattr(current, "_torchdynamo_orig_callable", current)
        replacement = torch.compile(original, dynamic=True)
        replacement._lilo_dynamic_shapes = True
        setattr(fused_cross_entropy, name, replacement)


def configure_deterministic_gdn() -> None:
    """Pin the reduction layout of FLA Q/K normalization before compilation.

    Autotuning different warp/tile layouts can change FP32 reduction rounding.
    A fixed layout must be shared by cold, recompiled, and packed execution.
    Keep FLA's existing forward/backward arithmetic and autograd implementation.
    """
    import importlib
    import triton

    normalization = importlib.import_module("fla.modules.l2norm")
    for name in (
        "l2norm_fwd_kernel",
        "l2norm_bwd_kernel",
        "l2norm_fwd_kernel1",
        "l2norm_bwd_kernel1",
    ):
        original = getattr(normalization, name)
        if getattr(original, "_lilo_fixed_layout", False):
            continue
        scalar = name.endswith("kernel1")
        replacement = triton.autotune(
            configs=[triton.Config({} if scalar else {"BT": 32}, num_warps=4)],
            key=["D"] if scalar else ["D", "NB"],
        )(original.fn)
        replacement._lilo_fixed_layout = True
        setattr(normalization, name, replacement)
