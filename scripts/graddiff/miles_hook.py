"""Megatron before-train-step hook for the Miles arm of the step-0 grad diff.

Installed via ``--custom-megatron-before-train-step-hook-path
miles_hook.before_train_step``. On the first call it:

1. Copies Lilo's LoRA-A init into Miles' LoRA ``linear_in`` params (matching by
   module path suffix; exact shape match required) and verifies Miles' ``B``
   params are all zero.
2. Wraps ``miles.backends.megatron_utils.model.finalize_model_grads`` to dump
   adapter-param grads pre/post DP reduce (``grads_local`` / ``grads_reduced``).
3. Wraps ``optimizer.step`` to snapshot params before/after and write
   ``params_before``/``params_after``/``delta`` + ``meta``.

Env: ``MILES_GRADDIFF_DUMP_DIR`` (required to activate), ``LILO_DUMP_DIR``
(dir holding Lilo's ``rank*_tp{T}_dp0_params_before.pt``), ``MILES_GRADDIFF_TAG``
(default ``"miles"``). Per-token trainer logprobs are captured separately by
Miles' native ``--dump-details``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

DUMP_DIR = os.environ.get("MILES_GRADDIFF_DUMP_DIR")
LILO_DUMP_DIR = os.environ.get("LILO_DUMP_DIR")
TAG = os.environ.get("MILES_GRADDIFF_TAG", "miles")

_installed = False


def _rank_info() -> dict[str, int]:
    import torch.distributed as dist

    info = {"rank": dist.get_rank() if dist.is_initialized() else 0}
    try:
        from megatron.core import parallel_state as ps

        info["tp_rank"] = ps.get_tensor_model_parallel_rank()
        info["dp_rank"] = ps.get_data_parallel_rank()
    except (ImportError, AssertionError, RuntimeError):
        info.setdefault("tp_rank", 0)
        info.setdefault("dp_rank", 0)
    return info


def _write(
    kind: str, tensors: dict[str, Any], meta: dict[str, Any] | None = None
) -> None:
    info = _rank_info()
    out_dir = Path(DUMP_DIR) / TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"rank{info['rank']}_tp{info.get('tp_rank', 0)}_dp{info.get('dp_rank', 0)}_{kind}"
    torch.save(
        {k: v.detach().to(torch.float32).cpu() for k, v in tensors.items()},
        out_dir / f"{name}.pt",
    )
    if meta is not None:
        (out_dir / f"{name}_meta.json").write_text(
            json.dumps({**meta, **info, "tag": TAG}, default=str)
        )


def _grad_of(param: Any):
    grad = getattr(param, "main_grad", None)
    if grad is not None:
        return grad
    return param.grad


def _grads(params: dict[str, Any]) -> dict[str, Any]:
    return {
        name: grad
        for name, param in params.items()
        if (grad := _grad_of(param)) is not None
    }


def _adapter_params(model: Any) -> dict[str, Any]:
    chunks = model if isinstance(model, (list, tuple)) else [model]
    params: dict[str, Any] = {}
    for chunk in chunks:
        for name, param in chunk.named_parameters():
            if "linear_in" in name or "linear_out" in name:
                params[name] = param
    return params


_LILO_PREFIX_RE = re.compile(r"^slot0\.module\.module\.")
_LAYER_KEY_RE = re.compile(
    r"(language_model\.decoder\.layers\.\d+\.[\w.]+?)"
    r"\.(?:adapters?\.(?:\d+\.)?)?linear_in\.weight"
)


def _lilo_key(name: str) -> str | None:
    """Map a dump name to its canonical ``language_model...linear_in.weight`` key."""
    stripped = _LILO_PREFIX_RE.sub("", name)
    m = _LAYER_KEY_RE.search(stripped)
    return m.group(1) if m else None


def _miles_key(name: str) -> str | None:
    m = _LAYER_KEY_RE.search(name)
    return m.group(1) if m else None


def _copy_lilo_A(model: Any, params: dict[str, Any]) -> None:
    info = _rank_info()
    tp = info.get("tp_rank", 0)
    lilo_dir = Path(LILO_DUMP_DIR)
    candidates = list(lilo_dir.glob(f"rank*_tp{tp}_dp0_params_before.pt"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"tp{tp}: expected 1 Lilo params_before file, found {candidates}"
        )
    lilo = torch.load(candidates[0], weights_only=True)

    lilo_a: dict[str, Any] = {}
    for name, tensor in lilo.items():
        if "vision_model" in name:
            continue
        if not name.endswith("linear_in.weight"):
            continue
        key = _lilo_key(name)
        if key is None:
            raise RuntimeError(f"unparseable Lilo A name: {name}")
        lilo_a[key] = tensor

    copied, skipped, mismatched = 0, 0, []
    example_names = [n for n in params if n.endswith("linear_in.weight")][:4]
    for name in example_names:
        logger.info(
            f"[graddiff] miles A param example: {name} {tuple(params[name].shape)}"
        )

    for name, param in params.items():
        if not name.endswith("linear_in.weight"):
            continue
        key = _miles_key(name)
        if key is None or key not in lilo_a:
            skipped += 1
            mismatched.append(f"no lilo match: {name} -> {key}")
            continue
        src = lilo_a[key]
        if src.shape != param.shape:
            raise RuntimeError(
                f"shape mismatch for {name}: lilo {tuple(src.shape)} vs miles {tuple(param.shape)}"
            )
        with torch.no_grad():
            param.copy_(src.to(device=param.device, dtype=param.dtype))
        copied += 1

    b_names = [n for n in params if n.endswith("linear_out.weight")]
    b_nonzero = [n for n in b_names if torch.any(params[n].detach() != 0).item()]
    if b_nonzero:
        raise RuntimeError(f"miles B params nonzero before step: {b_nonzero[:4]}")

    # vision tower: only report
    vision = [n for n in lilo if "vision_model" in n and n.endswith("linear_in.weight")]
    logger.info(
        f"[graddiff] A-copy: copied={copied} skipped={skipped} "
        f"miles_B={len(b_names)}(all zero) lilo_vision_A_skipped={len(vision)}"
    )
    if mismatched:
        logger.warning(f"[graddiff] unmatched miles A params: {mismatched[:8]}")
    if copied == 0:
        raise RuntimeError("A-copy copied 0 params — name mapping failed")

    _write(
        "a_copy",
        {},
        {
            "n_copied": copied,
            "n_skipped": skipped,
            "unmatched": mismatched,
            "lilo_a_keys": len(lilo_a),
            "example_miles_names": example_names,
        },
    )


def before_train_step(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    global _installed
    if not DUMP_DIR or _installed:
        return
    _installed = True

    import miles.backends.megatron_utils.model as miles_model

    params = _adapter_params(model)
    logger.info(
        f"[graddiff] capturing {len(params)} adapter params; examples: {list(params)[:4]}"
    )

    _copy_lilo_A(model, params)
    # Refresh the FP32 master weights so the optimizer does not write Miles'
    # original init back into param.data at step end. ChainedOptimizer has no
    # reload_model_params; its LayerWise children do.
    def _reload_masters(opt: Any) -> int:
        if opt is None:
            return 0
        if hasattr(opt, "reload_model_params"):
            opt.reload_model_params()
            return 1
        return sum(
            _reload_masters(child)
            for child in getattr(opt, "chained_optimizers", [])
        )

    n_reload = _reload_masters(optimizer)
    logger.info(f"[graddiff] reloaded fp32 masters on {n_reload} optimizer children")
    _write(
        "master_reload",
        {},
        {"n_optimizer_children_reloaded": n_reload},
    )
    _write("params_before", {n: p.detach().clone() for n, p in params.items()})

    # ---- grads around DP reduce ----
    original_finalize = miles_model.finalize_model_grads

    def wrapped_finalize(*f_args: Any, **f_kwargs: Any):
        local = _grads(params)
        _write(
            "grads_local",
            local,
            {
                "phase": "pre_reduce",
                "dtypes": {n: str(g.dtype) for n, g in local.items()},
            },
        )
        return original_finalize(*f_args, **f_kwargs)

    miles_model.finalize_model_grads = wrapped_finalize

    # ---- optimizer.step snapshot ----
    if optimizer is not None:
        original_step = optimizer.step
        before = {n: p.detach().clone() for n, p in params.items()}

        def wrapped_step(*s_args: Any, **s_kwargs: Any):
            # At step entry the DP grad reduce is fully synced; snapshot the
            # reduced grads here (post-finalize is still async in Miles).
            reduced = _grads(params)
            _write(
                "grads_reduced",
                reduced,
                {
                    "phase": "pre_step",
                    "dtypes": {n: str(g.dtype) for n, g in reduced.items()},
                },
            )
            result = original_step(*s_args, **s_kwargs)
            after = {n: p.detach().clone() for n, p in params.items()}
            _write("params_after", after)
            _write("delta", {n: after[n] - before[n] for n in params})
            _write(
                "optim_step",
                {},
                {
                    "optimizer_class": type(optimizer).__name__,
                    "step_result": result,
                    "adam": {
                        "lr": getattr(args, "lr", None),
                        "beta1": getattr(args, "adam_beta1", None),
                        "beta2": getattr(args, "adam_beta2", None),
                        "eps": getattr(args, "adam_eps", None),
                        "weight_decay": getattr(args, "weight_decay", None),
                        "clip_grad": getattr(args, "clip_grad", None),
                    },
                },
            )
            miles_model.finalize_model_grads = original_finalize
            optimizer.step = original_step
            return result

        optimizer.step = wrapped_step
