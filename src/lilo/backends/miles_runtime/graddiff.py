"""Grad/param capture for the Lilo-vs-Miles step-0 gradient comparison.

Enabled by ``LILO_GRADDIFF_DUMP_DIR`` (unset → every entry point is a no-op).
Inside a ``forward_backward`` call the module wraps
``miles.backends.megatron_utils.model.finalize_model_grads`` once so slot-param
``main_grad``/``grad`` tensors are dumped right before the DP reduce
(``*_grads_local.pt``) and right after (``*_grads_reduced.pt``). Around
``optim_step`` it snapshots slot params before/after Adam and writes
``params_before``, ``params_after``, ``delta`` plus ``meta.json`` with the Adam
params and per-slot outcomes.

Files land at ``{DUMP_DIR}/{tag}/rank{R}_tp{T}_dp{D}_{kind}.pt`` where ``tag``
defaults to ``"lilo"`` via ``LILO_GRADDIFF_TAG``. Writes overwrite earlier
captures; only the most recent call is retained.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DUMP_DIR = os.environ.get("LILO_GRADDIFF_DUMP_DIR")
TAG = os.environ.get("LILO_GRADDIFF_TAG", "lilo")

_NAME_MAP_KEY = "_graddiff_name"
_calls = {"fwd": 0, "opt": 0}


def enabled() -> bool:
    return bool(DUMP_DIR)


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


def _slot_params(model: Any, slot_optimizers: dict[int, Any]) -> dict[str, Any]:
    """id-ordered {name: param} for every loaded adapter slot."""
    from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

    chunks = model if isinstance(model, (list, tuple)) else [model]
    names: dict[int, str] = {}
    for chunk in chunks:
        for name, param in chunk.named_parameters():
            names.setdefault(id(param), name)
    params: dict[str, Any] = {}
    for slot in sorted(slot_optimizers):
        for param in adapter_slot_parameters(model, slot):
            name = names.get(id(param), f"param_{id(param):x}")
            params[f"slot{slot}.{name}"] = param
    return params


def _write(
    kind: str, tensors: dict[str, Any], meta: dict[str, Any] | None = None
) -> None:
    import torch

    info = _rank_info()
    out_dir = Path(DUMP_DIR) / TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"rank{info['rank']}_tp{info.get('tp_rank', 0)}_dp{info.get('dp_rank', 0)}_{kind}"
    torch.save(
        {key: value.detach().to(torch.float32).cpu() for key, value in tensors.items()},
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


def _grad_dtypes(grads: dict[str, Any]) -> dict[str, str]:
    return {name: str(grad.dtype) for name, grad in grads.items()}


@contextmanager
def capture_grads(model: Any, slot_optimizers: dict[int, Any]) -> Iterator[None]:
    """Wrap the forward_backward call; dump slot grads around the DP reduce."""
    if not enabled():
        yield
        return
    import miles.backends.megatron_utils.model as miles_model

    params = _slot_params(model, slot_optimizers)
    call_index = _calls["fwd"]
    _calls["fwd"] += 1
    meta = {"call_index": call_index, "n_params": len(params)}
    original = miles_model.finalize_model_grads

    def wrapped(*args: Any, **kwargs: Any):
        local = _grads(params)
        _write(
            "grads_local",
            local,
            {**meta, "phase": "pre_reduce", "dtypes": _grad_dtypes(local)},
        )
        try:
            return original(*args, **kwargs)
        finally:
            reduced = _grads(params)
            _write(
                "grads_reduced",
                reduced,
                {**meta, "phase": "post_reduce", "dtypes": _grad_dtypes(reduced)},
            )

    miles_model.finalize_model_grads = wrapped
    try:
        yield
    finally:
        miles_model.finalize_model_grads = original


@contextmanager
def capture_optim_step(
    model: Any, slot_optimizers: dict[int, Any], adam_params_by_slot: dict[int, dict]
) -> Iterator[Any]:
    """Snapshot slot params around the Adam step; yields a dict to fill with outcomes."""
    if not enabled():
        yield None
        return
    params = _slot_params(model, slot_optimizers)
    before = {name: param.detach().clone() for name, param in params.items()}
    _write("params_before", before)
    outcomes: dict[int, dict] = {}
    yield outcomes
    after = {name: param.detach().clone() for name, param in params.items()}
    _write("params_after", after)
    _write("delta", {name: after[name] - before[name] for name in params})
    _calls["opt"] += 1
    _write(
        "optim_step",
        {},
        {
            "call_index": _calls["opt"] - 1,
            "adam_params": adam_params_by_slot,
            "outcomes": outcomes,
        },
    )
