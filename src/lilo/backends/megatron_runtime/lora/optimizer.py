from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
from megatron.core.distributed import finalize_model_grads

from ..common.optimizer import set_adam_params
from .adapters import (
    iter_adapter_named_params_for_slot,
    restore_adapter_grads,
    snapshot_adapter_grads_for_slots,
    zero_adapter_grads_for_slots,
)

OptimizerState = dict[tuple[int, str], dict[str, Any]]
_PARAM_GROUP_STATE_PREFIX = "\0lilo_param_group:"


def _optimizer_parameter(parameter):
    main_parameter = getattr(parameter, "main_param", None)
    return parameter if main_parameter is None else main_parameter


def _inner_optimizers(optimizer) -> Iterator[tuple[int, Any]]:
    optimizers = getattr(optimizer, "chained_optimizers", (optimizer,))
    for index, wrapped in enumerate(optimizers):
        yield index, getattr(wrapped, "optimizer", wrapped)


def _slot_parameters(model, slot: int) -> dict[str, Any]:
    return {
        name: _optimizer_parameter(parameter)
        for name, parameter in iter_adapter_named_params_for_slot(model, slot)
    }


def _copy_state(
    state: dict[str, Any],
    *,
    cpu: bool,
) -> dict[str, Any]:
    copied = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            value = value.detach().cpu() if cpu else value.detach()
            copied[key] = value.clone()
        else:
            copied[key] = deepcopy(value)
    return copied


def _copy_state_to_device(
    state: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device=device) if torch.is_tensor(value) else deepcopy(value)
        for key, value in state.items()
    }


def _param_group_state_name(group_index: int) -> str:
    return f"{_PARAM_GROUP_STATE_PREFIX}{group_index}"


def _param_group_index(name: str) -> int | None:
    if not name.startswith(_PARAM_GROUP_STATE_PREFIX):
        return None
    return int(name.removeprefix(_PARAM_GROUP_STATE_PREFIX))


def _param_group_device(group: Mapping[str, Any], parameters: Mapping[str, Any]):
    group_parameter = next(iter(group.get("params", ())), None)
    if group_parameter is not None:
        return group_parameter.device
    slot_parameter = next(iter(parameters.values()), None)
    if slot_parameter is not None:
        return slot_parameter.device
    return torch.device("cuda", torch.cuda.current_device())


def capture_optimizer_state_for_adapter(
    optimizer,
    model,
    slot: int,
    *,
    cpu: bool = True,
) -> OptimizerState:
    names_by_parameter = {
        parameter: name for name, parameter in _slot_parameters(model, slot).items()
    }
    captured: OptimizerState = {}

    for optimizer_index, inner in _inner_optimizers(optimizer):
        for parameter, state in inner.state.items():
            name = names_by_parameter.get(parameter)
            if name is not None:
                captured[(optimizer_index, name)] = _copy_state(
                    state,
                    cpu=cpu,
                )
        for group_index, group in enumerate(inner.param_groups):
            if "step" in group:
                captured[(optimizer_index, _param_group_state_name(group_index))] = (
                    _copy_state({"step": group["step"]}, cpu=cpu)
                )

    return captured


def clear_optimizer_state_for_adapter(optimizer, model, slot: int) -> None:
    parameters = set(_slot_parameters(model, slot).values())
    for _, inner in _inner_optimizers(optimizer):
        for parameter in parameters:
            inner.state.pop(parameter, None)
        for group in inner.param_groups:
            group.pop("step", None)


def restore_optimizer_state_for_adapter(
    optimizer,
    model,
    slot: int,
    captured: OptimizerState,
    *,
    optimizer_step: int | None = None,
) -> None:
    parameters = _slot_parameters(model, slot)
    inner_optimizers = dict(_inner_optimizers(optimizer))
    restored_groups: set[tuple[int, int]] = set()

    clear_optimizer_state_for_adapter(optimizer, model, slot)
    for (optimizer_index, name), state in captured.items():
        inner = inner_optimizers.get(optimizer_index)
        if inner is None:
            continue
        group_index = _param_group_index(name)
        if group_index is not None:
            if not 0 <= group_index < len(inner.param_groups):
                continue
            group = inner.param_groups[group_index]
            copied = _copy_state_to_device(
                state,
                device=_param_group_device(group, parameters),
            )
            if "step" in copied:
                group["step"] = copied["step"]
                restored_groups.add((optimizer_index, group_index))
            continue
        parameter = parameters.get(name)
        if parameter is None:
            continue
        inner.state[parameter] = _copy_state_to_device(
            state,
            device=parameter.device,
        )
    if optimizer_step is not None and optimizer_step > 0:
        for optimizer_index, inner in inner_optimizers.items():
            for group_index, group in enumerate(inner.param_groups):
                if (optimizer_index, group_index) not in restored_groups:
                    if getattr(inner, "capturable", False):
                        group["step"] = torch.tensor(
                            [optimizer_step],
                            dtype=torch.int,
                            device=_param_group_device(group, parameters),
                        )
                    else:
                        group["step"] = optimizer_step


def run_optimizer_step(
    optimizers: Mapping[int, Any],
    model,
    *,
    active_slots: Sequence[int],
    preserve_grad_slots: Sequence[int],
    adam,
) -> tuple[bool, float]:
    active_optimizers = [optimizers[slot] for slot in active_slots]
    for optimizer in active_optimizers:
        set_adam_params(optimizer, adam)

    preserved_grads = snapshot_adapter_grads_for_slots(
        model,
        preserve_grad_slots,
    )
    zero_adapter_grads_for_slots(model, preserve_grad_slots)

    try:
        finalize_model_grads(model, None)
        successful = True
        grad_norm_squared = 0.0
        for optimizer in active_optimizers:
            step_successful, grad_norm, _ = optimizer.step()
            successful = successful and bool(step_successful)
            if grad_norm is not None:
                grad_norm_squared += float(grad_norm) ** 2
        return successful, grad_norm_squared**0.5
    finally:
        for chunk in model:
            chunk.zero_grad_buffer()
        for optimizer in active_optimizers:
            optimizer.zero_grad()
        restore_adapter_grads(preserved_grads)
