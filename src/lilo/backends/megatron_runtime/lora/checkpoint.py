"""Read and write rank-local Megatron training checkpoints.

These checkpoints support resuming training with a compatible model and
parallel topology. They are not portable model exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

from ..common.checkpoint_io import (
    commit_checkpoint_volume,
    rank_tag,
    reload_checkpoint_volume,
    write_checkpoint_metadata,
)


def write_training_checkpoint(
    uri: str,
    *,
    adapter_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    optimizer_step: int = 0,
    metadata: dict[str, Any],
    persistence_group,
) -> str:
    output = Path(uri)
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "adapter_megatron": adapter_state,
        "optimizer_step": optimizer_step,
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    torch.save(payload, output / f"checkpoint_{rank_tag()}.pt")
    write_checkpoint_metadata(uri, metadata)
    commit_checkpoint_volume(persistence_group)
    return uri


def load_training_checkpoint(uri: str) -> dict[str, Any]:
    reload_checkpoint_volume()
    path = Path(uri) / f"checkpoint_{rank_tag()}.pt"
    if not path.exists():
        raise FileNotFoundError(f"training checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"invalid training checkpoint: {path}")
    return checkpoint


def extract_adapter_state(*, model, slot: int) -> dict[str, Any]:
    state = {}
    with expose_adapter_slot(model, slot):
        for chunk in model:
            for name, parameter in chunk.named_parameters():
                if ".adapter." in name:
                    state[name] = parameter.data.detach().cpu().clone()
    return state
