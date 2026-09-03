from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal
import torch.distributed as dist
from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
from megatron.core import parallel_state
from safetensors.torch import save_file
from stitch.types import VersionRef

from lilo.inference.bulletin import SnapshotBulletin

from .bridge import patch_megatron_model

_TARGET_MODULES = {
    "linear_qkv": ("attn", ("q_proj", "k_proj", "v_proj")),
    "linear_q": ("attn", ("q_proj",)),
    "linear_k": ("attn", ("k_proj",)),
    "linear_v": ("attn", ("v_proj",)),
    "linear_proj": ("attn", ("o_proj",)),
    "linear_fc1": ("mlp", ("gate_proj", "up_proj")),
    "linear_fc1_gate": ("mlp", ("gate_proj",)),
    "linear_fc1_up": ("mlp", ("up_proj",)),
    "linear_fc2": ("mlp", ("down_proj",)),
    "output_layer": ("unembed", ("lm_head",)),
    "in_proj": (
        "attn",
        ("in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"),
    ),
    "out_proj": ("attn", ("out_proj",)),
    "q_proj": ("attn", ("q_proj",)),
    "k_proj": ("attn", ("k_proj",)),
    "v_proj": ("attn", ("v_proj",)),
    "o_proj": ("attn", ("o_proj",)),
    "gate_proj": ("mlp", ("gate_proj",)),
    "up_proj": ("mlp", ("up_proj",)),
    "down_proj": ("mlp", ("down_proj",)),
    "lm_head": ("unembed", ("lm_head",)),
}
_PROJECTION_INDEX = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "gate_proj": 0,
    "up_proj": 1,
    "in_proj_q": 0,
    "in_proj_k": 1,
    "in_proj_v": 2,
    "in_proj_z": 3,
}


@dataclass(frozen=True)
class CapturedAdapterSnapshot:
    model_id: str
    publish_version: int
    state: dict[str, Any]
    config: dict[str, Any]
    bulletin_root: str
    bulletin_volume: str
    writer: bool


def peft_target_modules(
    target_modules: Sequence[str],
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> list[str]:
    return _peft_target_modules(
        target_modules,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=train_unembed,
        split_mamba_weights=False,
    )


def _peft_target_modules(
    target_modules: Sequence[str],
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
    split_mamba_weights: bool,
) -> list[str]:
    enabled = {
        "attn": train_attn,
        "mlp": train_mlp,
        "unembed": train_unembed,
    }
    targets = []
    for module in target_modules:
        name = module.rsplit(".", 1)[-1]
        if name == "in_proj" and "mixer" in module.split("."):
            converted = ("gate_proj", "x_proj") if split_mamba_weights else ("in_proj",)
            category = "attn"
        else:
            category, converted = _TARGET_MODULES.get(name, (None, (name,)))
        if category is not None and not enabled[category]:
            continue
        for target in converted:
            if target not in targets:
                targets.append(target)
    return targets


def slice_lora_to_rank(
    hf_name: str,
    tensor: Any,
    adapter_rank: int,
    *,
    fused_projection_rank_layout: bool = True,
) -> Any:
    axis = 0 if "lora_A" in hf_name else 1 if "lora_B" in hf_name else None
    if axis is None:
        return tensor
    if len(tensor.shape) <= axis:
        raise ValueError(f"invalid LoRA tensor shape for {hf_name}: {tensor.shape}")
    padded_rank = tensor.shape[axis]
    if padded_rank < adapter_rank:
        raise ValueError(
            f"LoRA tensor rank is smaller than job rank for {hf_name}: "
            f"{padded_rank} < {adapter_rank}"
        )
    if padded_rank == adapter_rank:
        return tensor
    projection_index = (
        next(
            (
                _PROJECTION_INDEX[part]
                for part in hf_name.split(".")
                if part in _PROJECTION_INDEX
            ),
            None,
        )
        if fused_projection_rank_layout
        else None
    )
    if projection_index is not None:
        start = projection_index * adapter_rank
        stop = start + adapter_rank
        if padded_rank < stop:
            raise ValueError(
                f"LoRA tensor rank is smaller than projection for {hf_name}: "
                f"{padded_rank} < {stop}"
            )
        output_slice = [slice(None)] * len(tensor.shape)
        output_slice[axis] = slice(start, stop)
        return tensor[tuple(output_slice)]
    remainder_slice = [slice(None)] * len(tensor.shape)
    remainder_slice[axis] = slice(adapter_rank, None)
    remainder = tensor[tuple(remainder_slice)]
    maximum = remainder.abs().max().item()
    if maximum != 0:
        raise ValueError(
            f"LoRA padded dimensions are non-zero for {hf_name}: {maximum}"
        )
    output_slice = [slice(None)] * len(tensor.shape)
    output_slice[axis] = slice(0, adapter_rank)
    return tensor[tuple(output_slice)]


def collect_adapter_weights(
    bridge,
    model,
    *,
    slot: int,
    adapter_rank: int,
    split_qkv: bool = False,
    split_gdn: bool = False,
    split_mamba: bool = False,
) -> dict[str, Any]:
    exported = {}
    with expose_adapter_slot(model, slot), patch_megatron_model(model):
        for hf_name, weight, _ in bridge.export_adapter_weights(
            model,
            cpu=True,
            show_progress=False,
        ):
            exported[hf_name] = weight
    return {
        hf_name: slice_lora_to_rank(
            hf_name,
            weight,
            adapter_rank,
            fused_projection_rank_layout=not (
                split_qkv
                and any(
                    projection in hf_name.split(".")
                    for projection in ("q_proj", "k_proj", "v_proj")
                )
                or split_gdn
                and any(
                    projection in hf_name.split(".")
                    for projection in (
                        "in_proj_q",
                        "in_proj_k",
                        "in_proj_v",
                        "in_proj_z",
                    )
                )
                or split_mamba
                and "mixer" in hf_name.split(".")
                and any(
                    projection in hf_name.split(".")
                    for projection in ("gate_proj", "x_proj")
                )
            ),
        ).clone()
        for hf_name, weight in _split_gdn_adapter_weights(exported).items()
    }


def _split_gdn_adapter_weights(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    for a_name in tuple(state):
        if ".in_proj_qkv.lora_A.weight" not in a_name:
            continue
        b_name = a_name.replace(".lora_A.weight", ".lora_B.weight")
        z_b_name = b_name.replace(".in_proj_qkv.", ".in_proj_z.")
        if b_name not in state or z_b_name not in state:
            raise ValueError(f"incomplete GDN adapter export for {a_name}")
        qkv_a = state.pop(a_name)
        qkv_b = state.pop(b_name)
        v_dim = int(state[z_b_name].shape[0])
        qk_total = int(qkv_b.shape[0]) - v_dim
        if qk_total <= 0 or qk_total % 2:
            raise ValueError(
                f"invalid GDN qkv adapter shape for {b_name}: {qkv_b.shape}"
            )
        qk_dim = qk_total // 2
        offset = 0
        for target, size in (
            ("in_proj_q", qk_dim),
            ("in_proj_k", qk_dim),
            ("in_proj_v", v_dim),
        ):
            target_a = a_name.replace("in_proj_qkv", target)
            target_b = b_name.replace("in_proj_qkv", target)
            state[target_a] = qkv_a
            state[target_b] = qkv_b[offset : offset + size, :]
            offset += size
    for name in tuple(state):
        if not any(f".{target}." in name for target in ("in_proj_b", "in_proj_a")):
            continue
        if ".lora_B.weight" in name:
            maximum = state[name].abs().max().item()
            if maximum != 0:
                raise ValueError(
                    f"disabled GDN adapter projection is non-zero for {name}"
                )
        del state[name]
    return state


def capture_adapter_snapshot(
    *,
    bridge,
    model,
    slot: int,
    model_id: str,
    publish_version: int,
    base_model: str,
    adapter_rank: int,
    adapter_alpha: float,
    lora_dropout: float,
    split_qkv: bool,
    split_gdn: bool,
    split_mamba: bool,
    target_modules: Sequence[str],
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
    bulletin_root: str,
    bulletin_volume: str,
) -> CapturedAdapterSnapshot:
    state = collect_adapter_weights(
        bridge,
        model,
        slot=slot,
        adapter_rank=adapter_rank,
        split_qkv=split_qkv,
        split_gdn=split_gdn,
        split_mamba=split_mamba,
    )
    enabled_targets = peft_target_modules(
        target_modules,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=train_unembed,
    )
    weight_targets = _peft_target_modules(
        target_modules,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=train_unembed,
        split_mamba_weights=split_mamba,
    )
    state = {
        name: weight
        for name, weight in state.items()
        if any(target in name.split(".") for target in weight_targets)
    }
    writer = (
        dist.get_rank() == 0
        and parallel_state.get_data_parallel_rank(with_context_parallel=False) == 0
        and parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_pipeline_model_parallel_rank() == 0
    )
    if not writer:
        return CapturedAdapterSnapshot(
            model_id=model_id,
            publish_version=publish_version,
            state={},
            config={},
            bulletin_root=bulletin_root,
            bulletin_volume=bulletin_volume,
            writer=False,
        )
    missing_targets = [
        target
        for target in weight_targets
        if not any(target in name.split(".") for name in state)
    ]
    if missing_targets:
        raise ValueError(
            f"bridge exported no weights for PEFT targets: {missing_targets}"
        )

    config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": adapter_alpha,
        "lora_dropout": lora_dropout,
        "peft_type": "LORA",
        "r": adapter_rank,
        "target_modules": enabled_targets,
        "task_type": "CAUSAL_LM",
    }
    return CapturedAdapterSnapshot(
        model_id=model_id,
        publish_version=publish_version,
        state=state,
        config=config,
        bulletin_root=bulletin_root,
        bulletin_volume=bulletin_volume,
        writer=True,
    )


def persist_adapter_snapshot(snapshot: CapturedAdapterSnapshot) -> bool:
    if not snapshot.writer:
        return False

    board = SnapshotBulletin(
        Path(snapshot.bulletin_root),
        commit=modal.Volume.from_name(
            snapshot.bulletin_volume,
            version=2,
        ).commit,
    )
    with tempfile.TemporaryDirectory(prefix="lilo-peft-") as temp:
        source = Path(temp)
        save_file(
            {name: snapshot.state[name] for name in sorted(snapshot.state)},
            str(source / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        (source / "adapter_config.json").write_text(
            json.dumps(
                snapshot.config,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return board.publish(
            VersionRef(snapshot.model_id, snapshot.publish_version),
            source,
        )
