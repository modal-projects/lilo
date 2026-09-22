"""Validate replay tensors before admitting work to distributed execution."""

from __future__ import annotations

from math import prod

from lilo.replay import REPLAY_FIELDS


def replay_row(inputs, targets: list[int]) -> dict:
    row = {}
    n = len(targets)
    for name in REPLAY_FIELDS & inputs.keys():
        tensor = inputs[name]
        values = list(tensor.data)
        shape = list(tensor.shape or ())
        if (
            tensor.dtype != "int64"
            or tensor.sparse_crow_indices is not None
            or any(type(v) is not int for v in values)
            or prod(shape) != len(values)
        ):
            raise ValueError(
                f"{name} must be a dense int64 tensor with a matching shape"
            )
        if any(
            v < (-1 if name == "routed_experts" else 0) or v > 2**31 - 1 for v in values
        ):
            raise ValueError(f"{name} values are outside the supported int32 range")
        if name == "routed_experts":
            if len(shape) != 3 or shape[0] != n or min(shape[1:]) < 1:
                raise ValueError(
                    "routed_experts must have shape [input_tokens, layers, experts_per_token]"
                )
            row[name] = {"values": values, "shape": shape}
        else:
            if shape != [len(values)]:
                raise ValueError(f"{name} must be one-dimensional")
            row[name] = values
    ids, offsets = row.get("sampling_mask_ids"), row.get("sampling_mask_offsets")
    if (ids is None) != (offsets is None):
        raise ValueError(
            "sampling_mask_ids and sampling_mask_offsets must be supplied together"
        )
    if offsets is not None:
        if (
            len(offsets) != n + 1
            or offsets[0] != 0
            or offsets[-1] != len(ids)
            or any(b < a for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError(
                "sampling mask offsets must have input_tokens + 1 entries, start at zero, and end at the id count"
            )
        for token, a, b in zip(targets, offsets, offsets[1:], strict=False):
            if a != b and (token not in ids[a:b] or len(set(ids[a:b])) != b - a):
                raise ValueError(
                    "each sampling support must contain its target and have no duplicate ids"
                )
    return row


def add_replay_to_train_data(train_data: dict, datums: list[dict]) -> None:
    import numpy as np

    routes = [datum.get("routed_experts") for datum in datums]
    if any(r is not None for r in routes):
        if not all(r is not None for r in routes):
            raise ValueError(
                "router replay must be supplied for every datum in a batch"
            )
        train_data["rollout_routed_experts"] = [
            np.asarray(r["values"], dtype=np.int32).reshape(r["shape"]) for r in routes
        ]
    if any("sampling_mask_ids" in d for d in datums):
        train_data["rollout_sampling_mask_ids"] = [
            np.asarray(d.get("sampling_mask_ids", []), dtype=np.int32) for d in datums
        ]
        train_data["rollout_sampling_mask_offsets"] = [
            np.asarray(
                d.get("sampling_mask_offsets", [0] * (d["target_len"] + 1)),
                dtype=np.int64,
            )
            for d in datums
        ]


def install_bridge_replay() -> None:
    from miles.tinker import runtime

    original = runtime._build_train_data
    if getattr(original, "_lilo_replay", False):
        return

    def build(slot_datums):
        result = original(slot_datums)
        add_replay_to_train_data(result, [datum for _, datum in slot_datums])
        return result

    build._lilo_replay = True
    runtime._build_train_data = build
