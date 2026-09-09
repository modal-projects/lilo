from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from miles.backends.megatron_utils.actor import MegatronTrainRayActor


def _preserve_advantages_in_dp_shards() -> None:
    """Work around radixark/miles#3145 omitting Tinker advantages from DP shards."""

    from miles.ray.rollout import train_data_conversion

    original = train_data_conversion._package_shards
    if getattr(original, "__lilo_preserves_advantages__", False):
        return

    def package_shards(args, data, partitions):
        shards = original(args, data, partitions)
        if "advantages" in data:
            for shard, partition in zip(shards, partitions, strict=True):
                shard["advantages"] = [data["advantages"][index] for index in partition]
        return shards

    package_shards.__lilo_preserves_advantages__ = True
    train_data_conversion._package_shards = package_shards


_preserve_advantages_in_dp_shards()


class LiloMilesTrainRayActor(MegatronTrainRayActor):
    """Miles training actor with one filesystem PEFT export command."""

    def export_slot_peft(
        self,
        *,
        slot: int,
        path: str,
        rank: int,
        alpha: float,
        base_model: str,
        target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> None:
        import torch.distributed as dist
        from miles.utils.multi_lora import AdapterSpec

        writer = dist.get_rank() == 0
        prefix = "__lilo_export__:"
        state = {}
        iterator = self.weight_updater._hf_weight_iterator
        for bucket in iterator.iter_hf_weights(
            None,
            include_base=False,
            adapters=(("__lilo_export__", AdapterSpec(slot, rank, alpha)),),
            materialize=writer,
        ):
            if not writer:
                continue
            for name, tensor in bucket:
                if not name.startswith(prefix):
                    raise RuntimeError(f"unexpected exported adapter name: {name}")
                state[name.removeprefix(prefix)] = tensor.detach().cpu().contiguous()

        if not writer:
            return
        if not state:
            raise RuntimeError(f"slot {slot} exported no adapter tensors")

        from safetensors.torch import save_file

        destination = Path(path)
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True)
        try:
            save_file(
                dict(sorted(state.items())),
                temporary / "adapter_model.safetensors",
                metadata={"format": "pt"},
            )
            config = {
                "base_model_name_or_path": base_model,
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "lora_alpha": alpha,
                "lora_dropout": lora_dropout,
                "peft_type": "LORA",
                "r": rank,
                "target_modules": list(target_modules),
                "task_type": "CAUSAL_LM",
            }
            (temporary / "adapter_config.json").write_text(
                json.dumps(config, sort_keys=True),
                encoding="utf-8",
            )
            if destination.exists():
                raise FileExistsError(f"adapter capture already exists: {destination}")
            os.replace(temporary, destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
