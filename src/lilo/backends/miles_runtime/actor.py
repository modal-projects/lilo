from __future__ import annotations

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import dist_checkpointing
from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor
from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir

from .profiling import RankProfiler, TorchProfileConfig


class LiloMilesTrainRayActor(MultiLoRATrainRayActor):
    """Upstream multi-LoRA actor with Qwen MTP and weights-only save support."""

    def init(self, args, role, **kwargs):
        # Miles's LoRA builder inherits checkpoint MTP heads without honoring
        # enable_mtp_training. Qwen3.5 then injects an auxiliary backward loss
        # even for a client datum whose weights are all zero.
        if args.enable_mtp_training:
            return super().init(args, role, **kwargs)
        original = AutoBridge.to_megatron_provider

        def provider_without_mtp(bridge, *provider_args, **provider_kwargs):
            provider = original(bridge, *provider_args, **provider_kwargs)
            provider.mtp_num_layers = None
            provider.mtp_hybrid_override_pattern = None
            return provider

        AutoBridge.to_megatron_provider = provider_without_mtp
        try:
            return super().init(args, role, **kwargs)
        finally:
            AutoBridge.to_megatron_provider = original

    def torch_profile_start(self) -> None:
        if not TorchProfileConfig.from_env().profiles_rank(dist.get_rank()):
            self._lilo_profiler = None
            return
        self._lilo_profiler = RankProfiler()
        self._lilo_profiler.start()

    def torch_profile_stop(self, output_dir: str) -> dict | None:
        profiler = self._lilo_profiler
        if profiler is None:
            return None
        self._lilo_profiler = None
        return profiler.stop(output_dir, f"rank{dist.get_rank()}")

    def forward_backward(self, *args, **kwargs):
        return self._profiled("forward_backward", *args, **kwargs)

    def optim_step(self, *args, **kwargs):
        return self._profiled("optim_step", *args, **kwargs)

    def forward_only(self, *args, **kwargs):
        return self._profiled("forward_only", *args, **kwargs)

    def _profiled(self, operation: str, *args, **kwargs):
        with torch.profiler.record_function(f"lilo/{operation}"):
            result = getattr(super(), operation)(*args, **kwargs)
        self._log_peak_memory(operation)
        return result

    @staticmethod
    def _log_peak_memory(operation: str) -> None:
        gib = 1024**3
        print(
            f"lilo_memory op={operation} "
            f"allocated_gb={torch.cuda.memory_allocated() / gib:.2f} "
            f"max_allocated_gb={torch.cuda.max_memory_allocated() / gib:.2f} "
            f"reserved_gb={torch.cuda.memory_reserved() / gib:.2f} "
            f"max_reserved_gb={torch.cuda.max_memory_reserved() / gib:.2f}",
            flush=True,
        )

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
        with torch.profiler.record_function("lilo/export_slot_peft"):
            return super().export_slot_peft(
                slot=slot,
                path=path,
                rank=rank,
                alpha=alpha,
                base_model=base_model,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
            )

    def save_slot_weights(self, slot: int, path: str) -> None:
        self._save_slot_weights(slot, path)

    def _save_slot_weights(self, slot: int, path: str) -> None:
        weights = checkpoint._slot_weights_sharded_state_dict(self.model, slot)
        sharded = {checkpoint._WEIGHTS_KEY: weights}
        checkpoint._canonicalize_slot_keys(sharded, slot)
        write_checkpoint_dir(
            path, lambda temporary: dist_checkpointing.save(sharded, str(temporary))
        )
