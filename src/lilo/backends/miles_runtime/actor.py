from __future__ import annotations

import logging

from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor


class LiloMilesTrainRayActor(MultiLoRATrainRayActor):
    """Upstream multi-LoRA actor with Qwen MTP and weights-only save support."""

    def init(self, args, role, **kwargs):
        # Miles's LoRA builder inherits checkpoint MTP heads without honoring
        # enable_mtp_training. Qwen3.5 then injects an auxiliary backward loss
        # even for a client datum whose weights are all zero.
        from megatron.bridge import AutoBridge

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

    def forward_backward(self, unit_id, rollout_data_ref):
        import torch

        with torch.profiler.record_function("lilo/forward_backward"):
            result = super().forward_backward(unit_id, rollout_data_ref)
        self._log_peak_memory("forward_backward")
        return result

    def optim_step(self, adam_params_by_slot):
        import torch

        with torch.profiler.record_function("lilo/optim_step"):
            result = super().optim_step(adam_params_by_slot)
        self._log_peak_memory("optim_step")
        return result

    def forward_only_logprobs(self, unit_id, rollout_data_ref):
        import torch

        with torch.profiler.record_function("lilo/forward_only_logprobs"):
            result = super().forward_only_logprobs(unit_id, rollout_data_ref)
        self._log_peak_memory("forward_only_logprobs")
        return result

    @staticmethod
    def _log_peak_memory(operation: str) -> None:
        import torch

        gib = 1024**3
        logging.getLogger(__name__).info(
            "lilo_memory op=%s allocated_gb=%.2f max_allocated_gb=%.2f "
            "reserved_gb=%.2f max_reserved_gb=%.2f",
            operation,
            torch.cuda.memory_allocated() / gib,
            torch.cuda.max_memory_allocated() / gib,
            torch.cuda.memory_reserved() / gib,
            torch.cuda.max_memory_reserved() / gib,
        )

    def torch_profile_start(self) -> None:
        from .profiling import RankProfiler

        self._lilo_profiler = RankProfiler()
        self._lilo_profiler.start()

    def torch_profile_stop(self, output_dir: str) -> dict | None:
        profiler = getattr(self, "_lilo_profiler", None)
        if profiler is None:
            return None
        import torch.distributed as dist

        self._lilo_profiler = None
        return profiler.stop(output_dir, f"rank{dist.get_rank()}")

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
        import torch

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
        from megatron.core import dist_checkpointing
        from miles.backends.megatron_utils.lora import checkpoint
        from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir

        weights = checkpoint._slot_weights_sharded_state_dict(self.model, slot)
        sharded = {checkpoint._WEIGHTS_KEY: weights}
        checkpoint._canonicalize_slot_keys(sharded, slot)
        write_checkpoint_dir(
            path, lambda temporary: dist_checkpointing.save(sharded, str(temporary))
        )
