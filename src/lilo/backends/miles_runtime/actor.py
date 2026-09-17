from __future__ import annotations

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
            return super().forward_backward(unit_id, rollout_data_ref)

    def optim_step(self, adam_params_by_slot):
        import torch

        with torch.profiler.record_function("lilo/optim_step"):
            return super().optim_step(adam_params_by_slot)

    def forward_only(self, unit_id, rollout_data_ref):
        import torch

        with torch.profiler.record_function("lilo/forward_only"):
            return super().forward_only(unit_id, rollout_data_ref)

    def export_slot(self, slot, rank, alpha, path, metadata=None):
        import torch

        with torch.profiler.record_function("lilo/export_slot"):
            return super().export_slot(slot, rank, alpha, path, metadata=metadata)

    def torch_profile_start(self) -> None:
        from .profiling import RankProfiler

        self._lilo_profiler = RankProfiler()
        self._lilo_profiler.start()

    def torch_profile_stop(self, output_dir: str) -> dict | None:
        profiler = self._lilo_profiler
        if profiler is None:
            return None
        import torch.distributed as dist

        self._lilo_profiler = None
        return profiler.stop(output_dir, f"rank{dist.get_rank()}")

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
