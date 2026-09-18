from __future__ import annotations

from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor

from .loss_scaling import configure_deterministic_loss_scaling


class LiloMilesTrainRayActor(MultiLoRATrainRayActor):
    """Upstream multi-LoRA actor with Qwen MTP and weights-only save support."""

    def init(self, args, role, **kwargs):
        if getattr(args, "lilo_tp_reduce_precision", None):
            from .precision import install_tp_reductions

            install_tp_reductions(args.lilo_tp_reduce_precision)

        deterministic_attention = getattr(args, "lilo_deterministic_attention", False)
        if deterministic_attention:
            from .batching import configure_deterministic_batching
            from .precision import (
                configure_deterministic_attention,
                configure_deterministic_gdn,
                configure_deterministic_losses,
            )

            configure_deterministic_attention()
            configure_deterministic_losses()
            configure_deterministic_loss_scaling()
            configure_deterministic_gdn()
            configure_deterministic_batching()

        # Miles's LoRA builder inherits checkpoint MTP heads without honoring
        # enable_mtp_training. Qwen3.5 then injects an auxiliary backward loss
        # even for a client datum whose weights are all zero.
        from megatron.bridge import AutoBridge

        if args.enable_mtp_training and not deterministic_attention:
            return super().init(args, role, **kwargs)
        original = AutoBridge.to_megatron_provider

        def provider_without_mtp(bridge, *provider_args, **provider_kwargs):
            provider = original(bridge, *provider_args, **provider_kwargs)
            if not args.enable_mtp_training:
                provider.mtp_num_layers = None
                provider.mtp_hybrid_override_pattern = None
            if deterministic_attention:
                provider.batch_invariant_mode = True
            return provider

        AutoBridge.to_megatron_provider = provider_without_mtp
        try:
            return super().init(args, role, **kwargs)
        finally:
            AutoBridge.to_megatron_provider = original

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
