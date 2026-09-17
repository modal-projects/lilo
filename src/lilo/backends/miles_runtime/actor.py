from __future__ import annotations

from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor

from lilo.telemetry.performance import stages


class LiloMilesTrainRayActor(MultiLoRATrainRayActor):
    """Upstream multi-LoRA actor with Qwen MTP and weights-only save support."""

    def forward_backward(self, batch_id, rollout_data_ref):
        with stages("miles_actor").track(
            "forward_backward", attributes={"lilo.batch_id": batch_id}
        ):
            return super().forward_backward(batch_id, rollout_data_ref)

    def forward_only(self, batch_id, rollout_data_ref):
        with stages("miles_actor").track(
            "forward", attributes={"lilo.batch_id": batch_id}
        ):
            return super().forward_only(batch_id, rollout_data_ref)

    def optim_step(self, adam_params_by_slot):
        with stages("miles_actor").track("optimizer"):
            return super().optim_step(adam_params_by_slot)

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
