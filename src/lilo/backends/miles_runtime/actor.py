from __future__ import annotations

import logging

from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor


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


def _pad_local_shard(
    tensor,
    total_length: int,
    response_length: int,
    *,
    qkv_format: str,
    max_seq_len,
):
    """Place this CP rank's zigzag logprob shard into a full-length response.

    Replicates the placement logic of miles'
    ``all_gather_with_cp`` without its differentiable ``dist.nn.all_reduce``.
    """
    import torch
    from miles.backends.training_utils.cp_utils import (
        get_logits_and_tokens_offset_with_cp,
    )

    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len
    )
    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(length: int):
        return torch.zeros(
            [length] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        padded = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        padded = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        padded = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        padded = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert padded.shape[0] == response_length, (
        f"Expected {response_length}, got {padded.shape}"
    )
    return padded


def _gather_tinker_logprobs_across_cp() -> None:
    """Reassemble full-response logprobs for Miles' Tinker loss path under CP>1."""
    import torch.distributed as dist
    from miles.backends.training_utils.loss_hub import logit_processors
    from miles.backends.training_utils.parallel import get_parallel_state

    original = logit_processors.get_log_probs_and_entropy
    if getattr(original, "__lilo_gathers_cp__", False):
        return

    def get_log_probs_and_entropy(
        logits,
        *,
        args,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        with_entropy=False,
        entropy_requires_grad=True,
        non_loss_data=True,
        max_seq_lens=None,
        rollout_sampling_mask=None,
    ):
        out = original(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            non_loss_data=non_loss_data,
            max_seq_lens=max_seq_lens,
            rollout_sampling_mask=rollout_sampling_mask,
        )
        parallel_state = get_parallel_state()
        if parallel_state.cp.size == 1 or getattr(args, "allgather_cp", False):
            return out
        log_probs = []
        for index, (lp, total_length, response_length) in enumerate(
            zip(out["log_probs"], total_lengths, response_lengths, strict=True)
        ):
            max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
            padded = _pad_local_shard(
                lp,
                total_length,
                response_length,
                qkv_format=args.qkv_format,
                max_seq_len=max_seq_len,
            )
            summed = padded.detach().clone()
            dist.all_reduce(summed, group=parallel_state.cp.group)
            log_probs.append(padded + (summed - padded.detach()))
        out["log_probs"] = log_probs
        return out

    get_log_probs_and_entropy.__lilo_gathers_cp__ = True
    logit_processors.get_log_probs_and_entropy = get_log_probs_and_entropy

    # Callers on the multi-LoRA path that bound the name at import time must
    # be re-pointed. Miles' non-Tinker losses handle CP natively.
    import miles.backends.megatron_utils.model as megatron_model
    from miles.backends.training_utils.loss_hub import tinker_losses

    for module in (tinker_losses, megatron_model):
        if getattr(module, "get_log_probs_and_entropy", None) is original:
            module.get_log_probs_and_entropy = get_log_probs_and_entropy


_gather_tinker_logprobs_across_cp()


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
