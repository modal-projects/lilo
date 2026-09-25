"""Bridge sampler replay into Miles' Tinker loss and multi-LoRA execution path.

Miles already owns CP/SP packing and expert-router replay. These hooks provide
its native replay queues and sampling-mask primitive with Tinker datum metadata.
"""

from __future__ import annotations

from argparse import Namespace
from functools import wraps

import torch

_replayed_run = None


class TinkerSamplingMask:
    """CSR support over target positions; empty rows leave logits unmasked."""

    def __init__(self, ids, offsets):
        self._ids = torch.as_tensor(ids, dtype=torch.int32).clone()
        self._offsets = torch.as_tensor(offsets, dtype=torch.long).clone()

    def __len__(self):
        return self._offsets.numel() - 1

    def _select_masks(self, response_indices):
        # Match Miles' mask protocol while allowing unrestricted prompt rows.
        if isinstance(response_indices, range) and response_indices.step == 1:
            start, stop = response_indices.start, response_indices.stop
            lengths = self._offsets[start + 1 : stop + 1] - self._offsets[start:stop]
            return self._ids[self._offsets[start] : self._offsets[stop]], lengths
        indices = torch.as_tensor(response_indices, dtype=torch.long)
        starts = self._offsets[indices]
        lengths = self._offsets[indices + 1] - starts
        output_starts = lengths.cumsum(0) - lengths
        positions = torch.arange(int(lengths.sum())) + torch.repeat_interleave(
            starts - output_starts, lengths
        )
        return self._ids[positions], lengths


def build_tinker_sampling_mask(
    original, logits, sampling_mask, response_indices, *, tp_rank
):
    mask = original(logits, sampling_mask, response_indices, tp_rank=tp_rank)
    if isinstance(sampling_mask, TinkerSamplingMask):
        _, lengths = sampling_mask._select_masks(response_indices)
        mask[lengths.to(logits.device) == 0] = True
    return mask


def install_replay_hooks(
    *,
    megatron_model,
    lora_model,
    logit_processors,
    tinker_losses,
    fill_replay_data,
    manager,
) -> None:
    # The GPU actor imports Miles; this module also serves CPU-side mask tests.
    global _replayed_run
    if lora_model.run_forward_backward is _replayed_run:
        return

    original_mask = logit_processors.build_local_sampling_mask

    @wraps(original_mask)
    def local_mask(logits, sampling_mask, response_indices, *, tp_rank):
        return build_tinker_sampling_mask(
            original_mask, logits, sampling_mask, response_indices, tp_rank=tp_rank
        )

    logit_processors.build_local_sampling_mask = local_mask

    original_batch = megatron_model.get_batch

    @wraps(original_batch)
    def get_batch(iterator, keys, *args, **kwargs):
        for key in ("rollout_sampling_mask_ids", "rollout_sampling_mask_offsets"):
            if key in iterator.rollout_data and key not in keys:
                keys = [*keys, key]
        return original_batch(iterator, keys, *args, **kwargs)

    megatron_model.get_batch = get_batch
    original_target = tinker_losses._target_logprobs

    @wraps(original_target)
    def target_logprobs(args, batch, logits):
        if "rollout_sampling_mask_ids" not in batch:
            return original_target(args, batch, logits)
        masks = [
            TinkerSamplingMask(ids, offsets)
            for ids, offsets in zip(
                batch["rollout_sampling_mask_ids"],
                batch["rollout_sampling_mask_offsets"],
                strict=True,
            )
        ]
        vocab_size = args.vocab_size
        if vocab_size is None:
            raise ValueError("sampling replay requires the true model vocabulary size")
        if any(torch.any(mask._ids >= vocab_size) for mask in masks):
            raise ValueError(
                "sampling mask contains a token outside the model vocabulary"
            )
        args = Namespace(**vars(args))
        args.rollout_temperature = batch["loss_fn_config"]["sampling_temperature"]
        label_tokens = [
            torch.cat(
                [
                    tokens[: len(tokens) - len(targets)],
                    torch.as_tensor(targets, dtype=tokens.dtype, device=tokens.device),
                ]
            )
            for tokens, targets in zip(
                batch["unconcat_tokens"], batch["target_tokens"], strict=True
            )
        ]
        return tinker_losses.get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=label_tokens,
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            with_entropy=False,
            max_seq_lens=batch.get("max_seq_lens"),
            rollout_sampling_mask=masks,
        )["log_probs"]

    tinker_losses._target_logprobs = target_logprobs
    original_iterator = lora_model.get_data_iterator

    @wraps(original_iterator)
    def get_iterator(args, model, rollout_data):
        iterators, num_microbatches = original_iterator(args, model, rollout_data)
        if manager.data_key in rollout_data:
            fill_replay_data(
                args=args,
                models=model,
                data_iterator=iterators,
                num_microbatches=num_microbatches,
                rollout_data=rollout_data,
                data_key=manager.data_key,
                replay_list=manager.replays,
                register_replay_list_func=manager.register_replay_list_func,
                if_sp_region=manager.if_sp_region,
                indices_are_token_positions=manager.replay_indices_are_token_positions,
            )
        return iterators, num_microbatches

    lora_model.get_data_iterator = get_iterator
    original_run = lora_model.run_forward_backward

    @wraps(original_run)
    def run(args, batch_id, model, rollout_data, **kwargs):
        replay = manager.data_key in rollout_data
        enabled, stage = manager.enabled, manager.stage
        if replay and (not enabled or not manager.replays):
            raise ValueError(
                "router replay requires a MoE trainer initialized with --use-rollout-routing-replay"
            )
        try:
            manager.clear_all()
            manager.enabled = replay
            # Miles switches to replay_forward around each forward; recompute
            # consumes the separate backward cursor after that stage restores.
            manager.stage = "replay_backward"
            return original_run(args, batch_id, model, rollout_data, **kwargs)
        finally:
            manager.clear_all()
            manager.enabled, manager.stage = enabled, stage

    _replayed_run = run
    lora_model.run_forward_backward = run
