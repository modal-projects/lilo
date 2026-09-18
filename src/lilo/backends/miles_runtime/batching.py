"""Preserve each adapter's gradient accumulation order when packing clients."""

from collections import defaultdict, deque
from functools import wraps


def pack_slot_microbatches(slot_batches, lengths, budget):
    """Combine whole per-slot microbatches, preserving their order and boundaries."""
    pending = {slot: deque(batches) for slot, batches in sorted(slot_batches.items())}
    result = []
    while any(pending.values()):
        indices = []
        used = 0
        for batches in pending.values():
            if not batches:
                continue
            candidate = batches[0]
            size = sum(lengths[i] for i in candidate)
            if size > budget:
                raise ValueError(
                    "A deterministic client microbatch exceeds the token budget"
                )
            if used + size <= budget:
                indices.extend(batches.popleft())
                used += size
        if not indices:
            raise ValueError("Cannot construct a nonempty deterministic microbatch")
        result.append(indices)
    return result


def configure_deterministic_batching():
    from miles.backends.megatron_utils import actor
    from miles.backends.megatron_utils.lora import model as lora_model
    from miles.backends.training_utils import data
    from miles.utils.data import get_minimum_num_micro_batch_size
    from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions

    original = lora_model.get_data_iterator
    if getattr(original, "_lilo_stable_microbatches", False):
        return

    @wraps(original)
    def plan(args, model, rollout_data):
        if (
            "adapter_slots" not in rollout_data
            or "micro_batch_indices" in rollout_data
            or not args.use_dynamic_batch_size
        ):
            return original(args, model, rollout_data)
        parallel = data.get_parallel_state()
        if parallel.effective_dp.size != 1 or parallel.vpp_size != 1:
            raise ValueError(
                "Deterministic multi-LoRA batching requires DP=1 and VPP=1"
            )
        if any(
            x is not None for x in rollout_data.get("multimodal_train_inputs", []) or []
        ):
            raise ValueError(
                "Deterministic multi-LoRA batching currently supports text inputs"
            )
        lengths = rollout_data["total_lengths"]
        if rollout_data.get("dynamic_global_batch_size") != len(lengths):
            raise ValueError(
                "Deterministic multi-LoRA batching expects one update per work unit"
            )
        by_slot = defaultdict(list)
        for i, slot in enumerate(rollout_data["adapter_slots"]):
            by_slot[int(slot)].append(i)
        budget = args.max_tokens_per_gpu * parallel.cp.size
        slot_batches = {}
        for slot, indices in by_slot.items():
            sizes = [lengths[i] for i in indices]
            count = get_minimum_num_micro_batch_size(sizes, budget)
            partitions = get_seqlen_balanced_partitions(sizes, count, equal_size=False)
            slot_batches[slot] = [[indices[i] for i in part] for part in partitions]
        schedule = pack_slot_microbatches(slot_batches, lengths, budget)
        prepared = dict(rollout_data)
        prepared.update(micro_batch_indices=schedule, num_microbatches=[len(schedule)])
        return original(args, model, prepared)

    plan._lilo_stable_microbatches = True
    lora_model.get_data_iterator = plan
    if actor.get_data_iterator is original:
        actor.get_data_iterator = plan
