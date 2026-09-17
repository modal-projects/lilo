from lilo.backends.miles_runtime.batching import pack_slot_microbatches
import pytest


def test_packing_preserves_each_clients_microbatches_and_order():
    lengths = [4, 3, 8, 2, 5, 5, 1, 1]
    slots = [0, 0, 0, 1, 1, 1, 2, 2]
    canonical = {0: [[0, 1], [2]], 1: [[3, 4], [5]], 2: [[6, 7]]}
    packed = pack_slot_microbatches(canonical, lengths, 10)
    assert sorted(i for batch in packed for i in batch) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in batch) <= 10 for batch in packed)
    assert any(len({slots[i] for i in batch}) > 1 for batch in packed)
    for slot, expected in canonical.items():
        projected = [[i for i in batch if slots[i] == slot] for batch in packed]
        assert [batch for batch in projected if batch] == expected
    assert (
        pack_slot_microbatches(dict(reversed(list(canonical.items()))), lengths, 10)
        == packed
    )


def test_packing_does_not_merge_two_microbatches_from_one_client():
    assert pack_slot_microbatches({0: [[0], [1]]}, [2, 2], 10) == [[0], [1]]


def test_packing_rejects_an_oversized_client_microbatch():
    with pytest.raises(ValueError, match="exceeds"):
        pack_slot_microbatches({0: [[0, 1]]}, [6, 6], 10)


def test_deterministic_hook_uses_upstream_lora_model(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from lilo.backends.miles_runtime.batching import configure_deterministic_batching

    # The merged Miles implementation has lora.model and no lora.executor.
    modules = {}
    for name in (
        "miles",
        "miles.backends",
        "miles.backends.megatron_utils",
        "miles.backends.megatron_utils.actor",
        "miles.backends.megatron_utils.lora",
        "miles.backends.megatron_utils.lora.model",
        "miles.backends.training_utils",
        "miles.backends.training_utils.data",
        "miles.utils",
        "miles.utils.data",
        "miles.utils.seqlen_balancing",
    ):
        module = modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    monkeypatch.delitem(
        sys.modules, "miles.backends.megatron_utils.lora.executor", raising=False
    )
    lora_model = modules["miles.backends.megatron_utils.lora.model"]
    actor = modules["miles.backends.megatron_utils.actor"]
    original = lambda args, model, rollout: rollout
    lora_model.get_data_iterator = actor.get_data_iterator = original
    modules["miles.backends.training_utils.data"].get_parallel_state = lambda: (
        SimpleNamespace(
            effective_dp=SimpleNamespace(size=1), vpp_size=1, cp=SimpleNamespace(size=1)
        )
    )
    modules["miles.utils.data"].get_minimum_num_micro_batch_size = (
        lambda lengths, budget: len(lengths)
    )
    modules["miles.utils.seqlen_balancing"].get_seqlen_balanced_partitions = (
        lambda lengths, count, equal_size: [[i] for i in range(count)]
    )

    configure_deterministic_batching()
    plan = lora_model.get_data_iterator
    assert plan is not original
    assert actor.get_data_iterator is plan
    configure_deterministic_batching()
    assert lora_model.get_data_iterator is plan

    args = SimpleNamespace(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    rollout = {
        "adapter_slots": [0, 0, 1, 1],
        "total_lengths": [6, 4, 4, 6],
        "dynamic_global_batch_size": 4,
    }
    prepared = plan(args, None, rollout)
    assert prepared["micro_batch_indices"] == [[0, 2], [1, 3]]
    assert prepared["num_microbatches"] == [2]
    assert "micro_batch_indices" not in rollout
    assert plan(args, None, prepared) is prepared
    ordinary = {"total_lengths": [4]}
    assert plan(args, None, ordinary) is ordinary
