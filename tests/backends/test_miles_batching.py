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
