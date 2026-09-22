import pytest
import torch

from lilo.backends.miles_runtime.replay import (
    TinkerSamplingMask,
    build_tinker_sampling_mask,
)


def reference_local_mask(logits, mask, indices, *, tp_rank):
    """Reference support expansion, independent of CSR vectorized selection."""
    rows = []
    for i in indices:
        allowed = mask._ids[mask._offsets[i] : mask._offsets[i + 1]].tolist()
        rows.append(
            [tp_rank * logits.shape[1] + j in allowed for j in range(logits.shape[1])]
        )
    return torch.tensor(rows, dtype=torch.bool).reshape_as(logits)


@pytest.mark.parametrize("indices", [range(4), [0, 3], [2, 0, 3], []])
def test_csr_selection_preserves_cp_order_and_unmasked_prefix(indices):
    mask = TinkerSamplingMask([1, 4, 2], [0, 0, 0, 2, 3])
    ids, lengths = mask._select_masks(indices)
    supports = [[], [], [1, 4], [2]]
    assert lengths.tolist() == [len(supports[i]) for i in indices]
    assert ids.tolist() == [token for i in indices for token in supports[i]]


@pytest.mark.parametrize("rank", [0, 1])
def test_mask_handles_tp_vocab_shards_and_cp_token_positions(rank):
    replay = TinkerSamplingMask([1, 4, 2], [0, 0, 0, 2, 3])
    local_logits = torch.zeros(3, 3)
    mask = build_tinker_sampling_mask(
        reference_local_mask, local_logits, replay, [0, 2, 3], tp_rank=rank
    )
    assert mask.tolist() == (
        [[True, True, True], [False, True, False], [False, False, True]]
        if rank == 0
        else [[True, True, True], [False, True, False], [False, False, False]]
    )


def test_replay_normalization_and_gradient_ignore_excluded_logits():
    logits = torch.tensor(
        [[2.0, 1.0, 0.0, 3.0], [1.0, 2.0, 3.0, 20.0]], requires_grad=True
    )
    replay = TinkerSamplingMask([0, 1, 2], [0, 0, 3])
    mask = build_tinker_sampling_mask(
        reference_local_mask, logits, replay, range(2), tp_rank=0
    )
    lp = (logits / 0.7).masked_fill(~mask, -torch.inf).log_softmax(-1)
    expected = (logits[1, :3] / 0.7).log_softmax(0)[2]
    assert lp[1, 2] == expected
    assert torch.equal(lp[0], (logits[0] / 0.7).log_softmax(0))
    (-lp[1, 2]).backward()
    assert logits.grad[1, 3] == 0
    assert torch.count_nonzero(logits.grad[0]) == 0
    # The excluded, high-logit token must not dilute the sampled distribution.
    assert lp[1, 2] > logits[1].log_softmax(0)[2]
