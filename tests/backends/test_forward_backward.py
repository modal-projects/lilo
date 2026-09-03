import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
from tinker import Datum, ModelInput

torch = pytest.importorskip("torch")

from lilo.backends import ForwardBatch, ForwardItem, LossFn  # noqa: E402
from lilo.backends.megatron_runtime.common.forward_backward import (  # noqa: E402
    _loss,
    _loss_config,
    _packed_seq_idx,
    _vocab_parallel_logprobs,
    log_packing,
    pack_microbatches,
    shard_microbatches,
)
from lilo.backends.megatron_runtime.common.forward_backward import (  # noqa: E402
    build_sequence_batches as build_microbatches,
)


def forward_batch(
    loss_name: LossFn,
    datum: Datum,
    loss_config: dict[str, float] | None = None,
) -> ForwardBatch:
    return ForwardBatch(
        items=(ForwardItem("model", (datum,)),),
        loss_fn=loss_name,
        loss_fn_config=loss_config or {},
    )


def training_datum(
    *,
    input_ids: tuple[int, ...],
    target_tokens: tuple[int, ...],
    weights: tuple[float, ...],
    loss_inputs: dict[str, tuple[float, ...]] | None = None,
) -> Datum:
    inputs = {key: list(values) for key, values in (loss_inputs or {}).items()}
    if target_tokens:
        inputs["target_tokens"] = list(target_tokens)
    if weights:
        inputs["weights"] = list(weights)
    return Datum(ModelInput.from_ints(list(input_ids)), inputs)


def test_importance_sampling_loss_and_gradient() -> None:
    datum = training_datum(
        input_ids=(1, 2, 3),
        target_tokens=(2, 3, 4),
        weights=(),
        loss_inputs={
            "logprobs": (0.0, -0.7, -0.3),
            "advantages": (0.0, 1.0, -1.0),
        },
    )
    batch = build_microbatches(
        forward_batch("importance_sampling", datum),
        {"model": 0},
        max_slots=1,
        max_seq_length=8,
    )[0]
    target_logprobs = torch.tensor([0.0, -0.5, -0.2], requires_grad=True)

    loss, token_count = _loss(target_logprobs, batch)
    loss.backward()

    assert loss.item() == pytest.approx(-(math.exp(0.2) - math.exp(0.1)))
    assert token_count.item() == 2
    assert target_logprobs.grad is not None
    assert target_logprobs.grad.tolist() == pytest.approx(
        [0.0, -math.exp(0.2), math.exp(0.1)]
    )


def test_importance_sampling_requires_per_token_inputs() -> None:
    datum = training_datum(
        input_ids=(1, 2),
        target_tokens=(2, 3),
        weights=(),
        loss_inputs={"logprobs": (-0.1,), "advantages": (1.0, 1.0)},
    )

    with pytest.raises(
        ValueError,
        match="target_tokens, logprobs, and advantages must match input_ids length",
    ):
        build_microbatches(
            forward_batch("importance_sampling", datum),
            {"model": 0},
            max_slots=1,
            max_seq_length=8,
        )


def test_cross_entropy_loss_is_unchanged() -> None:
    datum = training_datum(
        input_ids=(1, 2),
        target_tokens=(2, 3),
        weights=(1.0, 0.5),
    )
    batch = build_microbatches(
        forward_batch("cross_entropy", datum),
        {"model": 0},
        max_slots=1,
        max_seq_length=8,
    )[0]

    loss, token_count = _loss(torch.tensor([-0.2, -0.4]), batch)

    assert loss.item() == pytest.approx(0.4)
    assert token_count.item() == 2


def test_zero_weight_targets_still_produce_logprobs() -> None:
    datum = training_datum(
        input_ids=(1, 2),
        target_tokens=(2, 3),
        weights=(0.0, 0.0),
    )
    batch = build_microbatches(
        forward_batch("cross_entropy", datum),
        {"model": 0},
        max_slots=1,
        max_seq_length=8,
    )[0]

    assert batch["labels"].tolist() == [2, 3]
    assert batch["loss_mask"].tolist() == [0.0, 0.0]


def test_full_parameter_microbatches_do_not_include_adapter_routing() -> None:
    datum = training_datum(
        input_ids=(1, 2),
        target_tokens=(2, 3),
        weights=(1.0, 1.0),
    )

    batch = build_microbatches(
        forward_batch("cross_entropy", datum),
        None,
        max_slots=0,
        max_seq_length=8,
    )[0]

    assert "adapter_token_counts" not in batch


@pytest.mark.parametrize(
    ("loss_name", "target_logprobs", "advantages", "config", "expected"),
    [
        (
            "ppo",
            [math.log(2.0), math.log(0.5), math.log(1.1), math.log(0.9)],
            [1.0, -1.0, 1.0, -1.0],
            {},
            -0.6,
        ),
        (
            "cispo",
            [math.log(2.0), math.log(0.5)],
            [1.0, -1.0],
            {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
            -(1.2 * math.log(2.0) - 0.8 * math.log(0.5)),
        ),
        (
            "dro",
            [0.2, -0.2],
            [1.0, -1.0],
            {"beta": 0.2},
            -0.392,
        ),
    ],
)
def test_rl_loss_matches_tinker_formula(
    loss_name: str,
    target_logprobs: list[float],
    advantages: list[float],
    config: dict[str, float],
    expected: float,
) -> None:
    length = len(target_logprobs)
    datum = training_datum(
        input_ids=tuple(range(length)),
        target_tokens=tuple(range(1, length + 1)),
        weights=(),
        loss_inputs={
            "logprobs": (0.0,) * length,
            "advantages": tuple(advantages),
        },
    )
    batch = build_microbatches(
        forward_batch(loss_name, datum, config),
        {"model": 0},
        max_slots=1,
        max_seq_length=8,
    )[0]

    loss, token_count = _loss(torch.tensor(target_logprobs), batch)

    assert loss.item() == pytest.approx(expected)
    assert token_count.item() == length


@pytest.mark.parametrize(
    ("loss_name", "config", "message"),
    [
        ("cross_entropy", {"beta": 1.0}, "unsupported cross_entropy"),
        (
            "ppo",
            {"clip_low_threshold": 1.2, "clip_high_threshold": 0.8},
            "clip thresholds",
        ),
        ("cispo", {"clip_high_threshold": float("inf")}, "must be finite"),
        ("dro", {"beta": -0.1}, "beta must be non-negative"),
    ],
)
def test_loss_config_validation(
    loss_name: str,
    config: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _loss_config(loss_name, config)


def packed_batch(*lengths: int, model_id: str = "model-a") -> ForwardBatch:
    return ForwardBatch(
        items=(
            ForwardItem(
                model_id=model_id,
                data=tuple(
                    training_datum(
                        input_ids=tuple(range(length)),
                        target_tokens=tuple(range(length)),
                        weights=(1.0,) * length,
                    )
                    for length in lengths
                ),
            ),
        ),
        loss_fn="cross_entropy",
    )


def test_pack_microbatches_packs_arbitrary_lengths() -> None:
    unpacked = build_microbatches(
        packed_batch(5, 3, 3),
        {"model-a": 1},
        max_slots=2,
        max_seq_length=16,
    )

    packed = pack_microbatches(
        unpacked,
        max_tokens=8,
        pad_to_multiple=1,
    )

    assert [batch["tokens"].numel() for batch in packed] == [8, 3]
    assert packed[0]["cu_seqlens"].tolist() == [0, 5, 8]
    assert packed[0]["position_ids"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    assert packed[0]["output_indices"] == (0, 1)
    assert packed[0]["adapter_token_counts"].tolist() == [0, 8]


def test_pack_microbatches_keeps_max_length_sequence_intact() -> None:
    unpacked = build_microbatches(
        packed_batch(16, 3),
        {"model-a": 0},
        max_slots=1,
        max_seq_length=16,
    )

    packed = pack_microbatches(
        unpacked,
        max_tokens=4,
        pad_to_multiple=1,
    )

    assert [batch["tokens"].numel() for batch in packed] == [16, 3]
    assert packed[0]["cu_seqlens"].tolist() == [0, 16]


def test_pack_microbatches_tracks_cp_alignment_metadata() -> None:
    unpacked = build_microbatches(
        packed_batch(5, 3),
        None,
        max_slots=0,
        max_seq_length=16,
    )

    (packed,) = pack_microbatches(
        unpacked,
        max_tokens=16,
        pad_to_multiple=4,
    )

    assert packed["tokens"].numel() == 12
    assert packed["cu_seqlens"].tolist() == [0, 8, 12]
    assert "cu_seqlens_unpadded" not in packed
    assert packed["max_seqlen"] == 8
    assert packed["starts"] == (0, 8)
    assert packed["original_lengths"] == (5, 3)
    assert packed["output_indices"] == (0, 1)
    assert packed["position_ids"].tolist() == [
        0,
        1,
        2,
        3,
        4,
        0,
        0,
        0,
        0,
        1,
        2,
        0,
    ]
    assert packed["labels"].tolist() == [
        0,
        1,
        2,
        3,
        4,
        -100,
        -100,
        -100,
        0,
        1,
        2,
        -100,
    ]
    assert packed["loss_mask"].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert _packed_seq_idx(
        torch.tensor([0, 7, 8, 11]),
        packed["cu_seqlens"],
    ).tolist() == [[0, 0, 1, 1]]


def test_pack_microbatches_aligns_tp4_cp2_sequence_parallel_tokens() -> None:
    unpacked = build_microbatches(
        packed_batch(5, 3),
        None,
        max_slots=0,
        max_seq_length=16,
    )

    (packed,) = pack_microbatches(
        unpacked,
        max_tokens=16,
        pad_to_multiple=4,
        total_pad_to_multiple=8,
    )

    assert packed["tokens"].numel() == 16
    assert packed["cu_seqlens"].tolist() == [0, 8, 12, 16]
    assert "cu_seqlens_unpadded" not in packed
    assert packed["max_seqlen"] == 8


@pytest.mark.parametrize("length", range(1, 5))
def test_pack_microbatches_tracks_short_cp_sequences(length: int) -> None:
    (packed,) = pack_microbatches(
        build_microbatches(
            packed_batch(length),
            None,
            max_slots=0,
            max_seq_length=4,
        ),
        max_tokens=4,
        pad_to_multiple=4,
    )

    assert packed["cu_seqlens"].tolist() == [0, 4]
    assert packed["max_seqlen"] == 4
    assert "cu_seqlens_unpadded" not in packed
    assert packed["position_ids"].tolist() == [
        *range(length),
        *([0] * (4 - length)),
    ]
    for indices in (torch.tensor([0, 3]), torch.tensor([1, 2])):
        assert packed["loss_mask"].index_select(0, indices).tolist() == [
            float(index < length) for index in indices
        ]


def test_shard_microbatches_marks_packed_padding_dummy() -> None:
    packed = pack_microbatches(
        build_microbatches(
            packed_batch(3, 2),
            {"model-a": 0},
            max_slots=1,
            max_seq_length=8,
        ),
        max_tokens=8,
        pad_to_multiple=1,
    )

    local = shard_microbatches(
        packed,
        data_parallel_rank=1,
        data_parallel_size=2,
    )

    assert local[0]["output_indices"] == (-1, -1)
    assert local[0]["is_dummy"] == (True, True)
    assert local[0]["loss_mask"].count_nonzero().item() == 0


@pytest.mark.parametrize(
    ("labels", "expected_grad"),
    [
        ((1, -100), [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]]),
        ((-100, -100), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_vocab_parallel_logprobs_preserves_packed_kernel_dtype(
    monkeypatch,
    labels: tuple[int, int],
    expected_grad: list[list[float]],
) -> None:
    observed = {}

    def fused_cross_entropy(logits, labels, group):
        observed.update(logits=logits, labels=labels, group=group)
        return logits.sum(dim=-1)

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    fusions = ModuleType("megatron.core.fusions")
    fused = ModuleType("megatron.core.fusions.fused_cross_entropy")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: "tp")
    fused.fused_vocab_parallel_cross_entropy = fused_cross_entropy
    megatron.core = core
    core.fusions = fusions
    fusions.fused_cross_entropy = fused
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.fusions", fusions)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.fusions.fused_cross_entropy",
        fused,
    )
    logits = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    logprobs = _vocab_parallel_logprobs(logits, torch.tensor(labels))
    assert logprobs.grad_fn is not None
    logprobs.sum().backward()

    assert observed["logits"].dtype == torch.bfloat16
    assert observed["logits"].data_ptr() == logits.data_ptr()
    assert observed["labels"].reshape(-1).tolist() == [
        max(label, 0) for label in labels
    ]
    assert logprobs.dtype == torch.float32
    assert logits.grad is not None
    assert logits.grad.reshape(2, 3).tolist() == expected_grad


def test_packing_metrics_are_rank_zero_and_reduction_compatible(capsys) -> None:
    request = packed_batch(5, 3)
    packed = pack_microbatches(
        build_microbatches(
            request,
            {"model-a": 0},
            max_slots=1,
            max_seq_length=8,
        ),
        max_tokens=16,
        pad_to_multiple=4,
    )

    log_packing(
        request,
        packed,
        input_sequences=2,
        raw_tokens=8,
        token_capacity=16,
        rank=1,
    )
    metrics = log_packing(
        request,
        packed,
        input_sequences=2,
        raw_tokens=8,
        token_capacity=16,
        rank=0,
    )

    assert capsys.readouterr().out.count("packed forward_backward") == 1
    assert metrics == {
        "packing_input_sequences:sum": 2.0,
        "packing_raw_tokens:sum": 8.0,
        "packing_padded_tokens:sum": 12.0,
        "packing_bins:sum": 1.0,
        "packing_token_capacity:max": 16.0,
        "packing_padding_fraction:mean": 1 / 3,
        "packing_utilization:mean": 0.5,
    }
