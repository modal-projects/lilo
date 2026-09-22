from types import SimpleNamespace

import pytest
from tinker import types

from lilo.backends import ForwardBatch, ForwardItem
from lilo.backends.megatron_runtime.common.forward_backward import (
    build_sequence_batches,
)
from lilo.backends.miles_runtime.data import prepare_batch
from lilo.engine import OperationKind
from lilo.engine.server import Engine
from tests.backends.test_miles import _backend, _spec
from tests.replay.test_roundtrip import sample_response


def replay_datum():
    replay = sample_response().sequences[0].replay
    return types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 5]),
        loss_fn_inputs={
            "target_tokens": [2, 3, 5, 6],
            "advantages": [0.0, 0.0, 1.0, 1.0],
            **replay.training_inputs(2, num_layers=2, experts_per_token=1),
        },
    )


def test_context_parallel_alignment_preserves_routes_and_supports():
    prepared = prepare_batch(
        ForwardBatch(items=(ForwardItem("m", (replay_datum(),)),), loss_fn="ppo"),
        {"m": 0},
        sequence_alignment=8,
    )
    row = prepared.slot_rows[0][1]
    assert prepared.target_lengths == (4,)
    assert row["tokens"] == [1, 2, 3, 5, 6, 6, 6, 6]
    assert row["target_len"] == 7
    assert row["routed_experts"]["shape"] == [7, 2, 1]
    assert row["routed_experts"]["values"] == [*range(8), *([-1] * 6)]
    assert row["sampling_mask_offsets"] == [0, 0, 0, 2, 5, 5, 5, 5]
    assert row["sampling_mask_ids"] == [5, 7, 6, 8, 9]
    assert row["advantages"] == [0, 0, 1, 1, 0, 0, 0]


def test_router_requests_not_coalesced_with_ordinary_requests():
    datum = replay_datum()
    plain = types.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            k: v for k, v in datum.loss_fn_inputs.items() if k != "routed_experts"
        },
    )

    def key(d):
        return Engine._forward_backward_batch_key(
            SimpleNamespace(
                kind=OperationKind.FORWARD_BACKWARD,
                payload=types.ForwardBackwardInput(data=[d], loss_fn="ppo"),
            )
        )

    assert key(datum) != key(plain)
    assert key(datum) == key(replay_datum())


def test_router_requires_enabled_trainer(tmp_path):
    backend = _backend(tmp_path)
    backend.accept_model("m", _spec())
    with pytest.raises(ValueError, match="use-rollout-routing-replay"):
        backend.forward_backward(
            ForwardBatch(
                items=(ForwardItem("m", (replay_datum(),)),),
                loss_fn="ppo",
                loss_fn_config={"sampling_temperature": 0.7},
            )
        )
    assert not any(c[0] == "forward_backward" for c in backend.runtime.calls)


@pytest.mark.parametrize("temperature", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_mask_requires_explicit_positive_temperature(tmp_path, temperature):
    backend = _backend(tmp_path)
    backend.accept_model("m", _spec())
    datum = replay_datum()
    datum = types.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            k: v for k, v in datum.loss_fn_inputs.items() if k != "routed_experts"
        },
    )
    config = {} if temperature is None else {"sampling_temperature": temperature}
    with pytest.raises(ValueError, match="sampling_temperature"):
        backend.forward_backward(
            ForwardBatch(
                items=(ForwardItem("m", (datum,)),),
                loss_fn="ppo",
                loss_fn_config=config,
            )
        )
    assert not any(c[0] == "forward_backward" for c in backend.runtime.calls)


def test_unsupported_backend_rejects_replay():
    with pytest.raises(ValueError, match="Miles LoRA backend"):
        build_sequence_batches(
            ForwardBatch(items=(ForwardItem("m", (replay_datum(),)),), loss_fn="ppo"),
            None,
            max_slots=None,
            max_seq_length=32,
        )
