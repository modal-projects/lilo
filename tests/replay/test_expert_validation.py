"""Bad routes must fail in the controller without poisoning a shared GPU trainer."""

import asyncio
from itertools import count
from types import SimpleNamespace

import pytest
from tinker import types

from lilo.backends import ForwardBatch, ForwardItem
from lilo.backends.miles_runtime.data import prepare_batch
from lilo.backends.miles_runtime.replay_data import install_bridge_replay
from lilo.backends.miles_runtime.runtime import MilesRuntime


@pytest.fixture
def runtime():
    runtime = MilesRuntime.__new__(MilesRuntime)
    runtime._args = SimpleNamespace(num_layers=2, num_experts=8, moe_router_topk=2)
    runtime._unit_ids = count(1)
    runtime._failure = None
    runtime._closed = False
    runtime._call = asyncio.run
    calls = []

    async def forward(*args):
        calls.append(args)
        return [{"loss": 0.0, "logprobs": [0.0, 0.0]}]

    runtime._bridge = SimpleNamespace(forward_only=forward, forward_backward=forward)
    return runtime, calls


def _rows(values, shape=(2, 2, 2), *, alignment=1):
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": [2, 3],
            "weights": [1.0, 1.0],
            "routed_experts": types.TensorData(
                data=values, dtype="int64", shape=list(shape)
            ),
        },
    )
    return prepare_batch(
        ForwardBatch(items=(ForwardItem("model", (datum,)),), loss_fn="cross_entropy"),
        {"model": 0},
        sequence_alignment=alignment,
    ).slot_rows


def _forward(runtime, rows, forward_only=False):
    return runtime.forward_backward(
        rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=forward_only
    )


@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize(
    "values,shape,error",
    [
        ([8, 1] * 4, (2, 2, 2), "expert count"),
        ([-1, 0] * 4, (2, 2, 2), "entire top-k row"),
        ([0, 0] * 4, (2, 2, 2), "distinct"),
        ([0, 1] * 6, (2, 3, 2), "2 layers"),
        ([0] * 4, (2, 2, 1), "2 experts per token"),
    ],
)
def test_invalid_routes_never_reach_trainer_and_next_request_still_works(
    runtime, forward_only, values, shape, error
):
    runtime, calls = runtime
    with pytest.raises(ValueError, match=error):
        _forward(runtime, _rows(values, shape), forward_only)
    assert calls == []
    assert runtime._failure is None

    _forward(runtime, _rows([0, 7] * 4), forward_only)
    assert len(calls) == 1


def test_dense_model_rejects_routes_before_dispatch(runtime):
    runtime, calls = runtime
    runtime._args.num_experts = None
    with pytest.raises(ValueError, match="MoE model"):
        _forward(runtime, _rows([0, 1] * 4))
    assert calls == []
    assert runtime._failure is None


def test_parallel_padding_rows_are_allowed(runtime):
    runtime, calls = runtime
    rows = _rows([0, 7] * 4, alignment=8)
    _forward(runtime, rows)
    assert len(calls) == 1
    assert rows[0][1]["routed_experts"]["values"][-4:] == [-1] * 4


def test_bridge_installation_is_idempotent():
    bridge = SimpleNamespace(
        _build_train_data=lambda rows: {"tokens": [row["tokens"] for _, row in rows]}
    )
    install_bridge_replay(bridge)
    build = bridge._build_train_data
    install_bridge_replay(bridge)
    assert bridge._build_train_data is build
    batch = build(_rows([0, 7] * 4))
    routes = batch["rollout_routed_experts"][0]
    assert str(routes.dtype) == "int32"
    assert routes.shape == (2, 2, 2)
    assert routes.reshape(-1).tolist() == [0, 7] * 4
